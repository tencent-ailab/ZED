import math
import argparse
from util import logger, random, np, DistributedSampler, os, logging, ZED, ZEDDataLoader, maven_e_type_to_definition, \
    json, evaluation
import torch.nn
from transformers import (BertConfig, BertModel, BertTokenizer, RobertaConfig, RobertaModel, RobertaTokenizer,
                          AlbertConfig, AlbertModel, AlbertTokenizer)
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler)
from tqdm import tqdm, trange

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tmp_dataloader, all_paired_data, update_data=False):
    """ Train the model """
    print('Start Training...')
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_pct is None:
        scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=math.floor(args.warmup_pct * t_total),
                                         num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    counter = 0
    for _ in train_iterator:
        counter += 1
        if update_data and counter > 1:
            if args.train_source in ['warm', 'maven']:
                if args.hard_negative_sampling == 'no_hard':
                    tmp_p = 0
                elif args.hard_negative_sampling == 'all_hard':
                    tmp_p = 1
                else:
                    raise NotImplementedError
                new_data = tmp_dataloader.tensorize_aligned_examples_strong_negative_sampling(
                    input_pairs=all_paired_data, model=model,
                    number_training_example=args.num_train_examples,
                    number_negative_examples=2,
                    p=tmp_p)
            else:
                new_data = tmp_dataloader.tensorize_aligned_examples(input_pairs=all_paired_data,
                                                                     candidate_glosses=None,
                                                                     number_training_example=args.num_train_examples,
                                                                     first_tensorize=False, number_negative_examples=2,
                                                                     cache_path=args.cache_path)

            train_sampler = RandomSampler(new_data) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(new_data, sampler=train_sampler, batch_size=args.train_batch_size)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0],
                              mininterval=1, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'input_masks': batch[1],
                      'output_masks': batch[2],
                      'definition_ids': batch[3],
                      'definition_input_masks': batch[4],
                      'definition_output_masks': batch[5],
                      'labels': batch[6]}

            loss = model(**inputs)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    recent_loss = (tr_loss - logging_loss) / args.logging_steps
                    logging_loss = tr_loss

                    logger.info("Loss: %s", recent_loss)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    output_dir = args.output_dir
    if args.n_gpu > 1:
        torch.save(model.module.state_dict(), os.path.join(output_dir, 'parameters.bin'))
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, 'parameters.bin'))
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", output_dir)

    return global_step, tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()

    # Important parameters
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--default_model", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str,
                        help="Path to pre-trained model or shortcut names")
    parser.add_argument("--output_dir", default='output/pretrain', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--hard_negative_sampling", default='all_hard', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--batch_negative', action='store_true')
    parser.add_argument("--train_source", default='pretrain', type=str,
                        help="Select which data to train")
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument("--cache_path", default='cache/pretrain', type=str,
                        help="select the path towards to the cache")

    # Other parameters
    parser.add_argument("--logit_method", default='cosine', type=str,
                        help="choose which logit computation method to use: multiply, NN, or cosine")
    parser.add_argument("--def_representation", default='mean', type=str,
                        help="choose which definition representation to use: head or mean")
    parser.add_argument("--loss_function", default='ranking', type=str,
                        help="choose which loss function to use")
    parser.add_argument("--encoding_mode", default='joint', type=str,
                        help="how many encoding language models to use")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--context_max_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--definition_max_length", default=32, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--total_max_length", default=160, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_prediction", action='store_true',
                        help="Whether to run prediction on the test set. (Training will not be executed.)")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--run_on_test', action='store_true')
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--bd_weight", default=0.01, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--num_train_examples', type=int, default=-1,
                        help="Number of training examples.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_pct", default=0.1, type=float,
                        help="Linear warmup over warmup_pct*total_steps.")
    parser.add_argument("--train_variance", action='store_true',
                        help="Whether to fixed the mean embedding.")
    parser.add_argument("--output_size", default=768, type=int,
                        help="output embedding size")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=300,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()
    # setup(args.local_rank, 1)
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.default_model, do_lower_case=True)
    config = config_class.from_pretrained(args.default_model)

    context_encoder_model = model_class.from_pretrained(args.default_model,
                                                        from_tf=bool('.ckpt' in args.default_model),
                                                        config=config)
    def_encoder_model = model_class.from_pretrained(args.default_model,
                                                    from_tf=bool('.ckpt' in args.default_model),
                                                    config=config)
    print(args.model_name_or_path)
    if args.model_name_or_path == args.default_model:
        print('We initialize a new model from:', args.model_name_or_path)
        model = ZED(config, context_encoder_model, def_encoder_model, args)
    else:
        print('We retrieve a trained model from:', args.model_name_or_path)
        model = ZED(config, context_encoder_model, def_encoder_model, args)
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'parameters.bin')))
        except AttributeError:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'parameters.bin')))
            model = model.module

    if args.local_rank == 0:
        torch.distributed.barrier()
    # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    data_loader = ZEDDataLoader(args, tokenizer)

    maven_glosses = list()

    for tmp_maven_e_type in maven_e_type_to_definition:
        maven_glosses.append(maven_e_type_to_definition[tmp_maven_e_type])

    train_dataset = list()
    maven_train = list()
    with open('maven_data/train.jsonl', 'r') as f:
        for line in f:
            maven_train.append(json.loads(line))
    for tmp_instance in maven_train:
        for tmp_e in tmp_instance['events']:
            for tmp_mention in tmp_e['mention']:
                tokens = tmp_instance['content'][tmp_mention['sent_id']]['tokens']
                target_position = tmp_mention['offset']
                train_dataset.append({'tokens': tokens, 'target_position': target_position, 'label': tmp_e['type'],
                                      'gloss': maven_e_type_to_definition[tmp_e['type']]})
        for tmp_c in tmp_instance['negative_triggers']:
            tokens = tmp_instance['content'][tmp_c['sent_id']]['tokens']
            target_position = tmp_c['offset']
            train_dataset.append({'tokens': tokens, 'target_position': target_position, 'label': 'NA', 'gloss': 'NA'})

    eval_dataset = list()
    maven_valid = list()
    with open('maven_data/valid.jsonl', 'r') as f:
        for line in f:
            maven_valid.append(json.loads(line))

    for tmp_instance in maven_valid:
        for tmp_e in tmp_instance['events']:
            for tmp_mention in tmp_e['mention']:
                tokens = tmp_instance['content'][tmp_mention['sent_id']]['tokens']
                target_position = tmp_mention['offset']
                eval_dataset.append({'tokens': tokens, 'target_position': target_position, 'label': tmp_e['type'],
                                     'gloss': maven_e_type_to_definition[tmp_e['type']]})
        for tmp_c in tmp_instance['negative_triggers']:
            tokens = tmp_instance['content'][tmp_c['sent_id']]['tokens']
            target_position = tmp_c['offset']
            eval_dataset.append({'tokens': tokens, 'target_position': target_position, 'label': 'NA', 'gloss': 'NA'})

    maven_eval_package = {'eval_dataset': eval_dataset, 'type2definitions': maven_e_type_to_definition}

    if args.do_train:
        if args.train_source == 'pretrain':
            with open('data/pre_trained_glosses.json', 'r') as f:
                selected_glosses = json.load(f)
            with open('data/pre_trained_pairs.json', 'r') as f:
                selected_paired_data = json.load(f)
        elif args.train_source == 'warm':
            with open('data/warm_glosses.json', 'r') as f:
                selected_glosses = json.load(f)
            with open('data/warm_pairs.json', 'r') as f:
                selected_paired_data = json.load(f)
        elif args.train_source == 'maven':
            selected_paired_data = train_dataset
            selected_glosses = maven_glosses
            selected_glosses.append('NA')
            selected_glosses = sorted(list(set(selected_glosses)))
        else:
            raise NotImplementedError
        training_dataset = data_loader.tensorize_aligned_examples(input_pairs=selected_paired_data,
                                                                  candidate_glosses=selected_glosses,
                                                                  number_training_example=args.num_train_examples,
                                                                  first_tensorize=True, number_negative_examples=2,
                                                                  cache_path=args.cache_path)
        train(args, training_dataset, model, data_loader, selected_paired_data, update_data=True)

    if args.do_eval:
        evaluation(args, data_loader, maven_eval_package, model, device)


if __name__ == "__main__":
    main()
