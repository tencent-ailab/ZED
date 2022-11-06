import json
import os
import random
import logging
import numpy as np
import torch.nn as nn
import faiss
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


maven_e_type_to_definition = dict()
maven_e_type_to_definition['Achieve'] = 'to gain with effort'
maven_e_type_to_definition['Action'] = 'direct the course of; manage or control'
maven_e_type_to_definition['Adducing'] = 'make reference to'
maven_e_type_to_definition['Agree_or_refuse_to_act'] = 'show unwillingness towards'
maven_e_type_to_definition['Aiming'] = 'intend (something) to move towards a certain goal'
maven_e_type_to_definition[
    'Arranging'] = 'cause to be structured or ordered or operating according to some principle or idea'
maven_e_type_to_definition['Arrest'] = 'take into custody'
maven_e_type_to_definition['Arriving'] = 'reach a destination, either real or abstract'
maven_e_type_to_definition['Assistance'] = 'give help or assistance; be of service'
maven_e_type_to_definition['Attack'] = 'launch an attack or assault on; begin hostilities or start warfare with'
maven_e_type_to_definition['Award'] = 'give, especially as an honor or reward'
maven_e_type_to_definition['Bearing_arms'] = 'prepare oneself for a military confrontation'
maven_e_type_to_definition['Becoming'] = 'enter or assume a certain state or condition'
maven_e_type_to_definition['Becoming_a_member'] = 'become part of; become a member of a group or organization'
maven_e_type_to_definition['Being_in_operation'] = 'direct or control; projects, businesses, etc.'
maven_e_type_to_definition['Besieging'] = 'surround so as to force to give up'
maven_e_type_to_definition['Bodily_harm'] = 'cause injuries or bodily harm to'
maven_e_type_to_definition['Body_movement'] = 'take hold of something and move it to a different location'
maven_e_type_to_definition['Breathing'] = 'draw air into, and expel out of, the lungs'
maven_e_type_to_definition['Bringing'] = 'take something or somebody with oneself somewhere'
maven_e_type_to_definition['Building'] = 'set up or found'
maven_e_type_to_definition[
    'Carry_goods'] = "move while supporting, either in a vehicle or in one's hands or on one's body"
maven_e_type_to_definition[
    'Catastrophe'] = 'a violent weather condition with winds 64-72 knots ' \
                     '(11 on the Beaufort scale) and precipitation and thunder and lightning'
maven_e_type_to_definition['Causation'] = 'give rise to; cause to happen or occur, not always intentionally'
maven_e_type_to_definition['Cause_change_of_position_on_a_scale'] = 'become bigger or greater in amount'
maven_e_type_to_definition['Cause_change_of_strength'] = 'lessen the strength of'
maven_e_type_to_definition['Cause_to_amalgamate'] = 'have or possess in combination'
maven_e_type_to_definition['Cause_to_be_included'] = 'have as a part, be made up out of'
maven_e_type_to_definition['Cause_to_make_progress'] = 'move forward, also in the metaphorical sense'
maven_e_type_to_definition['Change'] = 'cause to change; make different; cause a transformation'
maven_e_type_to_definition['Change_event_time'] = 'postpone indefinitely or annul something that was scheduled'
maven_e_type_to_definition[
    'Change_of_leadership'] = 'a drastic and far-reaching change in ways of thinking and behaving'
maven_e_type_to_definition['Change_sentiment'] = 'undergo an emotional sensation or be in a particular state of mind'
maven_e_type_to_definition['Change_tool'] = 'exchange or replace with another, usually of the same kind or category'
maven_e_type_to_definition['Check'] = 'establish or strengthen as with new evidence or facts'
maven_e_type_to_definition['Choosing'] = 'select by a vote for an office or membership'
maven_e_type_to_definition['Collaboration'] = 'an associate who provides cooperation or assistance'
maven_e_type_to_definition['Come_together'] = 'come together'
maven_e_type_to_definition['Coming_to_be'] = 'come to pass'
maven_e_type_to_definition['Coming_to_believe'] = 'decide by reasoning; draw or come to a conclusion'
maven_e_type_to_definition['Commerce_buy'] = 'obtain by purchase; acquire by means of a financial transaction'
maven_e_type_to_definition['Commerce_pay'] = 'give money, usually in exchange for goods or services'
maven_e_type_to_definition['Commerce_sell'] = 'exchange or deliver for money or its equivalent'
maven_e_type_to_definition['Commitment'] = 'perform an act, usually with a negative connotation'
maven_e_type_to_definition['Committing_crime'] = 'kill intentionally and with premeditation'
maven_e_type_to_definition['Communication'] = 'express in words'
maven_e_type_to_definition[
    'Competition'] = 'a sporting competition in which contestants play a series of games to decide the winner'
maven_e_type_to_definition['Confronting_problem'] = 'deal with (something unpleasant) head on'
maven_e_type_to_definition['Connect'] = 'connect, fasten, or put together two or more pieces'
maven_e_type_to_definition['Conquering'] = 'succeed in representing or expressing something intangible'
maven_e_type_to_definition['Containing'] = 'include or contain; have as a component'
maven_e_type_to_definition['Control'] = 'exercise authoritative control or power over'
maven_e_type_to_definition['Convincing'] = 'make a proposal, declare a plan for something'
maven_e_type_to_definition['Cost'] = 'be priced at'
maven_e_type_to_definition['Create_artwork'] = 'make a film or photograph of something'
maven_e_type_to_definition['Creating'] = 'bring forth or yield'
maven_e_type_to_definition['Criminal_investigation'] = 'conduct an inquiry or investigation of'
maven_e_type_to_definition[
    'Cure'] = 'subject to a process or treatment, with the aim of readying ' \
              'for some purpose, improving, or remedying a condition'
maven_e_type_to_definition['Damaging'] = 'inflict damage upon'
maven_e_type_to_definition[
    'Death'] = 'pass from physical life and lose all bodily attributes and functions necessary to sustain life'
maven_e_type_to_definition['Deciding'] = 'reach, make, or come to a decision about something'
maven_e_type_to_definition['Defending'] = 'secure and keep for possible future use or application'
maven_e_type_to_definition['Departing'] = 'go away from a place'
maven_e_type_to_definition['Destroying'] = 'do away with, cause the destruction or undoing of'
maven_e_type_to_definition['Dispersal'] = 'to cause to separate and go in different directions'
maven_e_type_to_definition[
    'Earnings_and_losses'] = 'fail to keep or to maintain; cease to have, either physically or in an abstract sense'
maven_e_type_to_definition['Education_teaching'] = 'create by training and teaching'
maven_e_type_to_definition[
    'Emergency'] = 'a sudden unforeseen crisis (usually involving danger) that requires immediate action'
maven_e_type_to_definition['Employment'] = 'engage or hire for work'
maven_e_type_to_definition['Emptying'] = 'empty completely; destroy the inside of'
maven_e_type_to_definition['Escaping'] = 'run away quickly'
maven_e_type_to_definition['Exchange'] = 'give to, and receive from, one another'
maven_e_type_to_definition['Expansion'] = 'extend in one or more directions'
maven_e_type_to_definition['Expend_resource'] = 'pass time in a specific way'
maven_e_type_to_definition['Expressing_publicly'] = 'state emphatically and authoritatively'
maven_e_type_to_definition['Extradition'] = 'hand over to the authorities of another country'
maven_e_type_to_definition['Filling'] = 'make full, also in a metaphorical sense'
maven_e_type_to_definition['Forming_relationships'] = 'discontinue an association or relation; go different ways'
maven_e_type_to_definition[
    'GetReady'] = 'make ready or suitable or equip in advance for a particular purpose or for some use, event, etc'
maven_e_type_to_definition['Getting'] = 'gain points in a game'
maven_e_type_to_definition['GiveUp'] = 'give up, such as power, as of monarchs and emperors, or duties and obligations'
maven_e_type_to_definition['Giving'] = 'cause to have, in the abstract sense or physical sense'
maven_e_type_to_definition['Having_or_lacking_access'] = 'reach or gain access to'
maven_e_type_to_definition['Hiding_objects'] = 'prevent from being seen or discovered'
maven_e_type_to_definition['Hindering'] = 'throw into disorder'
maven_e_type_to_definition['Hold'] = 'keep in a certain state, position, or activity; e.g.,'
maven_e_type_to_definition['Hostile_encounter'] = 'a hostile disagreement face-to-face'
maven_e_type_to_definition['Imposing_obligation'] = 'require as useful, just, or proper'
maven_e_type_to_definition['Incident'] = 'a single distinct event'
maven_e_type_to_definition['Influence'] = 'have an effect upon'
maven_e_type_to_definition['Ingestion'] = 'inhale and exhale smoke from cigarettes, cigars, pipes'
maven_e_type_to_definition['Institutionalization'] = 'admit into a hospital'
maven_e_type_to_definition['Judgment_communication'] = 'bring an accusation against; level a charge against'
maven_e_type_to_definition['Justifying'] = 'show to be reasonable or provide adequate ground for'
maven_e_type_to_definition[
    'Kidnapping'] = 'take away to an undisclosed location against their will and usually in order to extract a ransom'
maven_e_type_to_definition['Killing'] = 'cause to die; put to death, usually intentionally or knowingly'
maven_e_type_to_definition[
    'Know'] = 'be cognizant or aware of a fact or a specific piece of ' \
              'information; possess knowledge or information about'
maven_e_type_to_definition['Labeling'] = 'mark with a brand or trademark'
maven_e_type_to_definition['Legal_rulings'] = 'pronounce a sentence on (somebody) in a court of law'
maven_e_type_to_definition['Legality'] = 'prohibited by law or by official or accepted rules'
maven_e_type_to_definition[
    'Lighting'] = 'abrupt electric discharge from cloud to cloud or from ' \
                  'cloud to earth accompanied by the emission of light'
maven_e_type_to_definition['Limiting'] = 'restrict or confine,'
maven_e_type_to_definition['Manufacturing'] = 'make or cause to be or to become'
maven_e_type_to_definition['Military_operation'] = 'urge or force (a person) to an action; constrain or motivate'
maven_e_type_to_definition['Motion'] = 'change location; move, travel, or proceed, also metaphorically'
maven_e_type_to_definition['Motion_directional'] = 'go across or through'
maven_e_type_to_definition['Name_conferral'] = 'assign a specified (usually proper) proper name to'
maven_e_type_to_definition['Openness'] = 'cause to open or to become open'
maven_e_type_to_definition['Participation'] = 'become a participant; be involved in'
maven_e_type_to_definition['Patrolling'] = 'accompany as an escort'
maven_e_type_to_definition['Perception_active'] = 'perceive by sight or have the power to perceive by sight'
maven_e_type_to_definition['Placing'] = 'put into a certain place or abstract location'
maven_e_type_to_definition['Practice'] = 'carry out or practice; as of jobs and professions'
maven_e_type_to_definition['Presence'] = 'come to pass'
maven_e_type_to_definition['Preserving'] = 'keep in a certain state, position, or activity; e.g.,'
maven_e_type_to_definition[
    'Preventing_or_letting'] = 'make it possible through a specific action or lack of action for something to happen'
maven_e_type_to_definition['Prison'] = 'putting someone in prison or in jail as lawful punishment'
maven_e_type_to_definition['Process_end'] = 'bring to an end or halt'
maven_e_type_to_definition['Process_start'] = 'take the first step or steps in carrying out an action'
maven_e_type_to_definition['Protest'] = 'express opposition through action or words'
maven_e_type_to_definition['Publishing'] = 'prepare and issue for public distribution or sale'
maven_e_type_to_definition['Quarreling'] = 'have a disagreement over something'
maven_e_type_to_definition['Ratification'] = 'approve and express assent, responsibility, or obligation'
maven_e_type_to_definition['Receiving'] = 'get something; come into possession of'
maven_e_type_to_definition['Recording'] = 'make a record of; set down in permanent form'
maven_e_type_to_definition['Recovering'] = 'get over an illness or shock'
maven_e_type_to_definition[
    'Reforming_a_system'] = 'make changes for improvement in order to remove abuse and injustices'
maven_e_type_to_definition['Releasing'] = 'grant freedom to; free from confinement'
maven_e_type_to_definition[
    'Removing'] = 'remove something concrete, as by lifting, pushing, or taking off, or remove something abstract'
maven_e_type_to_definition['Renting'] = 'hold under a lease or rental agreement; of goods and services'
maven_e_type_to_definition['Reporting'] = 'to give an account or representation of in words'
maven_e_type_to_definition['Request'] = 'give instructions to or direct somebody to do something with authority'
maven_e_type_to_definition['Rescuing'] = 'free from harm or evil'
maven_e_type_to_definition['Research'] = 'determine the presence or properties of (a substance)'
maven_e_type_to_definition['Resolve_problem'] = 'find the solution'
maven_e_type_to_definition['Response'] = 'show a response or a reaction to something'
maven_e_type_to_definition[
    'Reveal_secret'] = 'make known to the public information that was previously ' \
                       'known only to a few people or that was meant to be kept a secret'
maven_e_type_to_definition['Revenge'] = 'take revenge for a perceived wrong'
maven_e_type_to_definition['Rewards_and_punishments'] = 'pronounce a sentence on (somebody) in a court of law'
maven_e_type_to_definition['Risk'] = 'expose to a chance of loss or damage'
maven_e_type_to_definition['Rite'] = 'make fit for use'
maven_e_type_to_definition['Robbery'] = 'take illegally; of intellectual property'
maven_e_type_to_definition['Scouring'] = 'sweep across or over'
maven_e_type_to_definition['Scrutiny'] = 'try to locate or discover, or try to establish the existence of'
maven_e_type_to_definition['Self_motion'] = 'march in a procession'
maven_e_type_to_definition['Sending'] = 'cause to go somewhere'
maven_e_type_to_definition['Sign_agreement'] = "mark with one's signature; write one's name (on)"
maven_e_type_to_definition['Social_event'] = 'a day or period of time set aside for feasting and celebration'
maven_e_type_to_definition['Statement'] = 'assert or affirm strongly; state to be true or existing'
maven_e_type_to_definition['Submitting_documents'] = 'record in a public office or in a court of law'
maven_e_type_to_definition['Supply'] = 'give something useful or necessary to'
maven_e_type_to_definition['Supporting'] = 'give moral or psychological support, aid, or courage to'
maven_e_type_to_definition['Surrendering'] = 'give up or agree to forgo to the power or possession of another'
maven_e_type_to_definition['Surrounding'] = 'extend on all sides of simultaneously; encircle'
maven_e_type_to_definition['Suspicion'] = 'imagine to be the case or true or probable'
maven_e_type_to_definition['Telling'] = 'express in words'
maven_e_type_to_definition['Temporary_stay'] = 'stay the same; remain in a certain state'
maven_e_type_to_definition[
    'Terrorism'] = 'the calculated use of violence (or the threat of violence) ' \
                   'against civilians in order to attain goals that are political or ' \
                   'religious or ideological in nature; this is done through ' \
                   'intimidation or coercion or instilling fear'
maven_e_type_to_definition['Testing'] = 'put to the test, as for its quality, or give experimental use to'
maven_e_type_to_definition['Theft'] = "take without the owner's consent"
maven_e_type_to_definition['Traveling'] = 'travel around something'
maven_e_type_to_definition['Use_firearm'] = 'hit with a missile from a weapon'
maven_e_type_to_definition[
    'Using'] = 'put into service; make work or employ for a particular purpose or for its inherent or natural purpose'
maven_e_type_to_definition['Violence'] = 'an act of aggression (as one against a person who resists)'
maven_e_type_to_definition['Vocalizations'] = 'deliver by singing'
maven_e_type_to_definition['Warning'] = 'notify of danger, potential harm, or risk'
maven_e_type_to_definition['Wearing'] = 'be dressed in'
maven_e_type_to_definition['Writing'] = 'communicate or express by writing'


def tokenized_to_origin_span(token_list, text):
    token_span = []
    pointer = 0
    for token in token_list:
        if len(token) == 0:
            token_span.append((pointer, pointer))
            continue
        while True:
            try:
                if token[0] == text[pointer]:
                    start = pointer
                    end = start + len(token) - 1
                    pointer = end + 1
                    break
                else:
                    pointer += 1
            except IndexError:
                print(token_list)
                print(text)
                print(token)
                print(pointer)
                break
        token_span.append((start, end))
    return token_span


class ZEDDataLoader:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.args.model_type == 'roberta':
            self.cls_token = '<s>'
            self.sep_token = '</s>'
        else:
            self.cls_token = '[CLS]'
            self.sep_token = '[SEP]'
        self.all_glosses = list()
        self.tensorized_gloss_data = list()
        self.tensorized_context_data = list()

    def _truncate_seq_pair(self, tokens, max_length, target_pos=-1):
        """Truncates a sequence pair in place to the maximum length."""
        if target_pos < max_length:
            return tokens[:max_length]
        if len(tokens) - target_pos < max_length:
            return tokens[len(tokens) - max_length:]
        return tokens[target_pos - int(max_length / 2):target_pos + int(max_length / 2)]

    def tensorize_sentence(self, tokens, limitation):
        tmp_input_ids = [self.tokenizer.convert_tokens_to_ids(self.cls_token)]
        tmp_input_mask = [1]
        word_to_token = list()
        for tmp_w in tokens:
            tmp_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tmp_w))
            word_to_token.append((len(tmp_input_ids), len(tmp_input_ids) + len(tmp_token_ids)))
            tmp_input_ids += tmp_token_ids
            tmp_input_mask += [1] * len(tmp_token_ids)
        if len(tmp_input_ids) > limitation - 1:
            tmp_input_ids = tmp_input_ids[:limitation - 1]
            tmp_input_mask = tmp_input_mask[:limitation - 1]
        tmp_input_ids += [self.tokenizer.convert_tokens_to_ids(self.sep_token)]
        tmp_input_mask += [1]
        tmp_output_mask = [1] * len(tmp_input_ids)
        padding_length = limitation - len(tmp_input_ids)
        tmp_input_ids = tmp_input_ids + ([0] * padding_length)
        tmp_input_mask = tmp_input_mask + ([0] * padding_length)
        tmp_output_mask = tmp_output_mask + ([0] * padding_length)
        assert len(tmp_input_ids) == limitation
        assert len(tmp_input_mask) == limitation
        assert len(tmp_output_mask) == limitation
        return tmp_input_ids, tmp_input_mask, tmp_output_mask, word_to_token

    def tensorize_sentence_with_target(self, tokens, target_position, limitation):
        tmp_input_ids = [self.tokenizer.convert_tokens_to_ids(self.cls_token)]
        tmp_input_mask = [1]
        tmp_output_mask = [0]
        target_token_pos = -1
        word_to_token = list()
        for i, tmp_w in enumerate(tokens):
            tmp_token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tmp_w))
            if target_position[0] <= i < target_position[1]:
                target_token_pos = i
                tmp_output_mask += [1] * len(tmp_token_ids)
            else:
                tmp_output_mask += [0] * len(tmp_token_ids)
            word_to_token.append((len(tmp_input_ids), len(tmp_input_ids) + len(tmp_token_ids), tmp_token_ids))
            tmp_input_ids += tmp_token_ids
            tmp_input_mask += [1] * len(tmp_token_ids)

        if len(tmp_input_ids) > limitation - 1:
            tmp_input_ids = self._truncate_seq_pair(tmp_input_ids, limitation - 1, target_token_pos)
            tmp_input_mask = self._truncate_seq_pair(tmp_input_mask, limitation - 1, target_token_pos)
            tmp_output_mask = self._truncate_seq_pair(tmp_output_mask, limitation - 1, target_token_pos)
        tmp_input_ids += [self.tokenizer.convert_tokens_to_ids(self.sep_token)]
        tmp_input_mask += [1]
        tmp_output_mask += [0]
        padding_length = limitation - len(tmp_input_ids)
        tmp_input_ids = tmp_input_ids + ([0] * padding_length)
        tmp_input_mask = tmp_input_mask + ([0] * padding_length)
        tmp_output_mask = tmp_output_mask + ([0] * padding_length)
        assert len(tmp_input_ids) == limitation
        assert len(tmp_input_mask) == limitation
        assert len(tmp_output_mask) == limitation
        return tmp_input_ids, tmp_input_mask, tmp_output_mask, word_to_token

    def tensorize_aligned_examples(self, input_pairs, candidate_glosses=None, number_training_example=-1,
                                   first_tensorize=True,
                                   number_negative_examples=2, cache_path='cache/pretrain'):

        if number_training_example > 0:
            input_pairs = input_pairs[:number_training_example]

        if first_tensorize:
            if self.args.use_cache:
                if not os.path.isdir(cache_path):
                    os.mkdir(cache_path)
                existing_files = os.listdir(cache_path)
                if len(existing_files) == 3:
                    # We can use the cached examples
                    with open(cache_path + '/all_glosses.json', 'r') as f:
                        self.all_glosses = json.load(f)
                    with open(cache_path + '/tensorized_gloss_data.json', 'r') as f:
                        self.tensorized_gloss_data = json.load(f)
                    with open(cache_path + '/tensorized_context_data.json', 'r') as f:
                        self.tensorized_context_data = json.load(f)
                else:
                    # We need to save to the cache
                    if candidate_glosses is None:
                        self.all_glosses = list()
                        for tmp_instance in input_pairs:
                            self.all_glosses.append(tmp_instance['gloss'])
                        self.all_glosses = list(set(self.all_glosses))
                    else:
                        self.all_glosses = candidate_glosses
                    self.tensorized_gloss_data = list()
                    for tmp_gloss in tqdm(self.all_glosses, desc="Tensorizing Gloss.."):
                        tmp_definition_ids, tmp_definition_input_mask, tmp_definition_output_mask, _ = \
                            self.tensorize_sentence(
                            tmp_gloss,
                            self.args.definition_max_length)
                        self.tensorized_gloss_data.append(
                            {'definition_ids': tmp_definition_ids, 'definition_input_masks': tmp_definition_input_mask,
                             'definition_output_masks': tmp_definition_output_mask})

                    self.tensorized_context_data = list()
                    for tmp_instance in tqdm(input_pairs, desc="Tensorizing Context.."):
                        tokens = tmp_instance['tokens']
                        target_position = tmp_instance['target_position']
                        gloss = tmp_instance['gloss']
                        s_ids, s_input_mask, s_output_mask, _ = \
                            self.tensorize_sentence_with_target(tokens, target_position, self.args.context_max_length)
                        self.tensorized_context_data.append(
                            {'input_ids': s_ids, 'all_input_masks': s_input_mask, 'all_output_masks': s_output_mask,
                             'label_pos': self.all_glosses.index(gloss)})
                    with open(cache_path + '/all_glosses.json', 'w') as f:
                        json.dump(self.all_glosses, f)
                    with open(cache_path + '/tensorized_gloss_data.json', 'w') as f:
                        json.dump(self.tensorized_gloss_data, f)
                    with open(cache_path + '/tensorized_context_data.json', 'w') as f:
                        json.dump(self.tensorized_context_data, f)
            else:
                if candidate_glosses is None:
                    self.all_glosses = list()
                    for tmp_instance in input_pairs:
                        self.all_glosses.append(tmp_instance['gloss'])
                    self.all_glosses = list(set(self.all_glosses))
                else:
                    self.all_glosses = candidate_glosses
                self.tensorized_gloss_data = list()
                for tmp_gloss in tqdm(self.all_glosses, desc="Tensorizing Gloss.."):
                    tmp_definition_ids, tmp_definition_input_mask, tmp_definition_output_mask, _ = \
                        self.tensorize_sentence(
                        tmp_gloss,
                        self.args.definition_max_length)
                    self.tensorized_gloss_data.append(
                        {'definition_ids': tmp_definition_ids, 'definition_input_masks': tmp_definition_input_mask,
                         'definition_output_masks': tmp_definition_output_mask})

                self.tensorized_context_data = list()
                for tmp_instance in tqdm(input_pairs, desc="Tensorizing Context.."):
                    tokens = tmp_instance['tokens']
                    target_position = tmp_instance['target_position']
                    gloss = tmp_instance['gloss']
                    s_ids, s_input_mask, s_output_mask, _ = \
                        self.tensorize_sentence_with_target(tokens, target_position, self.args.context_max_length)
                    self.tensorized_context_data.append(
                        {'input_ids': s_ids, 'all_input_masks': s_input_mask, 'all_output_masks': s_output_mask,
                         'label_pos': self.all_glosses.index(gloss)})

        all_input_ids = list()
        all_input_masks = list()
        all_output_masks = list()
        all_definition_ids = list()
        all_definition_input_masks = list()
        all_definition_output_masks = list()
        all_labels = list()

        all_gloss_positions = list(range(len(self.all_glosses)))
        random.shuffle(all_gloss_positions)
        for i in tqdm(range(len(input_pairs)), desc="creating training data..."):
            all_input_ids.append(self.tensorized_context_data[i]['input_ids'])
            all_input_masks.append(self.tensorized_context_data[i]['all_input_masks'])
            all_output_masks.append(self.tensorized_context_data[i]['all_output_masks'])
            positive_pos = self.tensorized_context_data[i]['label_pos']

            tmp_definition_ids = [self.tensorized_gloss_data[positive_pos]['definition_ids']]
            tmp_definition_input_mask = [self.tensorized_gloss_data[positive_pos]['definition_input_masks']]
            tmp_definition_output_mask = [self.tensorized_gloss_data[positive_pos]['definition_output_masks']]

            negative_positions = all_gloss_positions[:positive_pos] + all_gloss_positions[positive_pos:]

            selected_negative_positions = random.choices(negative_positions, k=number_negative_examples)

            for tmp_pos in selected_negative_positions:
                tmp_definition_ids.append(self.tensorized_gloss_data[tmp_pos]['definition_ids'])
                tmp_definition_input_mask.append(self.tensorized_gloss_data[tmp_pos]['definition_input_masks'])
                tmp_definition_output_mask.append(self.tensorized_gloss_data[tmp_pos]['definition_output_masks'])

            all_definition_ids.append(tmp_definition_ids)
            all_definition_input_masks.append(tmp_definition_input_mask)
            all_definition_output_masks.append(tmp_definition_output_mask)
            all_labels.append(0)

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
        all_output_masks = torch.tensor(all_output_masks, dtype=torch.long)
        all_definition_ids = torch.tensor(all_definition_ids, dtype=torch.long)
        all_definition_input_masks = torch.tensor(all_definition_input_masks, dtype=torch.long)
        all_definition_output_masks = torch.tensor(all_definition_output_masks, dtype=torch.long)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_masks, all_output_masks,
                             all_definition_ids,
                             all_definition_input_masks, all_definition_output_masks, all_labels)

    def tensorize_aligned_examples_strong_negative_sampling(self, input_pairs, model,
                                                            number_training_example=-1,
                                                            number_negative_examples=2,
                                                            p=0.1):
        if number_training_example > 0:
            input_pairs = input_pairs[:number_training_example]

        all_input_ids = list()
        all_input_masks = list()
        all_output_masks = list()
        all_definition_ids = list()
        all_definition_input_masks = list()
        all_definition_output_masks = list()
        all_labels = list()

        gloss2embedding, _ = compute_representation_for_all_glosses(self.args, model, self.all_glosses, self,
                                                                    return_type='list')
        context_id2embedding = compute_representation_for_all_contexts(self.args, model, input_pairs, self)
        gloss_embeddings = list()
        for tmp_gloss in self.all_glosses:
            tmp_embedding = np.asarray(gloss2embedding[tmp_gloss])
            gloss_embeddings.append(tmp_embedding / np.linalg.norm(tmp_embedding))
        gloss_embeddings = np.asarray(gloss_embeddings).astype(np.float32)
        for tmp_id in context_id2embedding:
            tmp_embedding = context_id2embedding[tmp_id]
            context_id2embedding[tmp_id] = tmp_embedding / np.linalg.norm(tmp_embedding)
        nlist = 2  # number of clusters
        quantiser = faiss.IndexFlatIP(768)
        index = faiss.IndexIVFFlat(quantiser, 768, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(gloss_embeddings)
        index.add(gloss_embeddings)  # add the vectors and update the index

        all_gloss_positions = list(range(len(self.all_glosses)))

        for i in tqdm(range(len(input_pairs))):

            positive_pos = self.tensorized_context_data[i]['label_pos']

            tmp_context_representation = context_id2embedding[i]

            if random.random() < p:

                _, indices = index.search(np.asarray([tmp_context_representation]).astype(np.float32),
                                                   number_negative_examples + 1)

                tmp_definition_ids = [self.tensorized_gloss_data[positive_pos]['definition_ids']]
                tmp_definition_input_mask = [self.tensorized_gloss_data[positive_pos]['definition_input_masks']]
                tmp_definition_output_mask = [self.tensorized_gloss_data[positive_pos]['definition_output_masks']]

                for tmp_pos in indices[0]:
                    if tmp_pos != positive_pos:
                        tmp_definition_ids.append(self.tensorized_gloss_data[tmp_pos]['definition_ids'])
                        tmp_definition_input_mask.append(self.tensorized_gloss_data[tmp_pos]['definition_input_masks'])
                        tmp_definition_output_mask.append(
                            self.tensorized_gloss_data[tmp_pos]['definition_output_masks'])
                    if len(tmp_definition_ids) == number_negative_examples + 1:
                        break
            else:
                negative_positions = all_gloss_positions[:positive_pos] + all_gloss_positions[positive_pos:]

                tmp_definition_ids = [self.tensorized_gloss_data[positive_pos]['definition_ids']]
                tmp_definition_input_mask = [self.tensorized_gloss_data[positive_pos]['definition_input_masks']]
                tmp_definition_output_mask = [self.tensorized_gloss_data[positive_pos]['definition_output_masks']]

                selected_negative_positions = random.choices(negative_positions, k=number_negative_examples)

                for tmp_pos in selected_negative_positions:
                    tmp_definition_ids.append(self.tensorized_gloss_data[tmp_pos]['definition_ids'])
                    tmp_definition_input_mask.append(self.tensorized_gloss_data[tmp_pos]['definition_input_masks'])
                    tmp_definition_output_mask.append(self.tensorized_gloss_data[tmp_pos]['definition_output_masks'])

            all_input_ids.append(self.tensorized_context_data[i]['input_ids'])
            all_input_masks.append(self.tensorized_context_data[i]['all_input_masks'])
            all_output_masks.append(self.tensorized_context_data[i]['all_output_masks'])
            all_definition_ids.append(tmp_definition_ids)
            all_definition_input_masks.append(tmp_definition_input_mask)
            all_definition_output_masks.append(tmp_definition_output_mask)
            all_labels.append(0)

        # logger.info("Successfully loaded %s examples", len(all_input_ids))
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
        all_output_masks = torch.tensor(all_output_masks, dtype=torch.long)
        all_definition_ids = torch.tensor(all_definition_ids, dtype=torch.long)
        all_definition_input_masks = torch.tensor(all_definition_input_masks, dtype=torch.long)
        all_definition_output_masks = torch.tensor(all_definition_output_masks, dtype=torch.long)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_masks, all_output_masks,
                             all_definition_ids,
                             all_definition_input_masks, all_definition_output_masks, all_labels)


class ZED(torch.nn.Module):
    def __init__(self, config, context_encoder, gloss_encoder, args):
        super(ZED, self).__init__()
        self.context_encoder = context_encoder
        self.gloss_encoder = gloss_encoder  # (config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softmax = nn.Softmax(dim=1)

        self.context_calculator = nn.Linear(config.hidden_size, args.output_size)
        self.mean_calculator = nn.Linear(config.hidden_size, args.output_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.cosine = nn.CosineSimilarity(dim=2)
        self.config = config
        self.args = args

    def forward(
            self,
            input_ids=None,  # batch_size, sentence_length
            input_masks=None,  # batch_size, sentence_length
            output_masks=None,  # batch_size, sentence_length
            definition_ids=None,  # batch_size, num_definition, sentence_length
            definition_input_masks=None,  # batch_size, num_definition, sentence_length
            definition_output_masks=None,  # batch_size, num_definition, sentence_length
            labels=None
    ):
        if labels is None:
            # we are in the evaluation mode
            if input_ids is not None:
                return self.predict_context(input_ids, input_masks, output_masks)
            elif definition_ids is not None:
                return self.predict_gloss(definition_ids,
                                          definition_input_masks,
                                          definition_output_masks)
            else:
                raise NotImplementedError
        batch_size = definition_ids.shape[0]
        num_gloss = definition_ids.shape[1]
        context_representation = self.predict_context(input_ids, input_masks, output_masks)
        all_gloss_ids = definition_ids.view(batch_size * num_gloss, -1)
        all_definition_input_masks = definition_input_masks.view(batch_size * num_gloss, -1)
        all_definition_output_masks = definition_output_masks.view(batch_size * num_gloss, -1)
        gloss_mean = self.predict_gloss(all_gloss_ids,
                                                        all_definition_input_masks,
                                                        all_definition_output_masks)
        gloss_mean = gloss_mean.view(batch_size, num_gloss, -1)
        all_context_representation = context_representation.unsqueeze(1).repeat(1, num_gloss,
                                                                                1)

        logits = self.cosine(all_context_representation, gloss_mean)  # batch_size, num_glosses

        margin_loss = MyLoss(margin=0.2)
        contrast_loss = margin_loss(logits, labels)

        bd_loss = BoundaryLoss()
        loss = bd_loss(logits, labels)
        return self.args.bd_weight * loss + (1 - self.args.bd_weight) * contrast_loss

    def predict_gloss(
            self,
            definition_ids=None,  # batch_size, num_definition, sentence_length
            definition_input_masks=None,  # batch_size, num_definition, sentence_length
            definition_output_masks=None,  # batch_size, num_definition, sentence_length
            drop_out=False
    ):
        batch_size = definition_ids.shape[0]
        definition_representation = self.gloss_encoder(definition_ids, attention_mask=definition_input_masks)
        pooled_definition_outputs = definition_representation[0]
        pooled_definition_outputs = pooled_definition_outputs.view(batch_size, -1,
                                                                   self.config.hidden_size)
        # [batch_size, sentence_length, hidden_size]
        if drop_out:
            pooled_definition_outputs = self.dropout(pooled_definition_outputs)
        definition_mask = definition_output_masks.unsqueeze(2).repeat(1, 1,
                                                                      self.config.hidden_size)
        # [batch_size, sentence_length, hidden_size]
        final_definition_representation = torch.mean(pooled_definition_outputs * definition_mask,
                                                     1)  # [batch_size, hidden_size]
        definition_representation_mean = self.mean_calculator(final_definition_representation)
        return definition_representation_mean

    def predict_context(
            self,
            input_ids=None,  # batch_size, sentence_length
            input_masks=None,  # batch_size, sentence_length
            output_masks=None,  # batch_size, sentence_length
            drop_out=False
    ):
        batch_size = input_ids.shape[0]
        context_outputs = self.context_encoder(input_ids, attention_mask=input_masks)  # batch_size, sentence_length
        pooled_context_output = context_outputs[0]  # [batch_size, k, sentence_length, hidden_size]

        pooled_context_output = pooled_context_output.view(batch_size, input_ids.shape[-1],
                                                           self.config.hidden_size)
        if drop_out:
            pooled_context_output = self.dropout(pooled_context_output)  # [batch_size, sentence_length, hidden_size]
        cand_mask = output_masks.unsqueeze(2).repeat(1, 1,
                                                     self.config.hidden_size)
        # [batch_size, sentence_length, hidden_size]
        context_representation = torch.mean(pooled_context_output * cand_mask, 1)  # [batch_size, hidden_size]
        return context_representation

    def compute_representation_for_all_glosses(self, glosses, dataloader, return_type='list'):
        definition_ids = list()
        definition_input_masks = list()
        definition_output_masks = list()
        ids = list()
        for tmp_id, tmp_gloss in enumerate(glosses):
            definition_id, definition_input_mask, definition_output_mask, _ = dataloader.tensorize_sentence(
                tmp_gloss, dataloader.args.definition_max_length)
            definition_ids.append(definition_id)
            definition_input_masks.append(definition_input_mask)
            definition_output_masks.append(definition_output_mask)
            ids.append(tmp_id)
        definition_ids = torch.tensor(definition_ids, dtype=torch.long)
        definition_input_masks = torch.tensor(definition_input_masks, dtype=torch.long)
        definition_output_masks = torch.tensor(definition_output_masks, dtype=torch.long)
        ids = torch.tensor(ids, dtype=torch.long)
        all_definition_info = TensorDataset(definition_ids, definition_input_masks, definition_output_masks, ids)
        compute_embedding_batch_size = 128 * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(all_definition_info) if self.args.local_rank == -1 else DistributedSampler(
            all_definition_info)
        gloss_dataloader = DataLoader(all_definition_info, sampler=eval_sampler,
                                      batch_size=compute_embedding_batch_size)
        gloss_to_mean = dict()
        for batch in gloss_dataloader:
            self.eval()

            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {'definition_ids': batch[0],
                          'definition_input_masks': batch[1],
                          'definition_output_masks': batch[2]}
                gloss_mean = self.predict_gloss(**inputs)
                if return_type == 'list':
                    gloss_mean = gloss_mean.tolist()
                else:
                    gloss_mean = gloss_mean
                ids = batch[3].tolist()
                for i, tmp_id in enumerate(ids):
                    gloss_to_mean[glosses[tmp_id]] = gloss_mean[i]
        return gloss_to_mean

    def compute_representation_for_all_contexts(self, input_instances, dataloader, return_type='list'):
        input_ids = list()
        input_masks = list()
        output_masks = list()
        ids = list()
        for tmp_id, tmp_instance in enumerate(input_instances):
            tokens = tmp_instance['tokens']
            target_position = tmp_instance['target_position']

            s_ids, s_input_mask, s_output_mask, _ = \
                dataloader.tensorize_sentence_with_target(tokens, target_position, dataloader.args.context_max_length)
            input_ids.append(s_ids)
            input_masks.append(s_input_mask)
            output_masks.append(s_output_mask)
            ids.append(tmp_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_masks = torch.tensor(input_masks, dtype=torch.long)
        output_masks = torch.tensor(output_masks, dtype=torch.long)
        ids = torch.tensor(ids, dtype=torch.long)
        all_context_info = TensorDataset(input_ids, input_masks, output_masks, ids)
        compute_embedding_batch_size = 64 * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(all_context_info) if self.args.local_rank == -1 else DistributedSampler(
            all_context_info)
        context_dataloader = DataLoader(all_context_info, sampler=eval_sampler, batch_size=compute_embedding_batch_size)
        id_to_embedding = dict()
        for batch in tqdm(context_dataloader, desc="Computing embedding for contexts"):
            self.eval()

            batch = tuple(t.to(self.args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'input_masks': batch[1],
                          'output_masks': batch[2]}
                context_representations = self.predict_context(**inputs)
                if return_type == 'list':
                    embeddings = context_representations.tolist()
                else:
                    embeddings = context_representations
                ids = batch[3].tolist()
                for i, tmp_id in enumerate(ids):
                    id_to_embedding[tmp_id] = embeddings[i]
        return id_to_embedding


def compute_representation_for_all_contexts(args, model, input_instances, dataloader, return_type='list'):
    input_ids = list()
    input_masks = list()
    output_masks = list()
    ids = list()

    counter = 0
    for tmp_instance in tqdm(input_instances, desc="Tensorizing Context.."):
        tokens = tmp_instance['tokens']
        target_position = tmp_instance['target_position']

        s_ids, s_input_mask, s_output_mask, _ = \
            dataloader.tensorize_sentence_with_target(tokens, target_position, dataloader.args.context_max_length)
        input_ids.append(s_ids)
        input_masks.append(s_input_mask)
        output_masks.append(s_output_mask)
        ids.append(counter)
        counter += 1

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    output_masks = torch.tensor(output_masks, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)
    all_context_info = TensorDataset(input_ids, input_masks, output_masks, ids)
    compute_embedding_batch_size = 64 * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(all_context_info) if args.local_rank == -1 else DistributedSampler(
        all_context_info)
    context_dataloader = DataLoader(all_context_info, sampler=eval_sampler, batch_size=compute_embedding_batch_size)
    id_to_embedding = dict()
    for batch in tqdm(context_dataloader, desc="Computing embedding for contexts"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'input_masks': batch[1],
                      'output_masks': batch[2]}
            context_representations = model(**inputs)
            if return_type == 'list':
                embeddings = context_representations.tolist()
            else:
                embeddings = context_representations
            ids = batch[3].tolist()
            for i, tmp_id in enumerate(ids):
                id_to_embedding[tmp_id] = embeddings[i]
    return id_to_embedding


def compute_representation_for_all_glosses(args, model, glosses, dataloader, return_type='list'):
    definition_ids = list()
    definition_input_masks = list()
    definition_output_masks = list()
    ids = list()

    for i, tmp_gloss in enumerate(glosses):
        try:
            tmp_definition_ids, tmp_definition_input_mask, tmp_definition_output_mask, _ = \
                dataloader.tensorize_sentence(
                tmp_gloss,
                dataloader.args.definition_max_length)
        except AttributeError:
            tmp_definition_ids, tmp_definition_input_mask, tmp_definition_output_mask, _ = \
                dataloader.module.tensorize_sentence(
                tmp_gloss,
                dataloader.args.definition_max_length)
        definition_ids.append(tmp_definition_ids)
        definition_input_masks.append(tmp_definition_input_mask)
        definition_output_masks.append(tmp_definition_output_mask)
        ids.append(i)

    definition_ids = torch.tensor(definition_ids, dtype=torch.long)
    definition_input_masks = torch.tensor(definition_input_masks, dtype=torch.long)
    definition_output_masks = torch.tensor(definition_output_masks, dtype=torch.long)
    ids = torch.tensor(ids, dtype=torch.long)
    all_definition_info = TensorDataset(definition_ids, definition_input_masks, definition_output_masks, ids)
    compute_embedding_batch_size = 128 * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(all_definition_info) if args.local_rank == -1 else DistributedSampler(
        all_definition_info)
    gloss_dataloader = DataLoader(all_definition_info, sampler=eval_sampler,
                                  batch_size=compute_embedding_batch_size)
    gloss_to_mean = dict()
    # for batch in tqdm(gloss_dataloader, desc="Computing embedding for glosses"):
    for batch in tqdm(gloss_dataloader, desc="Computing embedding for glosses"):

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'definition_ids': batch[0],
                      'definition_input_masks': batch[1],
                      'definition_output_masks': batch[2]}
            gloss_mean = model(**inputs)
            if return_type == 'list':
                gloss_mean = gloss_mean.tolist()
            else:
                gloss_mean = gloss_mean
            ids = batch[3].tolist()
            for i, tmp_id in enumerate(ids):
                gloss_to_mean[glosses[tmp_id]] = gloss_mean[i]
    return gloss_to_mean


class MyLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super(MyLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, labels):

        num_example = inputs.size(0)
        num_gloss = inputs.size(1)
        correct_predictions = []
        other_predictions = []
        for i in range(num_example):
            tmp_correct_prediction = []
            tmp_other_predictions = []
            for j in range(num_gloss):
                if j == labels[i]:
                    tmp_correct_prediction.append(inputs[i][j].unsqueeze(0))
                else:
                    tmp_other_predictions.append(inputs[i][j].unsqueeze(0))
            correct_predictions.append(torch.cat(tmp_correct_prediction, dim=0).unsqueeze(0))
            other_predictions.append(torch.cat(tmp_other_predictions, dim=0).unsqueeze(0))

        correct_predictions = torch.cat(correct_predictions, dim=0)
        other_predictions = torch.cat(other_predictions, dim=0)
        y = torch.ones_like(other_predictions)
        loss = self.ranking_loss(correct_predictions.repeat(1, num_gloss - 1), other_predictions, y)
        return loss


class BoundaryLoss(torch.nn.Module):
    def __init__(self, positive_margin=0.99999, negative_margin=0.1):
        super(BoundaryLoss, self).__init__()
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        self.positive_ranking_loss = torch.nn.MarginRankingLoss(margin=positive_margin)
        self.negative_ranking_loss = torch.nn.MarginRankingLoss(margin=negative_margin)

    def forward(self, inputs, labels):

        num_example = inputs.size(0)
        num_gloss = inputs.size(1)
        correct_predictions = []
        other_predictions = []
        for i in range(num_example):
            tmp_correct_prediction = []
            tmp_other_predictions = []
            for j in range(num_gloss):
                if j == labels[i]:
                    tmp_correct_prediction.append(inputs[i][j].unsqueeze(0))
                else:
                    tmp_other_predictions.append(inputs[i][j].unsqueeze(0))
            correct_predictions.append(torch.cat(tmp_correct_prediction, dim=0).unsqueeze(0))
            other_predictions.append(torch.cat(tmp_other_predictions, dim=0).unsqueeze(0))

        correct_predictions = torch.cat(correct_predictions, dim=0)
        other_predictions = torch.cat(other_predictions, dim=0)
        y = torch.ones_like(other_predictions)

        zeros = torch.zeros_like(other_predictions)
        return self.positive_ranking_loss(correct_predictions.repeat(1, num_gloss - 1), zeros, y)


def get_predictions(args, data_loader, eval_package, model, device):
    model.eval()
    eval_dataset = eval_package['eval_dataset'][:10000]
    type2definitions = eval_package['type2definitions']
    all_glosses = list()
    for tmp_e_type in type2definitions:
        all_glosses.append(type2definitions[tmp_e_type])
    gloss2mean = compute_representation_for_all_glosses(args, model, all_glosses, data_loader,
                                                                        return_type='list')
    type2embedding = dict()
    for tmp_e_type in type2definitions:
        type2embedding[tmp_e_type] = gloss2mean[type2definitions[tmp_e_type]]
    id_2_context_representation = compute_representation_for_all_contexts(args, model, eval_dataset, data_loader)

    torch_cosine = nn.CosineSimilarity(dim=1)
    all_types = list(type2definitions.keys())
    type_embeddings = list()
    for tmp_e_type in all_types:
        type_embeddings.append(torch.as_tensor(type2embedding[tmp_e_type]).to(device).unsqueeze(0))
    # print(type_embeddings)
    type_embeddings = torch.cat(type_embeddings, dim=0)
    # print(type_embeddings.size())
    paired_eval_data = list()
    prediction_results = list()
    for i, tmp_instance in enumerate(eval_dataset):
        paired_eval_data.append((i, tmp_instance))
    for i, tmp_instance in tqdm(paired_eval_data):
        type2score = dict()

        context_embedding = torch.as_tensor(id_2_context_representation[i]).to(device).unsqueeze(0).repeat(
            len(all_types), 1)
        cosine_scores = torch_cosine(context_embedding, type_embeddings)
        for j, tmp_type in enumerate(all_types):
            type2score[tmp_type] = cosine_scores[j].tolist()

        tmp_instance['prediction_results'] = type2score
        prediction_results.append(tmp_instance)
    return prediction_results


def evaluation(args, data_loader, eval_package, model, device):
    prediction_results = get_predictions(args, data_loader, eval_package, model, device)
    evaluate_cosine_prediction_result(prediction_results, 0.7)


def evaluate_cosine_prediction_result(test_instances, threshold, classification_only=False):
    prediction_count = 0
    gold_count = 0
    identification_correct_count = 0
    classification_correct_count = 0

    for tmp_instance in test_instances:
        sorted_types = sorted(tmp_instance['prediction_results'], key=lambda x: tmp_instance['prediction_results'][x],
                              reverse=True)
        if classification_only:
            if tmp_instance['label'] != 'NA':
                gold_count += 1
                prediction_count += 1
                identification_correct_count += 1
                if tmp_instance['label'] == sorted_types[0]:
                    classification_correct_count += 1
        else:
            if tmp_instance['label'] != 'NA':
                gold_count += 1
                if tmp_instance['prediction_results'][sorted_types[0]] > threshold:
                    prediction_count += 1
                    identification_correct_count += 1
                    if tmp_instance['label'] == sorted_types[0]:
                        classification_correct_count += 1
            else:
                if tmp_instance['prediction_results'][sorted_types[0]] > threshold:
                    prediction_count += 1
    try:
        identification_p = identification_correct_count / prediction_count
    except ZeroDivisionError:
        identification_p = 0.0
    try:
        identification_r = identification_correct_count / gold_count
    except ZeroDivisionError:
        identification_r = 0.0
    try:
        identification_f1 = 2 * identification_p * identification_r / (identification_p + identification_r)
    except ZeroDivisionError:
        identification_f1 = 0.0

    try:
        classification_p = classification_correct_count / prediction_count
    except ZeroDivisionError:
        classification_p = 0.0
    try:
        classification_r = classification_correct_count / gold_count
    except ZeroDivisionError:
        classification_r = 0.0
    try:
        classification_f1 = 2 * classification_p * classification_r / (classification_p + classification_r)
    except ZeroDivisionError:
        classification_f1 = 0.0
    if classification_only:
        print('Classification only: ' + 'Identification:', 'P:', identification_p, 'R:', identification_r,
              'F1:',
              identification_f1, 'Classification:', 'P:', classification_p, 'R:', classification_r, 'F1:',
              classification_f1)
    else:
        print('Threashold: ' + str(threshold) + ' Identification:', 'P:', identification_p, 'R:', identification_r,
              'F1:',
              identification_f1, 'Classification:', 'P:', classification_p, 'R:', classification_r, 'F1:',
              classification_f1)


logger = logging.getLogger(__name__)
