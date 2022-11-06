# ZED
This is the repository for EMNLP 2022 paper "Efficient Zero-shot Event Extraction with Context-Definition Alignment"


### Step 1: Setup
1. Setup the environment with "requirements.txt".
2. Download the preprocessed data from [data](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/Ea_-vqi24kpGko03HLKjawsBhILB5Ol0KeTMCalwaGx69A?e=cLoczZ).
3. create the cache and output directory for later use.

PS: If you want to reprocess the data, or you want to use your own data, simply follow "data_preparation.py". Otherwise, just unzip the downloaded data and put the two folders (i.e., data and maven_data) under the current directory.

### Step 2: Pretrain

```
python main.py --output_dir output/pretrain --train_source pretrain --do_train
```

### Step 3: Dataset Specific Warming

```
python main.py --model_name_or_path output/pretrain --output_dir output/warm --hard_negative_sampling all_hard --train_source warm --do_train
```

### Step 4: Inference

```
python main.py --model_name_or_path output/warm --do_eval
```

### Disclaimer
This is not an officially supported Tencent product. But if you have some questions about the code, you are welcome to open an issue or send an email, we will respond to that as soon as possible.