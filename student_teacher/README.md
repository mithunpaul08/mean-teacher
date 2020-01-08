 
# Fact Verification using Mean Teacher in PyTorch

In this fork of the original mean teacher code, we replace the feed forward networks in a mean teacher setup with 
 a decomposable attention. Also the data input is that from FEVER 2018 shared task.
 
# Pre reqs:
 
 The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
conda create --name mean_teacher python=3 numpy scipy pandas nltk tqdm
source activate mean_teacher
pip sklearn
pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119
conda install pytorch-cpu torchvision-cpu -c pytorch *
```
\* **note**: for pytorch instinstallation get the right command from the pytorch [homepage](https://pytorch.org/) based on your OS and configs.

*PS: I personally like/trust `pip install *` instead of `conda install` * because the repos of pip are more comprehensive


Run these commands from parent folder :

```
mkdir -p data/rte/fever/train/
mkdir -p data/rte/fever/dev/
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_lex_4labels.jsonl -O data/rte/fever/train/fever_train_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_delex_oaner_4labels.jsonl -O data/rte/fever/train/fever_train_delex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_lex_4labels.jsonl -O data/rte/fever/dev/fever_dev_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_delex_oaner_split_4labels.jsonl -O data/rte/fever/dev/fever_dev_delex.jsonl
 

```
Note that in this particular case the file train_full_with_evi_sents is a collection of all claims and the corresponding
 evidences in the training data of [FEVER](http://fever.ai/) challenge. This is not available in public unlike the FEVER data. 
 This is the output of the IR module of FEVER baseline [code](http://fever.ai/task.html).

To train on FEVER, run e.g.:


``` 
python main.py --add_student True --which_gpu_to_use 0
```

Notes: 
- if using in the mode of one teacher/classifier, remove `--add_student True`
- if you dont want have a gpu, remove `--which_gpu_to_use 0`


Notes to self:

- If you get: the import torch before comet error again. fixed it by forcefully upgrading to new version using pip install --no-cache-dir --upgrade comet_ml"
- every time you do a fresh run or branch change, do wget from the commands above. Then do a head -100 for each of these files as shown below to reduce size
```
head -100 data/rte/fever/train/fever_train_delex.jsonl > temp
mv temp data/rte/fever/train/fever_train_delex.jsonl
head -100 data/rte/fever/train/fever_train_lex.jsonl > temp
mv temp data/rte/fever/train/fever_train_lex.jsonl
head -20 data/rte/fever/dev/fever_dev_delex.jsonl > temp
mv temp data/rte/fever/dev/fever_dev_delex.jsonl
head -20 data/rte/fever/dev/fever_dev_lex.jsonl > temp
mv temp data/rte/fever/dev/fever_dev_lex.jsonl

```

or run `./reduce_size.sh`
