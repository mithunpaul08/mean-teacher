
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


Run these commands from folder `pytorch/` :

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


``` python main.py
```

Notes to self:

If you get: the import torch before comet error again. fixed it by forcefully upgrading to new version using pip install --no-cache-dir --upgrade comet_ml"
### Latest status as of Nov 12th 2019 4pm.
- fixed negative val accuracy problem. But every epoch now
has exactly same val accuracies..looks like classifier2 is not learning
training loss and accuracies have improved though
 INFO:main:92     :2882/3724      running_loss_lex:11.61      running_acc_lex:86.18   running_loss_delex:35.5     running_acc_delex:55.06