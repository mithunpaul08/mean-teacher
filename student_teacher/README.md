 
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
pip install gitpython
conda install pytorch-cpu torchvision-cpu -c pytorch *
```
\* **note**: for pytorch instinstallation get the right command from the pytorch [homepage](https://pytorch.org/) based on your OS and configs.

*PS: I personally like/trust `pip install *` instead of `conda install` * because the repos of pip are more comprehensive



Run these commands from parent folder :

Note: uncomment corresponding lines in `./get_data_run.sh` according to which datasets you want to train-dev-test

```
./get_glove.sh
```

Now to train , run:

```
./get_data_run.sh
```
You can keep track of the progress by doing `tail -f mean_teacher_sha.log` where sha is the sha of the latest git commit

Notes: 
- if using in the mode of one teacher/classifier (i.e no students), remove `--add_student True`
- if you dont have a gpu, remove `--which_gpu_to_use 0`
- in this particular case the file train_full_with_evi_sents is a collection of all claims and the corresponding
 evidences in the training data of [FEVER](http://fever.ai/) challenge. This is not available in public unlike the FEVER data. 
 This is the output of the IR module of FEVER baseline [code](http://fever.ai/task.html).
- if you would like to reuse a single project in comet.ml (instead of a new project everytime)
 to draw graphs do  `create_new_comet_graph False`. It is advised to create a new project because the graphs get left over from previous runs. Instead if you are tesitng on say a toy dataset, its ok, to reuse a project.
- The value of `--use_ema True` will make the teacher an exponential moving average of the student. This replicates the architecture in harry valpola's mean teacher [work](https://papers.nips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf)
- If you want to use a saved model set `load_model_from_disk_and_test` to `True`. The model is expected to be kept at  `model_storage/best_model.pth`. Also make sure the value of `delex_test` in `initializer.py` points to the file you want to test this model on. Usually this loading saved model
thing is done when you want to save a model that was trained on fever, and gave a very good performance on the cross-domain, fnc dataset's dev partition at say epoch 5. You save that model (if you dev is pointing to a particular partition, the code automatically saves the model before 
doing early stopping on that partition). Now you want to use this saved model to test (only once) on a test partition. Then you load this model and feed the test partition as dev by changing the path of `delex_test`  


Notes to self:
- to run on a laptop use `./get_glove_small.sh`
- on server dont do `./get_glove.sh` . instead do `cp ~/glove/glove.840B.300d.txt data/glove/glove.840B.300d.txt` 
- If you get: the import torch before comet error again. fixed it by forcefully upgrading to new version using pip install --no-cache-dir --upgrade comet_ml"
- in clara its better to use `--which_gpu_to_use 2` since everyone gets assigned gpu0 by default
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
update: Jan 18th2020
- to test with fnc, download fnc_train file using the command added to get_data.sh
- then do head -25413 on this and rename it as fnc_dev_delex.jsonl
- then make fever_delex_dev_local point to this file in initializer.py
- this is a hack. had to do it this way because i couldn't find a fnc dev or test file which was delexicalized using PERSON-C1 format. Everything we have is personc1 (i.e without the dash)
