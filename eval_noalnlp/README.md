
# Fact Verification 
In this folder a [decomposable attention](https://arxiv.org/pdf/1606.01933.pdf) based neural network model will be trained and tested
on the lexicalized and delexicalized/masked files. Ideally this will be the second step in the entire pipeline, 
the first step being the data masking done from the folder nn/

 

# Pre reqs:
 
 The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
conda create --name rte python=3 numpy scipy pandas nltk tqdm
source activate rte
pip install sklearn
pip install jsonlines
pip install comet
pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119 **refer note
conda install pytorch-cpu torchvision-cpu -c pytorch *refer note
```
**Note**:for pytorch installations get the right command from the pytorch [homepage](https://pytorch.org/) based on your OS and configs.

To download data run these command from the folder `pytorch/` :

```
git clone thisrepo.git
```


#### Testing:

To test using a model that was trained on FEVER lexicalized data, and test on FNC dataset, run the 
following commands from the folder `pytorch/`. 

```
./get_data_lex.sh
./get_glove_small.sh
./get_model_lex.sh
python main.py --run_type test --database_to_test_with fnc 
```
Note: to debug you can pass `--log_level DEBUG`

To test using a model trained on FEVER delexicalized data (mentioned as OANER in the paper), and test on FNC dataset, run the following commands from the folder `pytorch/`. 
```
./get_data_delex.sh
./get_glove_small.sh
./get_model_delex.sh
python main.py --run_type test --database_to_test_with fnc 
```


To test using a model trained on [MNLI](https://www.nyu.edu/projects/bowman/multinli/) 
lexicalized data and test on cross domain within MNLI mismatched
```
./get_glove_small.sh
./create_data_folders.sh 
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_train.jsonl -O data/rte/train_input_file.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_matched.jsonl  -O data/rte/dev_input_file.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_mismatched.jsonl -O data/rte/test_input_file.jsonl

wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/best_model_trained_on_mnli_lex.pth -O model_storage/best_model.pth
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/vectorizer_trained_on_mnli_lex.json -O model_storage/vectorizer.json
python main.py --run_type test --database_to_test_with mnli
```

To test using a model trained on [MNLI](https://www.nyu.edu/projects/bowman/multinli/) lexicalized data and test on 
cross domain within [MEDNLI](https://physionet.org/content/mednli/1.0.0/)
```
./get_glove_small.sh
mkdir -p data/rte/mednli/dev
wget https://storage.googleapis.com/fact_verification_mithun_files/mednli_converted_claim_ev_format/mli_dev_lex.jsonl -O data/rte/mednli/dev/mednli_dev_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/best_model_trained_on_mnli_lex.pth -O model_storage/best_model.pth
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/vectorizer_trained_on_mnli_lex.json -O model_storage/vectorizer.json
python main.py --run_type test --database_to_test_with mednli_lex
```

#### Note: To test on other data input files, you just need to replace the corresponding file names in the first wget command
with one from the list given below.
- train partition of mnli delexicalized with oaner: mnli_train_delex_oaner.jsonl
- dev partition of mnli delexicalized with oaner : mnli_dev_delex_oaner
- dev partition of mnli (aka matched partition) which is lexicalized  : mu_matched.jsonl

For example if you want to evaluate on the dev partition of delexicalized-with-oaner-MedNLI dataset  using a model that was trained on 
 delexicalized-with-oaner-MNLI, the corresponding commands 
(only ones that will be different) will be:


```
wget https://storage.googleapis.com/fact_verification_mithun_files/mednli_converted_claim_ev_format/mednli_dev_delex_oaner.jsonl -O data/rte/mednli/dev/mednli_dev.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/best_model_trained_on_mnli_delex_oaner.pth -O model_storage/best_model.pth
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/vectorizer_trained_on_mnli_delex_oaner.json -O model_storage/vectorizer.json
```

Similarly commands to download a model that was trained on mnli data (which was
 delexicalized using OANER)
and to evaluate on delexicalized dev of mnli (aka `matched` partition as per mnli nomenclature) which was also
 delexicalized using OANER use the commands below. (you should get 59.32% accuracy)
```
./get_glove_small.sh		
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_train.jsonl -O data/rte/train_input_file.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_mismatched.jsonl -O data/rte/test_input_file.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mnli_dev_delex_oaner.jsonl  -O data/rte/dev_input_file.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/best_model_trained_on_mnli_delex_oaner.pth -O model_storage/best_model.pth
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/vectorizer_trained_on_mnli_delex_oaner.json -O model_storage/vectorizer.json
python main.py --run_type test --database_to_test_with mnli
```

Commands to download a model that was trained on mnli data (which was
 delexicalized using OANER)
and to evaluate on delexicalized out of domain dev of mnli (aka `mismatched` partition as per mnli nomenclature) which was also
 delexicalized using OANER use the commands below. (you should get an accuracy of)
```
./get_glove_small.sh		
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_train.jsonl -O data/rte/train_input_file.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_mismatched.jsonl -O data/rte/dev_input_file.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mnli_dev_mismatched_delex_oaner.jsonl -O data/rte/test_input_file.jsonl
 
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/best_model_trained_on_mnli_delex_oaner.pth -O model_storage/best_model.pth
wget https://storage.googleapis.com/fact_verification_mithun_files/trained_models/MNLI_models/vectorizer_trained_on_mnli_delex_oaner.json -O model_storage/vectorizer.json
python main.py --run_type test --database_to_test_with mnli
```
Note: In all these commands the  2 wgets which download the train and dev files
 are dummy and is needed due to a design choice related to pandas data frames.

### Training:

To train on FEVER lexicalized, run the following command in the folder `pytorch/` :

``` 
./get_glove.sh
./get_data_lex.sh
python main.py --run_type train --database_to_train_with fever_lex

```


To train on FEVER delexicalized data (mentioned as OANER in the paper), run the following command in the folder `pytorch/` :

``` 
./get_glove.sh
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_delex_oaner_4labels.jsonl  -O data/rte/fever/train/fever_train_delex_oaner_4labels.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_delex_oaner_split_4labels.jsonl  -O data/rte/fever/dev/fever_dev_delex_oaner_4labels.jsonl
python main.py --run_type train --database_to_train_with fever_delex

```


To train on MNLI lexicalized data run the following command in the folder `pytorch/` :

``` 
./get_glove.sh
mkdir -p data/rte/mnli/
mkdir -p data/rte/mnli/train/
mkdir -p data/rte/mnli/dev/
mkdir -p data/rte/mnli/test/
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_train.jsonl -O data/rte/mnli/train/mnli_train.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mu_matched.jsonl  -O data/rte/mnli/dev/mnli_dev.jsonl
python main.py --run_type train --database_to_train_with mnli_lex
```
## Note: To train on other data input files, you just need to replace the corresponding file names with the list given below.
train partition of mnli delexicalized with oaner: mnli_train_delex_oaner.jsonl

dev partition of mnli delexicalized with oaner : mnli_dev_delex_oaner

For example if you want to train (and dev) on a delexicalized-with-oaner-MNLI the corresponding commands 
(only ones that will be different) will be:

```
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mnli_train_delex_oaner.jsonl -O data/rte/mnli/train/mnli_train.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/mnli/mnli_dev_delex_oaner.jsonl  -O data/rte/mnli/dev/mnli_dev.jsonl
```

## Notes
- You can keep track of the training and dev accuracies by doing `tail -f mean_teacher.log` 
- The trained model will be stored under `/model_storage/best_model.pth ` 


- Note that in this particular case the file train_full_with_evi_sents is a collection of all claims and the corresponding
 evidences in the training data of [FEVER](http://fever.ai/) challenge. This is not available in public unlike the FEVER data. 
 This is the output of the IR module of FEVER baseline [code](http://fever.ai/task.html).
 
 - The glove file kept at `data-local/glove/glove.840B.300d.txt` is a very small version of the actual glove file. You might want to replace it with the actual 840B [glove file](https://nlp.stanford.edu/projects/glove/)

 - I personally like/trust `pip install ` instead of `conda install`  because the repos of pip are more comprehensive

 - The code expects to find the data in specific directories inside the data-local directory.  For example some sample training and dev is kept here: `pytorch/data-local/rte/fever/`. Also you will have to replace the sample data with the the actual [train](https://drive.google.com/open?id=1bA32_zRn8V2voPmb1sN5YbLcVFo6KBWf) and [dev](https://drive.google.com/open?id=1xb6QHfMQUI3Q44DQZNVL481rYyMGN-sR) files from google drive


 - The code for decomposable attention is forked from [here](https://github.com/libowen2121/SNLI-decomposable-attention)