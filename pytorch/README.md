
# Fact Verification using Mean Teacher in PyTorch

Here we will take the output of the tagging process and feed it as input to a [decomposable attention](https://arxiv.org/pdf/1606.01933.pdf) based neural network model.

 

# Pre reqs:
 
 The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
conda create --name rte python=3 numpy scipy pandas nltk tqdm
source activate rte
pip install sklearn
pip install jsonlines
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

To test using a model trained on FEVER delexicalized data (mentioned as OANER in the paper), and test on FNC dataset, run the following commands from the folder `pytorch/`. 
```
./get_data_delex.sh
./get_glove_small.sh
./get_model_delex.sh
python main.py --run_type test --database_to_test_with fnc 
```

To test using Allennlp based code:

```
conda create --name rte python=3 
source activate rte
brew install npm
cd allennlp-simple-server-visualization/demo/
npm start
``` 
This should open browser and run the GUI in localhost:3000.

Now open another terminal and do:
```
conda create --name rte_run python=3 
source activate rte_run
cd allennlp-as-a-library-example/
pip install -r requirements.txt
python -m allennlp.service.server_simple \
  --archive-path tests/fixtures/FeverModels/Smart_NER/decomposable_attention.tar.gz \
  --predictor drwiki-te\
  --include-package my_library \
  --title "Academic Paper Classifier" \
  --field-name title \
  --field-name paperAbstract
```
If every thing runs fine you will see `Model loaded, serving demo on port 8000`. Now go back to the browser window
`localhost:3000` and just click `Run` (don't enter anything in the claim or evidence fields). This will start testing on the 
given test file. Once the testing is completed you will see an output in the browser with attention weights and labels of the final
input (takes around 15 mins in a MacBookPro with OSX Mojave, 8GB RAM, Intel core i5.)

Now enter:

```
python fnc_official_scorer.py 
```

That should give an output that looks like:

```-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    396    |    86     |    469    |    52     |
-------------------------------------------------------------
| disagree  |    101    |    35     |    183    |    33     |
-------------------------------------------------------------
|  discuss  |   1124    |    239    |    943    |    161    |
-------------------------------------------------------------
| unrelated |    681    |    345    |   3171    |   6037    |
-------------------------------------------------------------
Score: 3433.75 out of 6380.5	(53.81631533578873%)
```
### Testing with other models and masking strategies:
To test with a  model (that was trained on the same FEVER dataset) that was trained using a different masking strategy 
you have to change the corresponding directory path in the command line argument for `--archive-path` to any of the following.

- FullyLexicalized
- NoNER
- SSTagged
- SimpleNER
- Smart_NER

 For example to test with a model that was trained on FEVER dataset but using OANer+SSTagged model the `--archive-path` will be:

`tests/fixtures/FeverModels/SSTagged/decomposable_attention.tar.gz`

Similary to test with a model that was trained on FNC, `--archive-path` will be:

`tests/fixtures/FNCModels/SSTagged/decomposable_attention.tar.gz`

#### Training:

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
