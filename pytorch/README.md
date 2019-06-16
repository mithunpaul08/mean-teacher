
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
\* **note**: for conda install get the right command from the pytorch [homepage](https://pytorch.org/) based on your OS and configs.

*PS: I personally like/trust `pip install *` instead of `conda install` * because the repos of pip are more comprehensive


The code expects to find the data in specific directories inside the data-local directory. So do remember to 
 add the data before you run the code.
 
 For example the data for RTE-FEVER is kept here:

```
/data-local/rte/fever/train/train_full_with_evi_sents.jsonl
```
Note that in this particular case the file train_full_with_evi_sents is a collection of all claims and the corresponding
 evidences in the training data of [FEVER](http://fever.ai/) challenge. This is not available in public unlike the FEVER data. 
 This is the output of the IR module of FEVER baseline [code](http://fever.ai/task.html).

To train on FEVER, run e.g.:


``` 
python -u main.py 
--dataset fever 
--arch simple_MLP_embed_RTE 
--pretrained_wordemb false 
--update_pretrained_wordemb true
--epochs 2
--consistency 1
--run-name fever_transform
--data_dir data-local/rte/fever
--train_input_file  train_small_200_claims_with_evi_sents.jsonl 
--dev_input_file dev_90_with_evi_sents.jsonl
--print-freq 1
--workers 0
--run_student_only false
--batch_size 100
--labeled_batch_size 25
--labels 20.0
--consistency 1

```


# explanation of command line parameters

`--workers`: if you dont want multiprocessing make workers=0

`--run_student_only false` means : Run the code as both teacher and student. Difference here is 
that when running both, first the dataset is divided into labeled and unlabeled based on the percentage you mentioned in `--labels`.
now instead if you just want to work with labeled data: i.e supervised training. i.e you don't want to run mean teacher for some reason: then you turn this on/true.

 If you are doing `--run_student_only true` i.e to run as only a student (which might internally have say a simple feed forward supervised network)
  with all data points having labels, you shouldn't pass any of these argument parameters which are meant for mean teacher.

```
--labeled_batch_size
--labels
--consistency
```

#### reason: 


`--labels`: is the percentage or number of labels indicating the number of labeled data points amongst the entire training data. If its int, the code assumes that you are
passing the actual number of data points you want to be labeled. Else
if its float, the code assumes it is a percentage value.

`--labeled_batch_size` : say 20% (`--labels`) of the full data you are dropping the labels for/marking
as unlabeled. You still have a batch size (as specified in `----batch_size`). `--labeled_batch_size` is an int value
which says how many of the data points within a batch do you want to be marked as unlabeled.
Eg: Say there are 200 data points in your training batch and your command line looks like this:
`--labels 20.0 ----batch_size 32 --labeled_batch_size 10`. This means that, 20% of the labels (i.e 40 data points) overall will be marked as
unlabeled. Now all these 400 data points will be divided into approximately 12 batches with each batch containing 32 data points. Within this
32, atleast 10 of them will be marked as unlabeled, and the rest 22 will be left as is, i.e labeled.

Note: If you want to run both student and teacher, but without dropping any labels, you must not have any of these:
```
--labeled_batch_size
--labels
--consistency
```

i.e you should use only `----batch_size` from the 3 above . ALso you must make `--run_student_only false`

so the total command will look like
```
--dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6 --run-name fever_transform --batch_size 10 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file dev_with_50_evi_sents.jsonl --train_input_file train_with_100_evi_sents.jsonl --arch da_RTE --log_level INFO --use_gpu false --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --run_student_only false --labeled_batch_size 10
```

Further details of other command line parameters can be found in `pytorch/mean_teacher/tests/cli.py`



- also, make sure the value of `--labels` is removed.

- also note that due to the below code whenever `run_student_only=true`, 
the sampler becomes a simple `BatchSampler`, and not a `TwoStreamBatchSampler` (which is
the name of the sampler used in mean teacher).

```

if args.run_student_only:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)

elif args.labeled_batch_size:
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)

```

# Testing
To do testing (on dev or test partition), you need to run the code again with `--evaluate` set to `true`. i.e training and testing uses same code but are mutually exclusive. You cannot run testing immediately after training.
You need to finish training and use the saved model to do testing.

Use `python main.py --help` to see other command line arguments.

To reproduce the CIFAR-10 ResNet results of the paper run `python -m experiments.cifar10_test` using 4 GPUs.

To reproduce the ImageNet results of the paper run `python -m experiments.imagenet_valid` using 10 GPUs.

Note to anyone testing from clulab (including myself, mithun). Run on 
server:clara.
- cd meanteacher
- tmux
- git pull
- source activate meanteacher
- run one of the linux commands given [here](#commands_to_run)   

Note:
 
- the commands must be run at the folder level which has main.py eg:`mean-teacher/pytorch/main.py`
- from the same level you can keep track of the logs by doing `tail -f meanteacher.log`
- there will be a time delay of around 3 minutes while the vocabulary is created. The log file will have an entry just before that saying : `INFO:datasets:Done loading embeddings. going to create vocabulary ...`
    
#FAQ :
*These are questions I had when i was trying to load the mean teacher project. Noting it down for myself and for the sake of others who might end up using this code.*

#### Qn) What does transform() do?

Ans: `transform` decides what kind of noise you want to add. 
For example the class `RTEDataset` internally calls the function fever() in the file
`mean_teacher/datasets.py` which in turn call `data.RandomPatternWordNoise` from the file `mean-teacher/pytorch/mean_teacher/data.py`
Both the student and teacher will have different type of noise added. That is decided by transform.
Bottom line is: I don't know how the function fever() is called by the class RTEDataset. It is some kind of internal pytorch thing am assuming. But the function
fever() is where you specify what kinda tranformations you need. So if you want to turn on noise for your input data you uncomment:
`#'train_transformation': data.TransformTwiceNEC(addNoise),` in dataset.py. Don't look at me, I just inherited this code from someone else.

as of may 2019 the function fever() mentioned above can be found [here](https://github.com/mithunpaul08/mean-teacher/blob/add_decomp_attn/pytorch/mean_teacher/datasets.py#L22)

```
    def ontonotes():
    if NECDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(NECDataset.NUM_WORDS_TO_REPLACE, NECDataset.OOV, NECDataset.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(NECDataset.WORD_NOISE_TYPE)

    return {
        'train_transformation': data.TransformTwiceNEC(addNoise),
        'eval_transformation': None,
        'datadir': 'data-local/nec/ontonotes',
        'num_classes': 11
    }
    
```

#### Qn) What is ema? so the ema is teacher? and teacher is just a copy of student itself-but how/where do they do the moving average thing?

Ans: Yes. ema is exponetial moving average. This denotes the teacher. So whenever you want to create a techer, just make ema=True in:
 ```
 model = create_model()
    ema_model = create_model(ema=True)
```

#### Qn) I get this below error. What does it mean?

```
 File "/anaconda3/envs/meanteacher/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 224, in default_collate
    return torch.LongTensor(batch)
TypeError: an integer is required (got type str)


```

Ans: you are passing labels as string. Create a dictionary to make it an int.

#### Qn) what is `__getitem__`? who calls it?

Ans: This is a pytorch internal function. The train function in main.py calls it as:

`train(train_loader, model, ema_model, optimizer, epoch, dataset, training_log)`

#### Qn) In the mean teacher paper I read that there is no backpropagation within the teacher. However, where exactly do they achieve it ? the teacher is looks like a copy of the same student model itself right?

Ans: They do it in this code below in main.py

```
        if ema:
            for param in model.parameters():
                param.detach_() ##NOTE: Detaches the variable from the gradient computation, making it a leaf .. needed from EMA model



```

#### Qn) what does the below code in main.py do?
```if args.run_student_only:```

Ans: If you want to use the mean teacher as a simple feed forward network. Note that the
main contribution of valpola et al is actually the noise they add. However if you just want
to run the mean teacher as two parallel feed forward networks, without noise, but still with 
consistency cost, just turn this on: `args.run_student_only`
    
 
#### Qn) what models can the student/teacher contain? LSTM?

Ans: it can have anything ranging from a simple feed forward network to an LSTM. In 
the file `mean_teacher/architectures.py` look for the function class `FeedForwardMLPEmbed()`. That takes two inputs (eg:claim, evidence or entity, patterns) .
Similarly the class `class SeqModelCustomEmbed(nn.Module):` does the same but for LSTM.
 
 
 #### Qn) what does the below code do in datasets.py?
 `if args.eval_subdir not in dir:`

Ans: this is where you decide whether you want to do training or testing/eval/dev. Only difference
between training and dev is that, there is no noise added in dev.

**Qn) I see a log file is being created using `LOG = logging.getLogger('main')`. But I can't see any files. Where is the log file stored?**

Ans: Its printed into `stdout` by default. Alternately there is this log file which is
time stamped and logs all the training epoch parameters etc. It is done using `meters.update('data_time', time.time() - end)` in main.py
It is stored in the folder `/results/main`.
Update: found out that meters is just a dictionary. Its not printing anything to log file. 
all the `meters.update` are simply feeding data into the dictionary. You can print it using log.info as shown below

```
LOG.info('Epoch: [{0}][{1}/{2}]\t'
                    'ClassLoss {meters[class_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@2 {meters[top5]:.3f}'.format(
                        epoch, i, len(train_loader), meters=meters))
```




**Qn) I see one of my labels is -1. I clearly marked mine from [0,1,2]?**

Ans: Whenever the code removes a label  (for the mean teacher purposes) it assigns a label of -1

**Qn) Why do I get an error at `assert_exactly_one([args.run_student_only, args.labeled_batch_size])` ?**

Ans: If you are doing `--run_student_only true` i.e to run mean teacher as a simple feed forward supervised network
with all data points having labels, you shouldn't pass any of these argument parameters which are meant for mean teacher.
```

--labeled_batch_size 10
```
also make `--labels 100`


Qn) what exactly is shuffling, batching, sampler, pin_memory ,drop_last etc?

Ans: these are all properties of the pytorch dataloader class . 
Even though the official   tutorial is 
[this](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) one I really liked the [the stanford tutorial on dataloader](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)

also look at the  [source code](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html)
 and [documentation](https://pytorch.org/docs/0.4.0/data.html#torch.utils.data.DataLoader)
  of dataloader class

Qn) check what embedding is the libowen code (the code for decomposable attention) loading?

Ans: he is  using glove. he just names it function w2v. i  also checked the
    harvard code (the one which generates the hdf5 files which are inturn used by libowen). they use glove only to create hdf5
    
Qn)does libowen code have momentum? 
  Ans: no
    
    
# Todo :
- find embedding value of same word in both libowen and your code
    - tried printing the embedding of first value after loading glove. it was -1. i thought it was -1 because our local machine couldnt' find it. but its all -1s in server also. that is a problem.
    - words are ok. first 10 words after pad emb etc had embeddings.
    - print the first sentences inside forward() of inter_Attention code in both valpola and libowen
- compare line by line libowen vs my code
    - done until line 77 in libowen's `train_baseline_snli.py` 
- where is he using the glove embedding size?
    - ideally specifying embedding size must be done in hdf5 file creation
- is he doing gigaword normalization in harvard code for hdfs?
- will doing normalization change accuracy value?
- are you updating he is not
	- is he doing drop out
	- is he handling low frequency words
	- debug line by line and make sure all sizes and lengths especially w2v match
- compare command line input with libowen cli command line
- replace batch size as 32 which libowen uses
- go to allennlp +fever's [json file](https://github.com/mithunpaul08/decomp_attn_fever/blob/master/experiments/decomp_attn.json) and try to replicate the parameters here
- add/hardcode/randomly initialize an embedding for `</s>` also after you enable transform. right now it is taking that of `<unk>`
- why are we doing prediction before loss.backward? -confirm if libowen does it
- implement early stopping +prediction
#  marco
- batch average- which one to take...sum all individual per point average/divided by- refer my code
    - done. marco said for dev doesn't matter. infact i verified using both methods, i.e amassing claims and evidences vs amassing predictions. both gave same results 
- how do we know model() is trained, vs model_out. atleast forward, explicitly returns stuff...line 408- same pass by reference thing?


# parameters in the two layers

=========================
embedding.weight                                   5656 * 300 =   1,696,800
input_linear.weight                                 200 * 300 =      60,000
===========================================================================
all parameters count=2                           sum of above =   1,756,800

INFO:main:
List of model parameters:
=========================
mlp_f.1.weight                                      200 * 200 =      40,000
mlp_f.1.bias                                              200 =         200
mlp_f.4.weight                                      200 * 200 =      40,000
mlp_f.4.bias                                              200 =         200
mlp_g.1.weight                                      200 * 400 =      80,000
mlp_g.1.bias                                              200 =         200
mlp_g.4.weight                                      200 * 200 =      40,000
mlp_g.4.bias                                              200 =         200
mlp_h.1.weight                                      200 * 400 =      80,000
mlp_h.1.bias                                              200 =         200
mlp_h.4.weight                                      200 * 200 =      40,000
mlp_h.4.bias                                              200 =         200
final_linear.weight                                   3 * 200 =         600
final_linear.bias                                           3 =           3
===========================================================================
all parameters count=14                          sum of above =     321,803

**update: on april 12th 2019, becky suggested to match the batch size =20 that libowen was having, and guess what**
#I have a dev stable accuracy of around 83%

```INFO:main:
Dev Epoch: [30][754/755]      Dev Classification_loss:0.9863 (0.0000) Dev Prec_model: 50.000 (82.244)
INFO:main:*************************
dev_prec_cum_avg_method:82.24401160381268,
dev_prec_accumulate_pred_method :82.65230004144219
best_dev_accuracy_so_far:83.48114380439287,
best_epoch_so_far:18

training accuracy @epoch 30: 84.57166666666667,dev: 82.65230004144219
```


# commands_to_run
##### Some linux versions of the start up command*

Below is a version that runs on mean teacher on a mac command line-but with toy data- best for laptop:
```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 6  --run-name fever_transform --batch_size 20 --labels 20.0 --data_dir data-local/rte/fever --print_freq 1 --workers 0 --labeled_batch_size 5 --consistency 1 --dev_input_file dev_90_from_train_big145k.jsonl --train_input_file train_small_200_claims_with_evi_sents.jsonl
```
Below is a version that runs the code as a simple FFNN on a mac command line-but with toy data- best for laptop:
```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 1 --run-name fever_transform --data_dir data-local/rte/fever --train_input_file  train_small_200_claims_with_evi_sents.jsonl --dev_input_file dev_90_with_evi_sents.jsonl --workers 0 --run_student_only true --batch_size 20 --lr 0.0000001 --ema_decay 8 --print_freq 1

```
Below is a version that runs the code as a **decomposable attention** given [here](https://github.com/mithunpaul08/SNLI-decomposable-attention) 
inside the student only on a **mac command** line-but with toy data- best for laptop:

```
--dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6  --run-name fever_transform --batch_size 20 --labels 20.0 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file dev_90_from_train_big145k.jsonl --train_input_file train_small_200_claims_with_evi_sents.jsonl --arch da_RTE --run_student_only true --log_level INFO --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true
```

```--dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6  --run-name fever_transform --batch_size 20 --labels 20.0 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file fn_dev_ner_neutered_10.jsonl --train_input_file train_with_100_evi_sents.jsonl --arch da_RTE --run_student_only true --log_level INFO --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true
```

same (lexicalized data.) on mac, but with teacher trained on.
```
--dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6  --run-name fever_transform --batch_size 20 --data_dir data-local/ --print_freq 1 --workers 0 
--dev_input_file fn_dev_ner_neutered_10.jsonl --train_input_file train_with_100_evi_sents.jsonl 
--arch da_RTE --run_student_only false --log_level INFO --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true
```
same, on mac, but train on fever, test on fnc dev
```
--dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6  --run-name fever_transform --batch_size 20 --labels 20.0 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file fn_dev_ner_neutered_10.jsonl --train_input_file fever_training_NER_replaced_100.jsonl --arch da_RTE --run_student_only true --log_level INFO --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --type_of_data ner_replaced
```


Below is a version that runs the code as a **decomposable attention** given [here](https://github.com/mithunpaul08/SNLI-decomposable-attention) 
inside the student only on a mac command line-but with data that is NER neutered 
```
--dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6  --run-name fever_transform --batch_size 20 --labels 20.0 --data_dir data-local/ --print_freq 1 --workers 0 
--arch da_RTE --run_student_only true --log_level INFO --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --type_of_data ner_replaced
```

Below is a version that runs the code as a **decomposable attention**  as **mean teacher** 
 on a mac command line- using this on may16th 2019.

```python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 6 --run-name fever_transform --batch_size 10 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 0 --dev_input_file dev_with_50_evi_sents.jsonl --train_input_file train_with_100_evi_sents.jsonl --arch da_RTE --log_level DEBUG --use_gpu false --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --run_student_only true --labeled_batch_size 5 --labels 20.0 --consistency 1```

#### from here on every command is for a server machine (defined as a computer that has huge memory/huge gpu/huge disk space)
Below is a version that runs on linux command line (server/big memory-but with 12k training and 2.5k dev):

```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 100 --consistency 1 --run-name fever_transform --data_dir data-local/rte/fever --train_input_file  train_12k_with_evi_sents.jsonl --dev_input_file dev_2k_with_evi_sents.jsonl --print-freq 1 --workers 4 --consistency 1 --run_student_only false --batch_size 1000 --labeled_batch_size 100 --labels 20.0 
```
Below is a version that runs **mean teacher** on a linux command line (server/big memory:145k training 10k dev- ACTUAL fever competition data):
```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 100 --run-name fever_transform --data_dir data-local/rte/fever --train_input_file  train_full_with_evi_sents.jsonl --dev_input_file actual_fever_dev_with_9k.jsonl --print_freq 1 --workers 4 --consistency 8 --run_student_only false --batch_size 2000 --labeled_batch_size 1000 --labels 20.0 --lr=0.1 --ema_decay 0.999  
```

Below is a version that runs **FFNN** on linux command line (server/big memory:120k training 25k dev) -i.e: --run_student_only true
``` 
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 500 --run-name fever_transform --data_dir data-local/rte/fever --train_input_file  train_120k_with_evi_sents.jsonl --dev_input_file actual_fever_dev_with_9k.jsonl --print_freq 1 --workers 4 --run_student_only true --batch_size 2000 --lr 0.1      
 
```
Below is a version that runs **FFNN** on linux command line (server/big memory:145k training 10k dev) -i.e: --run_student_only true
``` 
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 500 --run-name fever_transform --data_dir data-local/rte/fever --train_input_file  train_full_with_evi_sents.jsonl --dev_input_file actual_fever_dev_with_9k.jsonl --print_freq 1 --workers 4 --run_student_only true --batch_size 2000 --lr 0.1 
```     

Below is a version that runs **Decomposable Attention** on linux command line (server/big memory:12k training 2k dev) student only -i.e: --run_student_only true
use conda environment: meanteacher in clara

``` 
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 50 --run-name fever_transform --batch_size 10 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 4 --train_input_file  train_12k_with_evi_sents.jsonl --dev_input_file dev_2k_with_evi_sents.jsonl --arch da_RTE --run_student_only true  --run_student_only true --log_level INFO --use_gpu True --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true
       
```
Below is a version that runs **Decomposable Attention** on linux command line (server/big memory-but with 120k training and 24k dev) student only -i.e: --run_student_only true
use conda environment: meanteacher in clara **and gave 82% accuracy, highest so far**

``` 
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 100 --run-name fever_transform --batch_size 32 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 4 --train_input_file  train_120k_with_evi_sents.jsonl --dev_input_file dev_24K_no_train_120k_overlap.jsonl --arch da_RTE --run_student_only true  --run_student_only true --log_level INFO --use_gpu True --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true  
```

Below is a version that runs the code as a **decomposable attention** as the student only version of a mean teacher on a linux
machine with 119197 lines in training data and 26252 lines in dev data-but with data that is NER neutered 
```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 100 --run-name fever_transform --batch_size 32 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 4 --train_input_file  fever_training_smartner_converted.jsonl --dev_input_file fever_dev_smartner_converted.jsonl --arch da_RTE --run_student_only true  --run_student_only true --log_level INFO --use_gpu True --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --type_of_data ner_replaced

```

also, here is the same one as above, instead does training on fnc, and dev on fever. Also glove will be loaded from a hardcoded path
```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 100 --run-name fever_transform --batch_size 32 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 4 --train_input_file  fnc_train_mithun_modified_with_ner_replacement.jsonl --dev_input_file fever_dev_smartner_converted.jsonl --arch da_RTE --run_student_only true  --run_student_only true --log_level INFO --use_gpu True --pretrained_wordemb_file /work/mithunpaul/meanteacher/pytorch/data-local/glove/glove.840B.300d.txt --use_double_optimizers true --type_of_data ner_replaced --use_local_glove False
```

same as above but training on fever and testing on fnc dev

```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 100 --run-name fever_transform --batch_size 32 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 4 --train_input_file  fever_training_smartner_converted.jsonl --dev_input_file fn_dev_smartner_neutered.jsonl --arch da_RTE --run_student_only true  --run_student_only true --log_level INFO --use_gpu True --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --type_of_data ner_replaced
```
Below is a version that runs the code as a **decomposable attention** as both student and teacher.

```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb true --update_pretrained_wordemb false --epochs 100 --run-name fever_transform --batch_size 32 --lr 0.005 --data_dir data-local/ --print_freq 1 --workers 4 --train_input_file  train_120k_with_evi_sents.jsonl --dev_input_file dev_24K_no_train_120k_overlap.jsonl --arch da_RTE --log_level INFO --use_gpu True --pretrained_wordemb_file glove.840B.300d.txt --use_double_optimizers true --run_student_only false --labeled_batch_size 25 --labels 20.0 --consistency 1
```

status as of june 2nd ,.11pm: 
- code now works for both teacher and student.

- next todo:
    - most important first thing to do: go through main.py and remove all vestigial/old comments. 
    - old/wrong comments are worse than no comments
    - merge with master
    - run student teacher on server with complete lex data? - deadline june 3rd (running on tmux 1 and 0 on amy)
        - update: at epoch 49 best dev accuracy is at 82.70 at epoch 1..
        - search on vim for best_dev_accuracy set ignorecase.
    - turn on noise/self.transform (this is to check my idea of using noise in a new domain + very less labeled data) - june 4th
            -  update: best dev accuracy is 83.29 in epoch 1
    - turn on noise/dropping labels (i.e mean teacher) one at a time.
        - turned off noise in function fever() in datasets.py
            - left label dropping on- i.e it is running both student and teacher
            - created a  new branch mithun_run_tests_lex
            - its running on tmux 2, watching on tmux 3. @2pm Thursday 13th june 2019
        - 
    
    - create another student and feed in lex into student1 and delex into student2 (this is mihai's idea of 2 mean teachers. check drawing from april)- june 30th
        - feed in the four class split up
        - the current existing student, rename it to student 1
        - create a new model in create_model() for student 2
        - create a create_data_loader for student2
            - separate out the function   
            - feed the lexicalized data into train loader(this we are already doing)
            - dont' do droppings of words/adding noise - i.e turn transform off
            - create a new train loader for student2
            - feed delex into this train loader of student2 and lex into train loader of student 1
            - add an if condition for student1 and student2 and pass two different train loaders into train() function
    - feed in lex into student and delex into student2 and also attach a teacher (this is mihai's idea of 2 mean teachers. check drawing from april)- 
        