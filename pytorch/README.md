
# Fact Verification using Mean Teacher in PyTorch

This is the PyTorch source code for the Mean Teacher paper. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install numpy scipy pandas sklearn nltk tqdm
pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119
conda install pytorch-cpu torchvision-cpu -c pytorch 

*note: for conda install get the right command from the pytorch home page based on your OS and configs.*

```

*PS: I personally like/trust `pip install *` instead of `conda install`*


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
--consistency 1
--exclude_unlabeled false
--batch_size 100
--labeled_batch_size 25
--labels 20.0


```
**removed labels**\
`--exclude_unlabeled true`\ (refer below)

`-- dev_input_file dev_90_with_evi_sents.jsonl` (use this when testing on a local machine/laptop with small ram)

`--train_input_file train_small_200_claims_with_evi_sents.jsonl`

`--train_input_file  train_12k_with_evi_sents.jsonl -- dev_input_file dev_2k_with_evi_sents.jsonl`

**Some linux versions of the start up command**

Below is a version that runs on linux command line (local machine/laptop):**
```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 6 --consistency=0.3 --run-name fever_transform --batch_size 10 --labels 20.0 --data_dir data-local/rte/fever --print-freq 1 --workers 0 --labeled_batch_size 2 --consistency 35.5 --dev_input_file dev_90_with_evi_sents.jsonl --train_input_file train_small_200_claims_with_evi_sents.jsonl
```
Below is a version that runs on linux command line (server/big memory-but with 12k training and 2.5k dev):

```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 100 --consistency 1 --run-name fever_transform --data_dir data-local/rte/fever --train_input_file  train_12k_with_evi_sents.jsonl --dev_input_file dev_2k_with_evi_sents.jsonl --print-freq 1 --workers 4 --consistency 1 --exclude_unlabeled false --batch_size 1000 --labeled_batch_size 100 --labels 20.0 
```
Below is a version that runs on linux command line (server/big memory:120k training 25k dev):
```
python -u main.py --dataset fever --arch simple_MLP_embed_RTE --pretrained_wordemb false --update_pretrained_wordemb true --epochs 6 --consistency=0.3 --run-name fever_transform --batch_size 1000 --labels 20.0 --data_dir data-local/rte/fever --print-freq 1 --workers 4 --labeled_batch_size 250 --consistency 35.5 --dev_input_file dev_25k_with_evi_sents.jsonl --train_input_file train_120k_with_evi_sents.jsonl
```

#explanation of command line parameters

`--workers`: if you dont want multiprocessing make workers=0

`--exclude_unlabeled true` means : first dataset is divided into labeled and unlabeled based on the percentage you mentioned in `--labels`.
now instead if you just want to work with labeled data: i.e supervised training. i.e you don't want to run mean teacher for some reason: then you turn this on/true.


 If you are doing --exclude_unlabeled true i.e to run mean teacher as a simple feed forward supervised network with all data points having labels, you shouldn't pass any of these argument parameters which are meant for mean teacher.

```
--labeled_batch_size
--labels
```
also, make sure the value of `--labels` is removed.

also note that due to the below code whenever `exclude_unlabeled=true`, 
the sampler becomes a simple `BatchSampler`, and not a `TwoStreamBatchSampler` (which is
the name of the sampler used in mean teacher).

```

if args.exclude_unlabeled:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)

elif args.labeled_batch_size:
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)

```



`--labels`: is the percentage or number of labels indicating the number of labeled data points amongst the entire training data. If its int, the code assumes that you are
passing the actual number of data points you want to be labeled. Else
if its float, the code assumes it is a percentage value.


Further details of other command line parameters can be found in `pytorch/mean_teacher/tests/cli.py`


#Testing
To do testing (on dev or test partition), you need to run the code again with `--evaluate` set to `true`. i.e training and testing uses same code but are mutually exclusive. You cannot run testing immediately after training.
You need to finish training and use the saved model to do testing.

Use `python main.py --help` to see other command line arguments.

To reproduce the CIFAR-10 ResNet results of the paper run `python -m experiments.cifar10_test` using 4 GPUs.

To reproduce the ImageNet results of the paper run `python -m experiments.imagenet_valid` using 10 GPUs.

Note to anyone testing from clulab (including myself, mithun). Run on 
server:clara.
- tmux
- git pull
- source activate meanteacher
- one of the linux commands above    

# FAQ :
*These are questions I had when i was trying to load the mean teacher project. Noting it down for myself and for the sake of others who might end up using this code.*

#### Qn) What does transform() do?

Ans: `transform` decides what kind of noise you want to add. 
For example the class `NECDataset` internally calls the function ontonotes() in the file
`mean_teacher/datasets.py`. 
Both the student and teacher will have different type of noise added. That is decided by transform


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
```if args.exclude_unlabeled:```

Ans: If you want to use the mean teacher as a simple feed forward network. Note that the
main contribution of valpola et al is actually the noise they add. However if you just want
to run the mean teacher as two parallel feed forward networks, without noise, but still with 
consistency cost, just turn this on: `args.exclude_unlabeled`
    
 
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

**Qn) Why do I get an error at `assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])` ?**

Ans: If you are doing `--exclude_unlabeled true` i.e to run mean teacher as a simple feed forward supervised network
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

