
# Fact Verification using Mean Teacher in PyTorch

This is the PyTorch source code for the Mean Teacher paper. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install numpy scipy pandas sklearn nltk tqdm
pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119
conda install pytorch-cpu torchvision-cpu -c pytorch

```

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
python main.py 
--dataset fever 
--arch simple_MLP_embed_RTE 
--pretrained_wordemb false 
--update_pretrained_wordemb true
--epochs 6
--consistency=0.3 
--run-name fever_transform
--batch_size 82
--labeled_batch_size 2
--labels 20.0
--data_dir data-local/rte/fever
--train_input_file train_small_100_claims_with_evi_sents.jsonl
--dev_input_file dev_90_with_evi_sents.jsonl
--print-freq 1
--workers 0

```

For the above command to run from linux command line, you might need to add a backwards slash as shown below:
```
python main.py \
--dataset fever 




```
#explanation of command line paramaeters

`--workers`: if you dont want multiprocessing make workers=0

`--exclude_unlabeled true` means : first dataset is divided into labeled and unlabeled based on the percentage you mentioned in `--labels`.
now instead if you just want to work with labeled data: i.e supervised training. i.e you don't want to run mean teacher for some reason: then you turn this on/true.
also note that due to the below code whenever exclude_unlabeled=true, 
the sampler becomes a simple batch sampler, and not a TwoStreamBatchSampler (which is
the name of the sampler used in mean teacher).

*Note:* when you are running validation on dev after training, you might want to turn  `--exclude_unlabeled true` just
after you call train_loader before calling eval_loader

```

if args.exclude_unlabeled:
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)

elif args.labeled_batch_size:
            batch_sampler = data.TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)

```



--labeled_data_percent: is the percentage or number of labels indicating the number of labeled data points amongst the entire training data.


Details of other command line parameters can be found in `pytorch/mean_teacher/tests/cli.py`

note to self: initially we are using a dataset of 100 only of which 20% are only labeled. So try to keep the --labeled_batch_size 10 --batch_size 40


#Testing
To do testing (on dev or test partition), you need to run the code again with `--evaluate` set to `true`. i.e training and testing uses same code but are mutually exclusive. You cannot run testing immediately after training.
You need to finish training and use the saved model to do testing.

Use `python main.py --help` to see other command line arguments.

To reproduce the CIFAR-10 ResNet results of the paper run `python -m experiments.cifar10_test` using 4 GPUs.

To reproduce the ImageNet results of the paper run `python -m experiments.imagenet_valid` using 10 GPUs.

# FAQ :
### these are questions I had when i was trying to load the mean teacher project. Noting it down for myself
and for the sake of others who might end up using this code.

Qn) What does transform() do?

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

Qn) What is ema? so the ema is teacher? and teacher is just a copy of student itself-but how/where do they do the moving average thing?

Ans: Yes. ema is exponetial moving average. This denotes the teacher. So whenever you want to create a techer, just make ema=True in:
 ```
 model = create_model()
    ema_model = create_model(ema=True)
```

Qn) I get this below error. What does it mean?

```
 File "/anaconda3/envs/meanteacher/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 224, in default_collate
    return torch.LongTensor(batch)
TypeError: an integer is required (got type str)


```

Ans: you are passing labels as string. Create a dictionary to make it an int.

Qn) what is `__getitem__`? who calls it?

Ans: This is a pytorch internal function. The train function in main.py calls it as:

`train(train_loader, model, ema_model, optimizer, epoch, dataset, training_log)`

Qn) In the mean teacher paper I read that there is no backpropagation within the teacher. However, where exactly do they achieve it ? the teacher is looks like a copy of the same student model itself right?

Ans: They do it in this code below in main.py

```
        if ema:
            for param in model.parameters():
                param.detach_() ##NOTE: Detaches the variable from the gradient computation, making it a leaf .. needed from EMA model



```

Qn) what does the below code in main.py do?
```if args.exclude_unlabeled:```

Ans: If you want to use the mean teacher as a simple feed forward network. Note that the
main contribution of valpola et al is actually the noise they add. However if you just want
to run the mean teacher as two parallel feed forward networks, without noise, but still with 
consistency cost, just turn this on: `args.exclude_unlabeled`
    
 
Qn) what models can the student/teacher contain? LSTM?

Ans: it can have anything ranging from a simple feed forward network to an LSTM. In 
the file `mean_teacher/architectures.py` look for the function class `FeedForwardMLPEmbed()`. That takes two inputs (eg:claim, evidence or entity, patterns) .
Similarly the class `class SeqModelCustomEmbed(nn.Module):` does the same but for LSTM.
 
 
 Qn) what does the below code do in datasets.py?
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


Qn) Why do I get an error at `assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])` 

Ans: If you are doing `--exclude_unlabeled true` i.e to run mean teacher as a simple feed forward supervised network
with all data points having labels, you shouldn't pass any of these argument parameters
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
# Todo Sun Jan 20 21:31:41 MST 2019:


- replace labels with int --done
- replace feed forward with that of ajay-2 classes --done
- get training to run without noise --done
- add transformation(i.e diff noise for student and teacher) --done
- get training to run with noise --done
- read the readme of original code again. the part where they talk about twostreamsampler --done
- check if consistency loss works, now that we have noise ---done
- add eval data 
  - (verify manually dataset_test has size of 10 or whatever you are feeding)

- add pre-trained embeddings
- read the pytorch documentation on dataloader again
- do 2xfeedforward -i.e make the mean teacher as a simple mlp
- remove low frequency words.
- after all debug purpoes, increase the value of `--workers` to enable multiprocessing

 