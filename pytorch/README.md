# Fact Verification using Mean Teacher in PyTorch

This is the PyTorch source code for the Mean Teacher paper. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install numpy scipy pandas sklearn nltk
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
--labels 20.0 
--arch simple_MLP_embed_RTE 
--pretrained_wordemb false
--update_pretrained_wordemb true
--epochs 60 
--consistency=0.3 
--run-name log_gids_labels20.0_epochs60_labeled-batch-size64_cons0.3_simple
--labeled_batch_size 10
--batch_size 50


```

note to self: initially we are using a dataset of 100 only of which 20% are only labeled. So try to keep the --labeled_batch_size 10 --batch_size 40

--labeled_data_percent: is the percentage or number of labels indicating the number of labeled data points amongst the entire training data.

Details of other command line parameters can be found in `pytorch/mean_teacher/tests/cli.py`

Use `python main.py --help` to see other command line arguments.

To reproduce the CIFAR-10 ResNet results of the paper run `python -m experiments.cifar10_test` using 4 GPUs.

To reproduce the ImageNet results of the paper run `python -m experiments.imagenet_valid` using 10 GPUs.
