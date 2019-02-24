from __future__ import division
import re
import os,sys
import shutil
import time
import logging
import parser
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torch.cuda


from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
import contextlib
import random

#askfan: where is log file stored? Ans: stdout
logging.basicConfig(filename='example.log',filemode='w+')
LOG = logging.getLogger('main')
LOG.setLevel(logging.INFO)


################
# NOTE: To enable logging on IPythonConsole output or IPyNoteBook
# LOG = logging.getLogger()
# LOG.setLevel(logging.DEBUG)
# LOG.debug("test")

# NOTE: To init args to Mean Teacher :
# parser = cli.create_parser()
# parser.set_defaults(dataset='cifar10') # OR any other param
# args = parser.parse_known_args()[0]
################

args = None
best_accuracy_across_epochs = 0
best_epochs = 0
global_step = 0
NA_label = -1
test_student_pred_match_noNA = 0.0
test_student_pred_noNA = 0.0
test_student_true_noNA = 0.0
test_teacher_pred_match_noNA = 0.0
test_teacher_pred_noNA = 0.0
test_teacher_true_noNA = 0.0

train_student_pred_match_noNA = 0.0
train_student_pred_noNA = 0.0
train_student_true_noNA = 0.0
train_teacher_pred_match_noNA = 0.0
train_teacher_pred_noNA = 0.0
train_teacher_true_noNA = 0.0
###########
# NOTE: To change to a new NEC dataset .. currently some params are hardcoded
# 1. Change args.dataset in the command line
###########

def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)

def create_data_loaders(LOG,train_transformation,
                        eval_transformation,
                        args):
    print(f"got inside create_data_loaders.")
    LOG.debug(f"from log.info inside create_data_loaders.")

    global NA_label
    traindir = os.path.join(args.data_dir , args.train_subdir)
    evaldir = os.path.join(args.data_dir , args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])


    #feb23rd2019: if  args.exclude_unlabeled: we are dropping/not running teacher model. So make sure consistency is 0.
    assert_mutually_exclusive(args.exclude_unlabeled, args.consistency)




    if torch.cuda.is_available():
        pin_memory = True
        LOG.info(f"found torch.cuda is true.")
        print(f"found torch.cuda is true.")
        LOG.info(torch.cuda.device_count())
        cuda0 = torch.cuda.set_device(0)
        LOG.info(torch.cuda.current_device())
        #LOG.info(torch.cuda.get_device_name(0))
        LOG.info(torch.cuda.device_count())
    else:
        pin_memory = False
        LOG.info(f"found torch.cuda is false. giong to exit")

    print(f"found value of args.dataset is {args.dataset}.")


    if args.dataset in ['conll', 'ontonotes','fever']:
        print(f"got inside args.dataset in fever.")



        LOG.info("traindir : " + traindir)
        LOG.info("evaldir : " + evaldir)

        train_input_file = traindir + args.train_input_file
        word_vocab = {"@UNKNOWN": 1,"</s>":2}
        dataset = datasets.RTEDataset(word_vocab,"train",train_input_file, args,train_transformation)
        print(
            f"after reading training dataset.value of word_vocab.size()={len(dataset.word_vocab.keys())}")




        LOG.info("Type of Noise : "+ dataset.WORD_NOISE_TYPE)
        LOG.info("Size of Noise : "+ str(dataset.NUM_WORDS_TO_REPLACE))

        # ans: if you want to do a simple feed forward - i.e ignore all labeled.=args.x_unlabeled=true
        if args.exclude_unlabeled:
            labeled_idxs = data.get_all_label_indices(dataset, args)
            sampler = SubsetRandomSampler(labeled_idxs)
            batch_sampler_local = BatchSampler(sampler, args.batch_size, drop_last=True)
        elif args.labeled_batch_size:
            # askfan what does this relabel_dataset do? Ans: taking the training set and dividing a part of it as labeled and rest as unlabeled (label =-1)
            if args.labels:
                labeled_idxs, unlabeled_idxs = data.relabel_dataset_nlp(dataset, args)
                batch_sampler_local = data.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, args.batch_size,
                                                       args.labeled_batch_size)
        else:
            assert False, "labeled batch size {}".format(args.labeled_batch_size)






                #mithun: pytorch thing. train_loader uses getitem internally-
                # train_loader.next gives you the next mini batch -
                # it picks randomly to create a batch, but it also has to have a minimum:args.batch_size, args.labeled_batch_size
                # for each mini batch: for each data point, it will call __getitem__

            ''' CLASS torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler_local=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)[SOURCE]
            Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.
            
            Parameters:	
            dataset (Dataset) – dataset from which to load the data.
            batch_size (int, optional) – how many samples per batch to load (default: 1).
            shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
            sampler (Sampler, optional) – defines the strategy to draw samples from the dataset. If specified, shuffle must be False.
            batch_sampler_local (Sampler, optional) – like sampler, but returns a batch of indices at a time. Mutually exclusive with batch_size, shuffle, sampler, and drop_last.
            num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
            collate_fn (callable, optional) – merges a list of samples to form a mini-batch.
            pin_memory (bool, optional) – If True, the data loader will copy tensors into CUDA pinned memory before returning them.
            drop_last (bool, optional) – set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size. If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller. (default: False)
            timeout (numeric, optional) – if positive, the timeout value for collecting a batch from workers. Should always be non-negative. (default: 0)
            worker_init_fn (callable, optional) – If not None, this will be called on each worker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None)'''

        found_not_supports_label1=False

        for lbl in dataset.lbl:
            if not (lbl == 2):
                found_not_supports_label1=True



        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_sampler=batch_sampler_local,
                                                   num_workers=args.workers,
                                                   pin_memory=True
                                                  )
                                                  # drop_last=False)
                                                  # batch_size=args.batch_size,
                                                  # shuffle=False)


        #do the same for eval data also. i.e read the dev data, and add a sampler..
        dev_input_file = evaldir + args.dev_input_file
        dataset_dev = datasets.RTEDataset(word_vocab,"dev",dev_input_file, args, eval_transformation) ## NOTE: test data is the same as train data

        found_not_supports_label2=False
        print(
            f"after reading dev dataset.value of word_vocab.size()={len(dataset_dev.word_vocab.keys())}")

        print(f"value of found_not_supports_label={found_not_supports_label1}")


        # debug. exit if gold has any label other than 2.
        for lbl in dataset.lbl:
            if not (lbl == 2):
                found_not_supports_label2=True

        # debug. exit if gold has any label other than 2.
        for lbl in dataset_dev.lbl:
            if not (lbl == 2):
                print(f"\n before eval loader after train loader found a new label in DEV other than SUPPORTS. label is {lbl}")
                import sys
                sys.exit(1)


        sys.exit(1)

        eval_loader = torch.utils.data.DataLoader(dataset_dev,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  drop_last=False,
                                                  num_workers=args.workers)




    #mithun: once you have both the train and test data in the DataLoader format that torch understands, return it to the calling function

    LOG.debug(f"just before return statement inside create_data_loaders. main.py line 229. value of word_vocab.size()={len(dataset.word_vocab.keys())}")





    return train_loader, eval_loader, dataset, dataset_dev

#mithun: this is whe4re they are doing the average thing -ema=exponential moving average
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch, dataset, log):
    global global_step
    global NA_label
    global train_student_pred_match_noNA
    global train_student_pred_noNA
    global train_student_true_noNA
    global train_teacher_pred_match_noNA
    global train_teacher_pred_noNA
    global train_teacher_true_noNA

    if torch.cuda.is_available():
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    else:
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.\
            consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train() ### From the documentation (nn.module,py) :
    # i) Sets the module in training mode.
    # (ii) This has any effect only on modules such as Dropout or BatchNorm. (iii) Returns: Module: self

    #if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
    if not args.exclude_unlabeled:
        ema_model.train()

    end = time.time()

    # datapoint: List(input_student, student_input teacher, labels)
    #i.e go through each data point within a mini batch
    for i, datapoint in enumerate(train_loader):
        # print("len(datapoint) = ", len(datapoint))
        # print("datapoint[0] shape: {0}".format(datapoint[0].shape))
        # sys.exit(1)

        # measure data loading time()
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        len_claims_this_batch = None
        len_evidences_this_batch= None

        if args.dataset in ['conll', 'ontonotes','fever']:

            #if there is no transformation, the data will be inside datapoint[0] itself
            if(dataset.transform) is None:
                student_input = datapoint[0]
                teacher_input = datapoint[0]
                target = datapoint[1]
                len_claims_this_batch = datapoint[2][0]
                len_evidences_this_batch = datapoint[2][1]

            else:
                student_input = datapoint[0]
                teacher_input = datapoint[1]
                target = datapoint[2]
                len_claims_this_batch       = datapoint[3][0]
                len_evidences_this_batch    = datapoint[3][1]



            ## Input consists of tuple (entity_id, pattern_ids)
            student_input_claim = student_input[0]
            student_input_evidence = student_input[1]

            teacher_input_claim = teacher_input[0]
            teacher_input_evidence = teacher_input[1]


            if torch.cuda.is_available():
                claims_var = torch.autograd.Variable(student_input_claim).cuda()
                evidences_var = torch.autograd.Variable(student_input_evidence).cuda()
                # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
                if not args.exclude_unlabeled:
                    ema_claims_var = torch.autograd.Variable(teacher_input_claim, volatile=True).cuda()
                    ema_evidences_var = torch.autograd.Variable(teacher_input_evidence, volatile=True).cuda()

            else:
                claims_var = torch.autograd.Variable(student_input_claim).cpu()
                evidences_var = torch.autograd.Variable(student_input_evidence).cpu()
                # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
                if not args.exclude_unlabeled:
                    ema_claims_var = torch.autograd.Variable(teacher_input_claim, volatile=True).cpu()
                    ema_evidences_var = torch.autograd.Variable(teacher_input_evidence, volatile=True).cpu()



        if torch.cuda.is_available():
            target_var = torch.autograd.Variable(target.cuda())
        else:
            target_var = torch.autograd.Variable(target.cpu())  # todo: not passing the async=True (as above) .. going along with it now .. to check if this is a problem

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        #the feed forward and prediction part happens here.
        if args.dataset in ['fever'] and args.arch == 'simple_MLP_embed_RTE':
            # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
            if not args.exclude_unlabeled:
                ema_model_out = ema_model(ema_claims_var, ema_evidences_var, len_claims_this_batch, len_evidences_this_batch)
            model_out = model(claims_var, evidences_var, len_claims_this_batch, len_evidences_this_batch)



        ## DONE: AJAY - WHAT IS THIS CODE BLK ACHIEVING ? Ans: THIS IS RELATED TO --logit-distance-cost .. (fc1 and fc2 in model) ...
        if isinstance(model_out, Variable):       # this is default
            assert args.logit_distance_cost < 0
            logit1 = model_out
            # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
            if not args.exclude_unlabeled:
                ema_logit = ema_model_out
        else:
            assert len(model_out) == 2
            assert len(ema_model_out) == 2
            logit1, logit2 = model_out
            # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
            if not args.exclude_unlabeled:
                ema_logit, _ = ema_model_out

            # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
        if not args.exclude_unlabeled:
                ema_logit = Variable(ema_logit.detach().data, requires_grad=False) ## DO NOT UPDATE THE GRADIENTS THORUGH THE TEACHER (EMA) MODEL

        if args.logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.data[0])
        else:                                 # this is the default
            class_logit, cons_logit = logit1, logit1    # class_logit.data.size(): torch.Size([256, 56])
            res_loss = 0

        loss_output=class_criterion(class_logit, target_var)

        LOG.debug(f"type of loss_output={type(loss_output)}")
        LOG.debug(f"value of minibatch_size={minibatch_size} ")
        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        LOG.debug(f"value of class_loss={class_loss} ")



        #note by mithun: this was originally class_loss.data[0], but changing to class_loss.data.item() since it was throwing error on [0]
        meters.update('class_loss', class_loss.data.item())

        # note by mithun: this was originally _.data[0], but changing to  _.data.item()since it was throwing error on
        # [0]. this error occurs because we are right now passing data into student and teavher without any transformation. so this change must be temporary

        # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
        if not args.exclude_unlabeled:
            ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size


            ## DONE: AJAY - WHAT IF target_var NOT PRESENT (UNLABELED DATAPOINT) ?
            # Ans: See  ignore index in  `class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()`
            meters.update('ema_class_loss', ema_class_loss.data.item())    # Do we need this?

        #mithun askajay: is this where they are doing consistency comparison-but where is the subtRACTION?
        # #askajay: where is the construction cost and consistency cost and back prop only on student etc?
        #ans: no. they are doing the subtraction around line

        # if you want to use consistency loss with given weight  pass --consistency in running script-
        # consistency has to be a positive value. give =1 they must be equally weighted
        # if you are doing feed forward this can be zero
        if args.consistency:
            # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
            #ideally if we are not doing ema, the args.consistency also must be mutually exclusive
            if not args.exclude_unlabeled:
                consistency_weight = get_current_consistency_weight(epoch)
                meters.update('cons_weight', consistency_weight)
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                meters.update('cons_loss', consistency_loss.data.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        loss = class_loss + consistency_loss + res_loss # NOTE: AJAY - loss is a combination of classification loss and consistency loss (+ residual loss from the 2 outputs of student model fc1 and fc2, see args.logit_distance_cost)

        #todo: note by mithun: uncomment this after we have transform turned on.
        # below line was originally _.data[0], but changing to  _.data.item()since it was throwing error on
        # [0]. this error occurs because we are right now passing data into student and teavher without any transformation. so this change must be temporary
        #assert not (np.isnan(loss.data[0]) or loss.data[0] > 1e5), 'Loss explosion: {}'.format(loss.data[0])
        #meters.update('loss', loss.data[0])

        meters.update('loss', loss.data.item())


        prec1 = accuracy_fever(class_logit.data, target_var.data,LOG) #Note: Ajay changing this to 2 .. since there are only 4 labels in CoNLL dataset
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100. - prec1, labeled_minibatch_size)
        #meters.update('top5', prec5[0], labeled_minibatch_size)
        #meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
        if not args.exclude_unlabeled:
            ema_prec1 = accuracy_fever(ema_logit.data, target_var.data,LOG) #Note: Ajay changing this to 2 .. since there are only 4 labels in CoNLL dataset
            meters.update('ema_top1', ema_prec1, labeled_minibatch_size)
            meters.update('ema_error1', 100. - ema_prec1, labeled_minibatch_size)
            # meters.update('ema_top5', ema_prec5[0], labeled_minibatch_size)
            # me   ters.update('ema_error5', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
        if not args.exclude_unlabeled:
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        #args.print_freq= 10

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            # if you are doing FFNN, just do student alone. don't confuse things with adding teacher model
            if not args.exclude_unlabeled:
                LOG.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Classification_loss:{meters[class_loss]:.4f}\t'
                    'Consistency_loss:{meters[cons_loss]:.4f}\t'
                    'Prec_student: {meters[top1]:.3f}\t'                    
                    'Prec_teacher: {meters[ema_top1]:.3f}\t'
                    'teacher_error: {meters[ema_error1]:.3f}\t'
                    'student_error:{meters[error1]:.3f}\t'
                        .format(
                    epoch, i, len(train_loader), meters=meters))
            else:
                LOG.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Classification_loss:{meters[class_loss]:.4f}\t'                    
                    'Prec_student: {meters[top1]:.3f}\t'
                        .format(
                        epoch, i, len(train_loader), meters=meters))

    log.info("end of all batches in training. going toexit")
    import sys
    sys.exit(1)



def validate(eval_loader, model, log, global_step, epoch, dataset, result_dir, model_type):
    LOG.debug(f"got here inside validate")
    global NA_label
    global test_student_pred_match_noNA
    global test_student_pred_noNA
    global test_student_true_noNA
    global test_teacher_pred_match_noNA
    global test_teacher_pred_noNA
    global test_teacher_true_noNA

    if torch.cuda.is_available():
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    else:
        class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()

    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval() ### From the documentation (nn.module,py) : i) Sets the module in evaluation mode. (ii) This has any effect only on modules such as Dropout or BatchNorm. (iii) Returns: Module: self
    LOG.debug(f"got here after model.eval()")
    end = time.time()

    save_custom_embed_condition = args.arch == 'custom_embed' \
                                  and args.save_custom_embedding \
                                  and epoch == args.epochs  # todo: only in the final epoch or best_epoch ?
    LOG.debug(f"got here after save_custom_embed_condition")


    if save_custom_embed_condition:
        # Note: contains a tuple: (custom_entity_embed, custom_patterns_embed, min-batch-size)
        # enumerating the list of tuples gives the minibatch_id
        custom_embeddings_minibatch = list()
        # eval_loader.batch_size = 1
        # LOG.info("NOTE: Setting the eval_loader's batch_size=1 .. to dump all the claims_dev and pattern embeddings ....")

    LOG.debug(f"inside validate function. value of  eval_loaderis {(eval_loader)}")
    LOG.debug(f"inside validate function. value of  len(eval_loader.dataset.claims) : {len(eval_loader.dataset.claims)}")
    LOG.debug(f"inside validate function. value of  len(eval_loader.dataset.claims) : {len(eval_loader.dataset.claims)}")
    LOG.debug(f"inside validate function. value of  len(eval_loader.sampler.data_source.claims) : {len(eval_loader.sampler.data_source.claims)}")
    LOG.debug(f"inside validate function. value of  len(eval_loader.sampler.data_source.lbl : {len(eval_loader.sampler.data_source.lbl)}")
    LOG.debug(f"inside validate function. value of  eval_loader.batch_size : {(eval_loader.batch_size)}")

    sum_all_acc=0
    total_no_batches=0
    #enumerate means enumerate through each of the batches.
    #the __getitem__ in datasets.py is called here
    for i, datapoint in enumerate(eval_loader):
        LOG.debug(f"got inside .i, datapoint in enumerate(eval_loader)")
        meters.update('data_time', time.time() - end)

        if args.dataset in ['conll', 'ontonotes','fever']:
            LOG.debug(f"got inside args.dataset in fever")

            # claims_dev = datapoint[0][0]
            # evidence_dev = datapoint[0][1]
            # labels_dev = datapoint[1]
            # len_claims_this_batch = datapoint[2][0]
            # len_evidences_this_batch = datapoint[2][1]

            # if there is no transformation, the data will be inside datapoint[0] itself

            '''Note: when you get here from devsomewhere in the code it is internally making self. transform=None.
             So by the time it is reaching here in validate, it will always take the same output. But yeah, good to check
             . Have updated the code with that from train() nevertheless, Just that it always gets into 
             Self.Transform=None branch'''
            if (dataset.transform) is None:
                student_input = datapoint[0]
                labels_dev = datapoint[1]
                len_claims_this_batch = datapoint[2][0]
                len_evidences_this_batch = datapoint[2][1]

            else:
                student_input = datapoint[0]
                labels_dev = datapoint[2]
                len_claims_this_batch = datapoint[3][0]
                len_evidences_this_batch = datapoint[3][1]

            ## Input consists of tuple (entity_id, pattern_ids)
            claims_dev = student_input[0]
            evidence_dev = student_input[1]


            if torch.cuda.is_available():
                claims_var = torch.autograd.Variable(claims_dev, volatile=True).cuda()
                evidence_var = torch.autograd.Variable(evidence_dev, volatile=True).cuda()
                target_var = torch.autograd.Variable(labels_dev.cuda(), volatile=True)
            else:
                claims_var = torch.autograd.Variable(claims_dev, volatile=True).cpu()
                evidence_var = torch.autograd.Variable(evidence_dev, volatile=True).cpu()
                target_var = torch.autograd.Variable(labels_dev.cpu(),
                                                     volatile=True)  ## NOTE: AJAY - volatile: Boolean indicating that the Variable should be used in inference mode,

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        ############## NOTE: AJAY -- changing this piece of code to make sure evaluation does not
        ############## thrown exception when the minibatch consists of only NAs. Skip the batch
        ############## TODO: AJAY -- To remove this later
        # assert labeled_minibatch_size > 0
        if labeled_minibatch_size == 0:
            print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%AJAY: Labeled_minibatch_size == 0 ....%%%%%%%%%%%%%%%%%%%%%%%")
            continue
        ###################################################
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        LOG.debug(f"value of args.arch is:{args.arch}")
        LOG.debug(f"value of args.dataset is:{args.dataset}")

        # compute output
        if args.dataset in ['conll', 'ontonotes'] and args.arch == 'custom_embed':
            output1, entity_custom_embed, pattern_custom_embed = model(claims_var, evidence_var)
            if save_custom_embed_condition:
                custom_embeddings_minibatch.append((entity_custom_embed, pattern_custom_embed))  # , minibatch_size))

        elif args.dataset in ['fever'] and args.arch == 'simple_MLP_embed_RTE':
            output1 = model(claims_var, evidence_var,len_claims_this_batch,len_evidences_this_batch)



        class_loss = class_criterion(output1, target_var) / minibatch_size
        LOG.debug(f"value of class_loss is:{class_loss}")

        #pred_labels=get_label_from_softmax(output1.data)

        LOG.debug(f"list of predictions are: of class_loss is:{output1.data}")
        LOG.debug(f"list of gold labels are:{target_var.data}")

            # measure accuracy and record loss
        prec1 = accuracy_fever(output1.data, target_var.data, LOG)

        LOG.debug(f"value of prec1 is :{prec1}")
        sum_all_acc=sum_all_acc+prec1


        meters.update('class_loss', class_loss.data.item(), labeled_minibatch_size)
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1, labeled_minibatch_size)
        # meters.update('top5', prec5[0], labeled_minibatch_size)
        # meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()


        LOG.info(
            'Epoch: [{0}][{1}/{2}]\t'
            'Prec_model: {meters[top1]:.3f}\t'
            'error_model:{meters[error1]:.3f}\t'
                .format(
                epoch, i, len(eval_loader), meters=meters))

        LOG.debug(f"value of  i after epoch {epoch} is :{i}")
        total_no_batches=i



    # LOG.info(' * Prec@1 {top1.avg:.3f}\tClassLoss {class_loss.avg:.3f}'
    #          .format(top1=meters['top1'], class_loss=meters['class_loss']))

    if save_custom_embed_condition:
        save_custom_embeddings(custom_embeddings_minibatch, dataset, result_dir, model_type)

    x=meters['top1'].avg
    LOG.debug(f"average precision after all the {total_no_batches} batches in epoch {epoch} is :{x}")
    LOG.debug(f"value of  total_no_batches  is :{total_no_batches}")
    LOG.debug(f"value of sum_all_acc after epoch {epoch} is :{sum_all_acc}")
    #todo: divide by total number of data points.-or keep track of how many true positives, divide by total data point count.
    x2=float(sum_all_acc)/float(total_no_batches)
    LOG.debug(f"value of average precision x2 after epoch {epoch} is :{x2}")


    return x2

#todo: do we need to save custom_embeddings?  - mihai
def save_custom_embeddings(custom_embeddings_minibatch, dataset, result_dir, model_type):

    start_time = time.time()
    mention_embeddings = dict()
    pattern_embeddings = dict()

    # dataset_id_list = list()

    for min_batch_id, datapoint in enumerate(custom_embeddings_minibatch):
        if torch.cuda.is_available():
            mention_embeddings_batch = datapoint[0].cuda().data.numpy()
            patterns_embeddings_batch = datapoint[1].permute(1, 0, 2).cuda().data.numpy()
        else:
            mention_embeddings_batch = datapoint[0].cpu().data.numpy()
            patterns_embeddings_batch = datapoint[1].permute(1, 0, 2).cpu().data.numpy()  # Note the permute .. to get the min-batches in the 1st dim
        # min_batch_sz = datapoint[2]

        # compute the custom entity embeddings
        for idx, embed in enumerate(mention_embeddings_batch):
            dataset_id = (min_batch_id * args.batch_size) + idx  # NOTE: `mini_batch_sz` here is a bug (in last batch)!! So changing to args.batch_size
            # dataset_id_list.append("ID="+str(min_batch_id)+"*"+str(min_batch_sz)+"+"+str(idx)+"="+str(dataset_id))
            mention_str = dataset.entity_vocab.get_word(dataset.mentions[dataset_id])
            if mention_str in mention_embeddings:
                prev_embed = mention_embeddings[mention_str]
                np.mean([prev_embed, embed], axis=0)
            else:
                mention_embeddings[mention_str] = embed

        # compute the custom pattern embeddings
        # print("==========================")
        # print("datapoint[0] (sz) = " + str(datapoint[0].size()))
        # print("datapoint[1] (sz) = " + str(datapoint[1].size()))
        # print("patterns_embeddings_batch (sz) = " + str(patterns_embeddings_batch.shape))
        # print("==========================")
        for idx, embed_arr in enumerate(patterns_embeddings_batch):
            dataset_id = (min_batch_id * args.batch_size) + idx  # NOTE: `mini_batch_sz` here is a bug (in last batch)!! So changing to args.batch_size
            patterns_arr = [dataset.context_vocab.get_word(ctxId) for ctxId in dataset.contexts[dataset_id]]
            num_patterns = len(patterns_arr)

            # print("-------------------------------------------------")
            # print("Patterns Arr : " + str(patterns_arr))
            # print("Num Patterns : " + str(num_patterns))
            # print("dataset_id : " + str(dataset_id))
            # print("embed_arr (sz) : " + str(embed_arr.shape))
            # print("-------------------------------------------------")
            for i in range(num_patterns):
                embed = embed_arr[i]
                pattern_str = patterns_arr[i]
                if pattern_str in pattern_embeddings:
                    prev_embed = pattern_embeddings[pattern_str]
                    np.mean([prev_embed, embed], axis=0)
                else:
                    pattern_embeddings[pattern_str] = embed

    entity_embed_file = os.path.join(result_dir, model_type + "_entity_embed.txt")
    with open(entity_embed_file, 'w') as ef:
        for string, embed in mention_embeddings.items():
            ef.write(string + "\t" + " ".join([str(i) for i in embed]) + "\n")
        # for string in dataset_id_list:
        #     ef.write(string + "\n")
    ef.close()

    pattern_embed_file = os.path.join(result_dir, model_type + "_pattern_embed.txt")
    with open(pattern_embed_file, 'w') as pf:
        for string, embed in pattern_embeddings.items():
            pf.write(string + "\t" + " ".join([str(i) for i in embed]) + "\n")
    pf.close()

    LOG.info("Saving the customs entity and pattern embeddings in dir :=> " + str(result_dir))
    LOG.info("Size of entity embeddings :=> " + str(len(mention_embeddings)))
    LOG.info("Size of pattern embeddings :=> " + str(len(pattern_embeddings)))
    LOG.info("COMPLETED writing the files in " + str(time.time() - start_time) + "s.")
    # LOG.info("Size of dataset_id : " + str(len(dataset_id_list)))
    # LOG.info("Size of dataset : " + str(len(dataset.mentions)))


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)   #new copy
        LOG.info("--- checkpoint copied to %s ---" % best_path)
        if args.epochs != epoch: # Note: Save the last checkpoint
            os.remove(checkpoint_path)
            LOG.info("--- removing original checkpoint %s ---" % checkpoint_path) # Note: I can as well not save the original file and only save the best config. But not changing the functionality too much, if need to revert later


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    #topk returns two values, first one is the actual list of top most values and second one is the indices of them
    _, pred = output.topk(maxk, 1, True, True)

    # transpose dimensions 0 and 1
    pred = pred.t()

    #if your max k is 1, target will almost always be the same size as pred (maybe after transfpose, but yes)
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    # target.size(): torch.Size([256])
    # target.view(1, -1): 1 * 256
    # expand_as(pred): copy the first row to be the second row to get 2*256
    # correct: 2*256

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res

#this accuracy is a simpler version where there are no topk elements. It just picks the element with highest value
def accuracy_fever(predicted_labels, gold_labels,LOG):
    m = nn.Softmax()
    output_sftmax = m(predicted_labels)
    LOG.debug(f"value of output_sftmax is :{output_sftmax}")

    labeled_minibatch_size = max(gold_labels.ne(NO_LABEL).sum(), 1e-8)

    LOG.debug(f"value of labeled_minibatch_size is :{labeled_minibatch_size}")

    #predictions, indices = torch.max(output_sftmax,0)
    _, pred = output_sftmax.topk(1, 1, True, True)
    LOG.info(f"value of pred is :{pred}")
    LOG.info(f"value of gold labels is is :{gold_labels}")


    #gold labels and predictions are in transposes (eg:1x15 vs 15x1). so take a transpose to correct it.
    pred_t=pred.t()


    #predicting everything as majority class: for debug purposes
    import itertools
    l1=list(itertools.repeat(2,labeled_minibatch_size))
    # check how many predictions you got right?
    l2=gold_labels.cpu().numpy().tolist()





    l2, correct = l2[:], [e for e in l1 if e in l2 and (l2.pop(l2.index(e)))]
    correct_k_float = float(sum(correct)/2)


    # correct = pred_t.eq(gold_labels.view(1, -1).expand_as(pred_t))
    # LOG.debug(f"value of correct is :{correct}")
    # #take sum because in correct_k all the LABELS that match are now denoted by 1. So the sum means, total number of correct answers
    # correct_k = correct.sum(1)
    # correct_k_float=float(correct_k.data.item())

    LOG.debug(f"value of correct_k as float is :{correct_k_float}")
    labeled_minibatch_size_f=float(labeled_minibatch_size)
    LOG.debug(f"value of labeled_minibatch_size is :{labeled_minibatch_size_f}")
    result2=(correct_k_float/labeled_minibatch_size_f)*100
    LOG.debug(f"value of result2 is :{result2}")
    #if out of 7 labeled, you got only 2 right, then your accuracy is 2/7*100

    #old-ajay code
    #result=correct_k.mul_(100.0 / labeled_minibatch_size)
    #LOG.debug(f"value of result is :{result}")

    return result2

def get_label_from_softmax(output):
    list_labels_pred=[]
    for tensor in output:
        values, indices = torch.max(tensor, 0)
        list_labels_pred.append(indices.data.item())
    return list_labels_pred

def prec_rec(output, target, NA_label, topk=(1,)):
    maxk = max(topk)
    assert maxk == 1, "Right now only computing P/R/F for topk=1"
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # size of targe and pred: 256 (it's the batch...)
    # some percentage of target have a label

    # number of instances WE predict that are not NA
    # includes the UNLABELED --> number of supervised(labeled) + unsupervised(labeled)

    tp_fn_1 = target.ne(NA_label)  # 1s where not NA
    tp_fn_2 = target.ne(NO_LABEL)  # 1s where labels not NONE
    tp_fn_idx = tp_fn_1.eq(
        tp_fn_2)  # 1s where target labels are not NA and not NONE; Note tp_fn_1 and tp_fn_2 do not equal to 0 at same index, tp_fn_1[idx] =0 means NA, tp_fn_2[idx] =0 means no label, they do not happen at the same time
    tp_fn = tp_fn_idx.sum()  # number of target labels which are not NA and not NONE (number of non NA labels in ONLY supervised portion of target)

    # index() takes same size of pred with idx value 0 and 1, and only return pred[idx] where idx is 1
    tp_fp = pred.index(tp_fn_2).ne(
        NA_label).sum()  # number of non NA labels in pred where target labels are not NONE  (Note: corresponded target labels can be NA)

    tp = pred.index(tp_fn_idx).eq(
        target.view(1, -1).index(tp_fn_idx)).sum()  # number of matches where target labels are not NA and not NONE

    return tp, tp_fn, tp_fp


def dump_result(batch_id, args, output, target, dataset, perm_idx, model_type='train_teacher', topk=(1,)):
    global test_student_pred_match_noNA
    global test_student_pred_noNA
    global test_student_true_noNA
    global test_teacher_pred_match_noNA
    global test_teacher_pred_noNA
    global test_teacher_true_noNA
    global train_student_pred_match_noNA
    global train_student_pred_noNA
    global train_student_true_noNA
    global train_teacher_pred_match_noNA
    global train_teacher_pred_noNA
    global train_teacher_true_noNA

    maxk = max(topk)
    assert maxk == 1, "Right now only computing for topk=1"
    score, prediction = output.topk(maxk, 1, True, True)

    dataset_config = datasets.__dict__[args.dataset]()
    evaldir = os.path.join(dataset_config['datadir'], args.eval_subdir)
    student_pred_file = evaldir + '/' + args.run_name + '_' + model_type + '_pred.tsv'
    teacher_pred_file = evaldir + '/' + args.run_name + '_' + model_type + '_pred.tsv'

    if torch.cuda.is_available():
        order_idx = perm_idx.cpu().numpy()
    else:
        order_idx = perm_idx.numpy()

    lbl_categories = dataset.categories

    if model_type == 'test_teacher':
        oov_label_lineid = dataset.oov_label_lineid
        dataset_file = evaldir + '/' + args.eval_subdir + '.txt'
        f = open(dataset_file)
        lines = []
        for line_id, line in enumerate(f):
            if line_id not in oov_label_lineid:
                lines.append(line)

        with open(teacher_pred_file, "a") as fo:
            for p, pre in enumerate(prediction):
                line_id = int(batch_id * args.batch_size + order_idx[p])
                line = lines[line_id].strip()
                lbl_id = int(pre)
                pred_label = lbl_categories[lbl_id].strip()
                if target[p] != NO_LABEL:
                    target_label = lbl_categories[target[p]].strip()
                else:
                    target_label = 'removed'

                vals = line.split('\t')
                true_label = vals[4].strip()
                match = pred_label == target_label

                if len(args.labels_set) == 0 or true_label in args.labels_set:
                    assert true_label == target_label

                if match and target_label != 'NA':
                    test_teacher_pred_match_noNA += 1.0
                if pred_label != 'NA':
                    test_teacher_pred_noNA += 1.0
                if target_label != 'NA':
                    test_teacher_true_noNA += 1.0

                line = line + '\t' + target_label + '\t' + pred_label + '\t' + str(match) + '\t' + str(float(score[p])) + '\n'
                fo.write(line)

    elif model_type == 'test_student':
        oov_label_lineid = dataset.oov_label_lineid
        dataset_file = evaldir + '/' + args.eval_subdir + '.txt'
        f = open(dataset_file)
        lines = []
        for line_id, line in enumerate(f):
            if line_id not in oov_label_lineid:
                lines.append(line)

        with open(student_pred_file, "a") as fo:
            for p, pre in enumerate(prediction):
                line_id = int(batch_id * args.batch_size + order_idx[p])
                line = lines[line_id].strip()
                lbl_id = int(pre)
                pred_label = lbl_categories[lbl_id].strip()
                if target[p] != NO_LABEL:
                    target_label = lbl_categories[target[p]].strip()
                else:
                    target_label = 'removed'

                vals = line.split('\t')
                true_label = vals[4].strip()
                match = pred_label == target_label

                if len(args.labels_set) == 0 or true_label in args.labels_set:
                    assert true_label == target_label

                if match and target_label != 'NA':
                    test_student_pred_match_noNA += 1.0
                if pred_label != 'NA':
                    test_student_pred_noNA += 1.0
                if target_label != 'NA':
                    test_student_true_noNA += 1.0

                line = line + '\t' + target_label + '\t' + pred_label + '\t' + str(match) + '\t' + str(float(score[p])) + '\n'
                fo.write(line)

    # cannot do the same way for train as test, becasue sentences with too many inbetween words were throw away, so it would not align with original train.txt
    elif model_type == 'train_teacher':

        with open(teacher_pred_file, "a") as fo:
            for p, pre in enumerate(prediction):
                lbl_id = int(pre)
                if lbl_id >= len(lbl_categories):
                    print('pred_label id'+ str(lbl_id))
                    print('number of labels: ' + str(len(lbl_categories)))
                pred_label = lbl_categories[lbl_id].strip()
                if target[p] != NO_LABEL:
                    target_label = lbl_categories[target[p]].strip()
                else:
                    target_label = 'removed'

                match = pred_label == target_label
                if match and target_label != 'NA' and target_label != 'removed':
                    train_teacher_pred_match_noNA += 1.0
                if pred_label != 'NA' and target_label != 'removed':
                    train_teacher_pred_noNA += 1.0
                if target_label != 'NA' and target_label != 'removed':
                    train_teacher_true_noNA += 1.0

                line = target_label + '\t' + pred_label + '\t' + str(match) + '\t' + str(float(score[p])) + '\n'
                fo.write(line)

    elif model_type == 'train_student':
        with open(student_pred_file, "a") as fo:
            for p, pre in enumerate(prediction):
                lbl_id = int(pre)
                if lbl_id >= len(lbl_categories):
                    print('pred_label id'+ str(lbl_id))
                    print('number of labels: ' + str(len(lbl_categories)))
                pred_label = lbl_categories[lbl_id].strip()
                if target[p] != NO_LABEL:
                    target_label = lbl_categories[target[p]].strip()
                else:
                    target_label = 'removed'

                match = pred_label == target_label
                if match and target_label != 'NA' and target_label != 'removed':
                    train_student_pred_match_noNA += 1.0
                if pred_label != 'NA' and target_label != 'removed':
                    train_student_pred_noNA += 1.0
                if target_label != 'NA' and target_label != 'removed':
                    train_student_true_noNA += 1.0

                line = target_label + '\t' + pred_label + '\t' + str(match) + '\t' + str(float(score[p])) + '\n'
                fo.write(line)



def main(context):
    global global_step
    global best_accuracy_across_epochs
    global best_epochs

    time_start = time.time()

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()

    num_classes=3
    if args.dataset in ['conll', 'ontonotes', 'riedel', 'gids','fever']:
        train_loader, eval_loader, dataset, dataset_test = create_data_loaders(LOG,**dataset_config, args=args)
        LOG.debug(f"after create_data_loaders. main.py line 1031. value of word_vocab.size()={len(dataset.word_vocab.keys())}")
        num_classes = len(dataset.categories)
        word_vocab_embed = dataset.word_vocab_embed
        LOG.debug(f"inside if arg.s dataset in fever value of word_vocab.size()={len(dataset.word_vocab.keys())}")
        word_vocab_size = len(dataset.word_vocab.keys())

    else:
        #mithun: i think this is the actual code from valpola that ran on cifar10 dataset
        train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)

    #uncomment this if you want to pop the number of classes instead from the config file
    # if args.dataset in ['riedel', 'gids','fever']:
    #
    # else:
    #     num_classes = dataset_config.pop('num_classes')

    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained '
            if args.pretrained else '',
            ema='EMA '
            if ema else '',arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)

        if args.dataset in ['conll', 'ontonotes', 'riedel', 'gids','fever']:

            #first two (word_vocab_embed,word_vocab_size) needs to be provided from command line
            model_params['word_vocab_embed'] = word_vocab_embed
            model_params['word_vocab_size'] = word_vocab_size
            model_params['wordemb_size'] = args.wordemb_size
            model_params['hidden_size'] = args.hidden_size
            model_params['update_pretrained_wordemb'] = args.update_pretrained_wordemb

        LOG.debug(f"value of word_vocab_embed={word_vocab_embed}")
        LOG.debug(f"value of word_vocab_size={word_vocab_size}")

        model = model_factory(**model_params)
        LOG.info("--------------------IMPORTANT: REMOVING nn.DataParallel for the moment --------------------")
        if torch.cuda.is_available():
            model = model.cuda()    # Note: Disabling data parallelism for now
            LOG.info(f"in line 1067 of main. found thatcUDA is available")

        else:
            model = model.cpu()

        #here if ema (e mean teacher)=True, they don't do back propagation. that is what param.detach does.
        if ema:
            for param in model.parameters():
                #LOG.debug("found that its ema . going to detach/no back prop")
                param.detach_() ##NOTE: Detaches the variable from the gradient computation, making it a leaf .. needed from EMA model

        return model

    #askfan: so the ema is teacher? and teacher is just a copy of student itself-but how/where do they do the moving average thing?
    model = create_model()
    ema_model = create_model(ema=True)

    LOG.info(parameters_string(model))

    evaldir = os.path.join(args.data_dir, args.eval_subdir)
    train_student_pred_file = evaldir  + args.run_name + '_train_student_pred.tsv'
    train_teacher_pred_file = evaldir  + args.run_name + '_train_teacher_pred.tsv'
    test_student_pred_file = evaldir  + args.run_name + '_test_student_pred.tsv'
    test_teacher_pred_file = evaldir  + args.run_name + '_test_teacher_pred.tsv'
    with contextlib.suppress(FileNotFoundError):
        os.remove(train_student_pred_file)
        os.remove(train_teacher_pred_file)
        os.remove(test_student_pred_file)
        os.remove(test_teacher_pred_file)


    #todo mithun: ask becky or fan if we need thi adam optimizer...also why are they using only when pretrained is false? damned tuning.
    if args.dataset in ['conll', 'ontonotes', 'riedel', 'gids'] and args.update_pretrained_wordemb is False:
        ## Note: removing the parameters of embeddings as they are not updated
        # https://discuss.pytorch.org/t/freeze-the-learnable-parameters-of-resnet-and-attach-it-to-a-new-network/949/9
        filtered_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(filtered_parameters)
        # optimizer = torch.optim.SGD(filtered_parameters, args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay,
        #                             nesterov=args.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_accuracy_across_epochs = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    #EVALUATE - ithink is for loading a trained model
    if args.evaluate:
        if args.dataset in ['conll', 'ontonotes','fever']:
            LOG.info("Evaluating the primary model:")
            validate(eval_loader, model, validation_log, global_step, args.start_epoch, dataset, context.result_dir, "student")
            LOG.info("Evaluating the EMA model:")
            validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch, dataset, context.result_dir, "teacher")
        elif args.dataset in ['riedel', 'gids']:
            LOG.info("Evaluating the primary model:")
            validate(eval_loader, model, validation_log, global_step, args.start_epoch, dataset_test, context.result_dir, "student")
            LOG.info("Evaluating the EMA model:")
            validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch, dataset_test, context.result_dir, "teacher")
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        #ask ajay: why are they not returning the trained models explicitly
        train(train_loader, model, ema_model, optimizer, epoch, dataset, training_log)
        LOG.info(f"--- done training epoch {epoch} in %s seconds ---" % (time.time() - start_time))

        LOG.debug(f"value of args.evaluation_epochs: {args.evaluation_epochs} ")
        LOG.debug(f"value of args.epoch: {epoch} ")



        #i don't completely understand what this  below modulo code is doing. THe official documentation of
        # "and" for integers in python says :
        # The expression x and y first evaluates x; if x is false, its value is returned;
        # otherwise, y is evaluated and the resulting value is returned.
        #i think you evaluate only in some epochs? why on earth would you do that?
        #update: this is the documentation from cli.py:evaluation frequency in epochs,
        # 0 to turn evaluation off (default: 1)').
        # ok, so probably default value 1 means it'll evaluate at every epoch, hopefully...
        #update commented it out. using simple modulo
        #if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:

        if (epoch) % args.evaluation_epochs == 0:
            LOG.debug("just got inside evaluation_epochs ")
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            LOG.debug(f"value of model: {model} ")
            LOG.debug(f"value of eval_loader: {eval_loader} ")
            LOG.debug(f"value of global_step: {global_step} ")
            LOG.debug(f"value of epoch: {validation_log} ")
            LOG.debug(f"value of dataset_test: {dataset_test} ")
            LOG.debug(f"value of context.result_dir: {context.result_dir} ")

            student_accuracy = validate(eval_loader, model, validation_log, global_step, epoch , dataset_test,
                             context.result_dir, "student")
            LOG.info("Evaluating the EMA model:")
            teacher_accuracy = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch , dataset_test,
                                 context.result_dir, "teacher")
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            local_best= max(teacher_accuracy, student_accuracy)
            is_best = teacher_accuracy > best_accuracy_across_epochs

            if(local_best>best_accuracy_across_epochs):
                best_accuracy_across_epochs = local_best
                best_epochs=epoch
            else:
                is_best = False

            LOG.info(f"best value of validation accuracy after epoch {epoch} is {local_best}")
            LOG.info(f"best value of best_accuracy_across_epochs so far is {best_accuracy_across_epochs} at epoch number {best_epochs}")

            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_accuracy_across_epochs,
                    'optimizer' : optimizer.state_dict(),
                    'dataset' : args.dataset,
                }, is_best, checkpoint_path, epoch + 1)




    # for testing only .. commented
    # LOG.info("For testing only; Comment the following line of code--------------------------------")
    # validate(eval_loader, model, validation_log, global_step, 0, dataset, context.result_dir, "student")
    LOG.info("--------Total end to end time %s seconds ----------- " % (time.time() - time_start))
    LOG.info(f"best best_accuracy_across_epochs  is:{best_accuracy_across_epochs} at epoch number:{best_epochs}")
    import sys
    sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    random_seed = args.random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        # torch.cuda.manual_seed_all(args.random_seed)
    else:
        torch.manual_seed(args.random_seed)

    print('----------------')
    print("Running mean teacher experiment with args:")
    print('----------------')
    print(args)
    print('----------------')
    main(RunContext(__file__, 0, args.run_name))
