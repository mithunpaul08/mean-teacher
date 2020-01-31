from comet_ml import Experiment,ExistingExperiment
import torch

from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.train_rao import Trainer
from mean_teacher.scripts.initializer import Initializer
from mean_teacher.utils.utils_rao import make_embedding_matrix,create_model,set_seed_everywhere
from mean_teacher.utils.logger import LOG
from mean_teacher.model import architectures
import os
import logging
import time
import random
import numpy as np
import sys


current_time={time.strftime("%c")}
LOG.info(f"starting the run at {current_time}.")



def initialize_comet(args):
    # for drawing graphs on comet:
    comet_value_updater=None
    if(args.run_type=="train"):
        if(args.create_new_comet_graph==True):
            comet_value_updater = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", project_name="rte-decomp-attention")
        else:
            comet_value_updater = ExistingExperiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", previous_experiment="74cf9e3531814abcb8733a5973f3413a")

    return comet_value_updater

initializer=Initializer()
initializer.set_default_parameters()
args = initializer.parse_commandline_args()




comet_value_updater=initialize_comet(args)
import torch

if (comet_value_updater) is not None:
    hyper_params = vars(args)
    comet_value_updater.log_parameters(hyper_params)




LOG.setLevel(args.log_level)

if args.run_type=="test":
    args.load_vectorizer=True
    args.load_model_from_disk=True



LOG.info(f"setting the manual seed as {args.random_seed} ")
LOG.setLevel(args.log_level)

current_time={time.strftime("%c")}

glove_filepath_in, lex_train_input_file, lex_dev_input_file, lex_test_input_file , delex_train_input_file, delex_dev_input_file, delex_test_input_file \
    =initializer.get_file_paths()
LOG.info(f"{current_time} loading glove from path:{glove_filepath_in}")


if args.reload_from_files:
    # training from a checkpoint
    dataset = RTEDataset.load_dataset_and_load_vectorizer(args.fever_lex_train,
                                                              args.vectorizer_file)
else:
    # create dataset and vectorizer
    dataset = RTEDataset.load_dataset_and_create_vocabulary_for_combined_lex_delex(lex_train_input_file, lex_dev_input_file, delex_train_input_file, delex_dev_input_file, delex_test_input_file, args)
    dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()

# taking embedding size from user initially, but will get replaced by original embedding size if its loaded
embedding_size=args.embedding_size

# Use GloVe or randomly initialized embeddings
if args.use_glove:
    words = vectorizer.claim_ev_vocab._token_to_idx.keys()
    embeddings,embedding_size = make_embedding_matrix(glove_filepath_in,words)
    LOG.info(f"{current_time:} Using pre-trained embeddings")
else:
    LOG.info(f"{current_time:} Not using pre-trained embeddings")
    embeddings = None

num_features=len(vectorizer.claim_ev_vocab)
classifier_teacher_lex=None
if(args.use_ema):
    classifier_teacher_lex = create_model(logger_object=LOG, args_in=args, num_classes_in=len(vectorizer.label_vocab)
                                      , word_vocab_embed=embeddings, word_vocab_size=num_features, wordemb_size_in=embedding_size,ema=True)
else:
    classifier_teacher_lex = create_model(logger_object=LOG, args_in=args, num_classes_in=len(vectorizer.label_vocab)
                                          , word_vocab_embed=embeddings, word_vocab_size=num_features,
                                          wordemb_size_in=embedding_size)

assert classifier_teacher_lex is not None
classifier_student_delex = create_model(logger_object=LOG, args_in=args, num_classes_in=len(vectorizer.label_vocab)
                                        , word_vocab_embed=embeddings, word_vocab_size=num_features, wordemb_size_in=embedding_size)

train_rte=Trainer(LOG)
if(args.load_model_from_disk_and_test):
    LOG.info(f"{current_time:} Found that need to load model and test using it.")
    train_rte.test(args,classifier_student_delex, dataset, "val")
    sys.exit(1)
train_rte.train(args, classifier_teacher_lex, classifier_student_delex, dataset, comet_value_updater, vectorizer)
