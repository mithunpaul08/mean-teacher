from comet_ml import Experiment,ExistingExperiment
import torch

from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.train_rao import Trainer
from mean_teacher.scripts.initializer import Initializer
from mean_teacher.utils.utils_rao import make_embedding_matrix,create_model,set_seed_everywhere,create_empty_json_file
from mean_teacher.utils.logger import Logger
from mean_teacher.model import architectures

import os
import logging
import time
import random
import numpy as np
import sys
import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha




start=time.time()
def initialize_comet(args):
    # for drawing graphs on comet:
    comet_value_updater=None
    if(args.run_type=="train"):
        if(args.create_new_comet_graph==True):
            comet_value_updater = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", project_name="rte-decomp-attention")
        else:
            comet_value_updater = ExistingExperiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", previous_experiment="8ee6669d2b854eaf834f8a56eaa9f235")

    return comet_value_updater

initializer=Initializer()
initializer.set_default_parameters()
args = initializer.parse_commandline_args()

if(args.load_model_from_disk_and_test):
    args.lex_test='fnc/test/fnc_test_lex.jsonl'
    args.delex_test='fnc/test/fnc_test_delex.jsonl'

current_time={time.strftime("%c")}
logger_client=Logger()
LOG=logger_client.initialize_logger(args)

LOG.info(f"starting the run at {current_time}.")



comet_value_updater=initialize_comet(args)
import torch

if (comet_value_updater) is not None:
    hyper_params = vars(args)
    comet_value_updater.log_parameters(hyper_params)






if args.run_type=="test":
    args.load_vectorizer=True
    args.load_model_from_disk=True



LOG.info(f"setting the manual seed as {args.random_seed} ")
LOG.setLevel(args.log_level)

current_time={time.strftime("%c")}

glove_filepath_in, lex_train_input_file, lex_dev_input_file, lex_test_input_file , delex_train_input_file, delex_dev_input_file, delex_test_input_file \
    =initializer.get_file_paths(LOG)


LOG.info(f"{current_time:}Going to read data")




if args.reload_data_from_files:
    # training from a checkpoint
    dataset = RTEDataset.load_dataset_and_load_vectorizer(args.fever_lex_train,
                                                              args.vectorizer_file)
else:
    # create dataset and vectorizer
    dataset = RTEDataset.load_dataset_and_create_vocabulary_for_combined_lex_delex(lex_train_input_file, lex_dev_input_file, delex_train_input_file, delex_dev_input_file, delex_test_input_file, lex_test_input_file,args)
    vectorizer_name=os.path.join(args.save_dir,"vectorizer_"+sha+".json")
    dataset.save_vectorizer(vectorizer_name)
vectorizer = dataset.get_vectorizer()

# taking embedding size from user initially, but will get replaced by original embedding size if its loaded
embedding_size=args.embedding_size

# Use GloVe or randomly initialized embeddings
LOG.info(f"{current_time} going to load glove from path:{glove_filepath_in}")
if args.use_glove:
    words = vectorizer.claim_ev_vocab._token_to_idx.keys()
    embeddings,embedding_size = make_embedding_matrix(glove_filepath_in,words)
    LOG.info(f"{current_time:} Using pre-trained embeddings")
else:
    LOG.info(f"{current_time:} Not using pre-trained embeddings")
    embeddings = None

num_features=len(vectorizer.claim_ev_vocab)


def run_training_eval():
    classifier_teacher_lex=None


    # and then try training student.
    # trial on march 2020 to 1) train a teacher model offline then 2) load that trained model as teacher (which doesn't have a backprop)
    if(args.use_trained_teacher_inside_student_teacher_arch):
        # when you are using a trained model, you should really be using the same vectorizer. Else embedding mismatch will happen
        vectorizer_loaded= RTEDataset.load_vectorizer_only(args.vectorizer_file)
        LOG.info(f"num_classes_in={len(vectorizer_loaded.label_vocab)}")
        LOG.info(f"word_vocab_size={len(vectorizer_loaded.claim_ev_vocab)}")



        words = vectorizer_loaded.claim_ev_vocab._token_to_idx.keys()
        labels = vectorizer_loaded.label_vocab._token_to_idx.keys()
        embeddings_loaded, embedding_size_loaded = make_embedding_matrix(glove_filepath_in, words)

        LOG.info(f"wordemb_size_in={(embedding_size_loaded)}")
        LOG.info(f"len word_vocab_embed={len(embeddings_loaded)}")
        classifier_teacher_lex = create_model(logger_object=LOG, args_in=args,
                                              num_classes_in=len(vectorizer_loaded.label_vocab)
                                              , word_vocab_embed=embeddings_loaded,
                                              word_vocab_size=len(vectorizer_loaded.claim_ev_vocab),
                                              wordemb_size_in=embedding_size_loaded)

        assert os.path.exists(args.trained_model_path) is True
        assert os.path.isfile(args.trained_model_path) is True
        if os.path.getsize(args.trained_model_path) > 0:
            classifier_teacher_lex.load_state_dict(
                torch.load(args.trained_model_path, map_location=torch.device(args.device)))
    else:
        #when the teacher is used in ema mode, no backpropagation will occur in teacher.
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

    #load a model that was trained on in-domain fever to test on fnc-test partition. this should be ideally done only once
    # since we are looking at the test-partition.
    if(args.load_model_from_disk_and_test):
        LOG.info(f"{current_time:} Found that need to load model and test using it.")
        partition_to_evaluate_on="test_delex"
        #if you are loading a teacher model trained on lexicalized data, evaluate on the lexical version of fnc-test
        if(args.type_of_trained_model=="teacher"):
            partition_to_evaluate_on = "test_lex"
        train_rte.load_model_and_eval(args,classifier_student_delex, dataset, partition_to_evaluate_on,vectorizer)
        end = time.time()
        LOG.info(f"time taken= {end-start}seconds.")
        sys.exit(1)
    args.validation_batch_10fcv=cvbatch
    train_rte.train(args, classifier_teacher_lex, classifier_student_delex, dataset, comet_value_updater, vectorizer)




#cvbatch is a temporary hack for trying out 10 fold cross validation for training teacher on april 2020
create_empty_json_file(args.predictions_teacher_file)
for cvbatch in range(10):
    print(f"****************starting 10fcv with dev batch as batch:{cvbatch}*********************")
    LOG.info(f"****************starting 10fcv with dev batch as batch:{cvbatch}*********************")
    run_training_eval()

end = time.time()
print(f"end of entire run")
LOG.info(f"end of entire run")
LOG.info(f"time taken= {end-start}seconds.")
