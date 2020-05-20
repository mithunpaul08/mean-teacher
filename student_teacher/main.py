from comet_ml import Experiment,ExistingExperiment
import torch
from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.utils.utils_rao import make_embedding_matrix,create_model,set_seed_everywhere,read_gigaword_freq_file,create_model_bert
from mean_teacher.model import architectures
from mean_teacher.model.train_rao import Trainer
from mean_teacher.scripts.initializer import Initializer
from mean_teacher.modules import vectorizer_with_embedding
from mean_teacher.utils.logger import Logger
import os
import logging
import time
import random
import numpy as np
import sys
import git


from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import sys
from sentence_transformers.readers import NLIDataReader
import os

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha




start=time.time()
def initialize_comet(args):
    # for drawing graphs on comet:
    comet_Expt_object=None
    if(args.run_type=="train"):
        if(args.create_new_comet_graph==True):
            comet_Expt_object = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", project_name="rte-decomp-attention",disabled=True)
        else:
            comet_Expt_object = ExistingExperiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", previous_experiment="8ee6669d2b854eaf834f8a56eaa9f235",disabled=True)

    return comet_Expt_object

initializer=Initializer()
initializer.set_default_parameters()
args = initializer.parse_commandline_args()
args=initializer.set_default_parameters2(args)



if(args.load_model_from_disk_and_test):
    args.lex_test='fnc/test/fnc_test_lex.jsonl'
    args.delex_test='fnc/test/fnc_test_delex.jsonl'

current_time={time.strftime("%c")}
logger_client=Logger()
LOG=logger_client.initialize_logger()

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

glove_filepath_in, lex_train_input_file, lex_dev_input_file, lex_test_input_file , delex_train_input_file, \
delex_dev_input_file, delex_test_input_file,gigaword_full_path=initializer.get_file_paths(LOG)


LOG.info(f"{current_time:}Going to read data")

avail=False
if torch.cuda.is_available():
    avail=True
LOG.info(f"cuda available:{avail}")

#create a vocabulary which is a union of training data and  top n freq words from gigaword. as per mihai this enhances/is usefull to reduce
#the number of uNK words in dev/test partitions- especially when either of those tend to be cross-domain. though i still think its cheating- mithun
#steps:
# - define min frequency
# - load list of words from gigaword - only words which have a minimum frequencey
# - get list of all words in training.
# - take each of training word and convert to lowercase
# - merge them- set{trainwords,gigawords}
# - for each word load its glove embedding
#
vectorizer_with_embedding.gigaword_freq=read_gigaword_freq_file(gigaword_full_path,args.gw_minfreq)


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

    # Create a vocabulary dictionary for the cross domain dataset and load its embeddings also in memory.
    # Reason: it was noticed that when we were testing on the cross domain dataset, there were way too many
    # unknown <UNK> tokens. So now, if a word is new in the cross domain, don't call it UNK just because it was not
    # there in in-domain vocabulary.  check if it exists in crossdomain vocab. if yes, load its glove
    # make_embedding_matrix()

    LOG.info(f"{current_time:} Using pre-trained embeddings")
else:
    LOG.info(f"{current_time:} Not using pre-trained embeddings")
    embeddings = None

num_features=len(vectorizer.claim_ev_vocab)


#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-uncased'
# Read the dataset
batch_size = 32
abs=os.path.abspath(os.path.dirname(__file__))
os.chdir(abs)
nli_reader_fever = NLIDataReader('data/rte/fever/allnli')
nli_reader_fnc = NLIDataReader('data/rte/fnc/allnli')
train_num_labels = nli_reader_fever.get_num_labels()
model_save_path = 'output/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


classifier_teacher_lex=None
if(args.use_ema):
    classifier_teacher_lex = create_model_bert(logger_object=LOG, args_in=args, num_classes_in=len(vectorizer.label_vocab)
                                      , word_vocab_embed=embeddings, word_vocab_size=num_features, wordemb_size_in=embedding_size,ema=True)
else:

    classifier_teacher_lex = create_model_bert()


# classifier_student_delex_ema = create_model(logger_object=LOG, args_in=args, num_classes_in=len(vectorizer.label_vocab)
#                                             , word_vocab_embed=embeddings, word_vocab_size=num_features,
#                                             wordemb_size_in=embedding_size, ema=True)
#
# classifier_teacher_lex_ema = create_model(logger_object=LOG, args_in=args, num_classes_in=len(vectorizer.label_vocab)
#                                             , word_vocab_embed=embeddings, word_vocab_size=num_features,
#                                             wordemb_size_in=embedding_size, ema=True)


# classifier_student_delex = create_model_bert(logger_object=LOG, args_in=args, num_classes_in=len(vectorizer.label_vocab)
#                                         , word_vocab_embed=embeddings, word_vocab_size=num_features, wordemb_size_in=embedding_size)



#assert classifier_student_delex_ema is not None
assert classifier_teacher_lex is not None
#assert classifier_student_delex is not None


# Convert the dataset to a DataLoader ready for training
logging.error("Read fever train dataset")

train_data = SentencesDataset(nli_reader_fever.get_examples('train.gz'), model=classifier_teacher_lex)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=classifier_teacher_lex, sentence_embedding_dimension=classifier_teacher_lex.get_sentence_embedding_dimension(), num_labels=train_num_labels)


#load a model that was trained on in-domain fever to test on fnc-test partition. this should be ideally done only once
# since we are looking at the test-partition.
if(args.load_model_from_disk_and_test):
    LOG.info(f"{current_time:} Found that need to load model and test using it.")
    #train_rte.load_model_and_eval(args,classifier_student_delex, dataset, "test_delex",vectorizer)

else:
    #this is plain training and will do eval on dev and test at the end of training.
    #1 teacher-1 student
    classifier_teacher_lex.train_1teacher(args, dataset, comet_value_updater, vectorizer,train_objectives=[(train_dataloader, train_loss)])
    #3teacher -1 student
    #train_rte.train(args, classifier_student_delex_ema, classifier_teacher_lex_ema, classifier_teacher_lex,
     #               classifier_student_delex, dataset, comet_value_updater, vectorizer)





def load_vectorizer_and_model():
    '''
    This entire function is vestigial. Just keeping it around so as to resuse the vectorizer loading and other code in it
    :return:
    '''
    # trial on march 2020 to 1) train a teacher model offline then 2) load that trained model as teacher (which doesn't have a backprop)
    # and then try training student.
    if (args.use_trained_teacher_inside_student_teacher_arch):
        # when you are using a trained model, you should really be using the same vectorizer. Else embedding mismatch will happen
        vectorizer_loaded = RTEDataset.load_vectorizer_only(args.vectorizer_file)

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

