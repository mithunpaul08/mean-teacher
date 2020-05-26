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
import math


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
            comet_Expt_object = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", project_name="rte-decomp-attention")
        else:
            comet_Expt_object = ExistingExperiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", previous_experiment="0c85d44875234ec1a7d0a45bdedb551b")

    return comet_Expt_object

initializer=Initializer()
initializer.set_default_parameters()
args = initializer.parse_commandline_args()
args=initializer.set_default_parameters2(args)




current_time={time.strftime("%c")}
logger_client=Logger()
LOG=logger_client.initialize_logger()

LOG.info(f"starting the run at {current_time}.")


comet_value_updater=initialize_comet(args)
import torch

if (comet_value_updater) is not None:
    hyper_params = vars(args)
    comet_value_updater.log_parameters(hyper_params)



LOG.info(f"setting the manual seed as {args.random_seed} ")
LOG.setLevel(args.log_level)


# glove_filepath_in, lex_train_input_file, lex_dev_input_file, lex_test_input_file , delex_train_input_file, \
# delex_dev_input_file, delex_test_input_file,gigaword_full_path=initializer.get_file_paths(LOG)


LOG.info(f"{current_time:}Going to read data")

avail=False
if torch.cuda.is_available():
    avail=True
LOG.info(f"cuda available:{avail}")

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-uncased'
# Read the dataset
batch_size = 16
abs=os.path.abspath(os.path.dirname(__file__))
os.chdir(abs)
nli_reader_fever_lex = NLIDataReader('data/rte/fever/allnli/lex/')
nli_reader_fnc_lex = NLIDataReader('data/rte/fnc/allnli/lex/')

# nli_reader_fever_delex = NLIDataReader('data/rte/fever/allnli/delex/')
# nli_reader_fnc_delex = NLIDataReader('data/rte/fnc/allnli/delex')

train_num_labels = nli_reader_fever_lex.get_num_labels()
model_save_path = 'output/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



classifier_teacher_lex = create_model_bert()



#assert classifier_student_delex_ema is not None
assert classifier_teacher_lex is not None
#assert classifier_student_delex is not None


# Convert the dataset to a DataLoader ready for training
logging.info("Reading fever train dataset")

train_data = SentencesDataset(nli_reader_fever_lex.get_examples('train.gz'), model=classifier_teacher_lex)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.SoftmaxLoss(model=classifier_teacher_lex, sentence_embedding_dimension=classifier_teacher_lex.get_sentence_embedding_dimension(), num_labels=train_num_labels)


logging.info("Reading fever dev dataset")
dev_data = SentencesDataset(nli_reader_fever_lex.get_examples('dev.gz'), model=classifier_teacher_lex)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator_fever = LabelAccuracyEvaluator(dev_dataloader,softmax_model = train_loss,grapher=comet_value_updater,logger=LOG,name="fever-dev")

dev_data = SentencesDataset(nli_reader_fnc_lex.get_examples('dev.gz'), model=classifier_teacher_lex)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator_fnc = LabelAccuracyEvaluator(dev_dataloader,softmax_model = train_loss,grapher=comet_value_updater,logger=LOG,name="fnc-dev")


if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Configure the training
num_epochs = 25

warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))




classifier_teacher_lex.train_1teacher(args,train_objectives=[(train_dataloader, train_loss)],
                                      evaluators = [evaluator_fever,evaluator_fnc],
                                      epochs = num_epochs,
                                      evaluation_steps = 1,
                                      warmup_steps = warmup_steps,
                                      output_path = model_save_path,
                                      grapher=comet_value_updater,
                                    optimizer_params= {'lr': args.learning_rate,'weight_decay':0.01, 'eps': 1e-6,
                                                        'correct_bias': False},
                                      )


