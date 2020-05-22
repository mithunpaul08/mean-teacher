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
            comet_Expt_object = ExistingExperiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT", previous_experiment="8ee6669d2b854eaf834f8a56eaa9f235")

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


LOG.info(f"{current_time:}Going to read data")

avail=False
if torch.cuda.is_available():
    avail=True
LOG.info(f"cuda available:{avail}")

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = 'bert-base-uncased'
# Read the dataset
batch_size = 20
abs=os.path.abspath(os.path.dirname(__file__))
os.chdir(abs)
nli_reader_fever_lex = NLIDataReader('data/rte/fever/allnli/lex/')
nli_reader_fever_delex = NLIDataReader('data/rte/fever/allnli/delex/')
nli_reader_fnc_lex = NLIDataReader('data/rte/fnc/allnli/lex')
nli_reader_fnc_delex = NLIDataReader('data/rte/fnc/allnli/delex')
train_num_labels = nli_reader_fever_lex.get_num_labels()
model_save_path = 'output/training_nli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



classifier_teacher_lex = create_model_bert()
classifier_student_delex = create_model_bert()




assert classifier_teacher_lex is not None
assert classifier_student_delex is not None


# Convert the dataset to a DataLoader ready for training
logging.info("Reading fever train dataset")

#all training related data
train_data_lex = SentencesDataset(nli_reader_fever_lex.get_examples('train.gz'), model=classifier_teacher_lex)
train_dataloader_lex = DataLoader(train_data_lex, shuffle=True, batch_size=batch_size)
train_loss_lex = losses.SoftmaxLoss(model=classifier_teacher_lex, sentence_embedding_dimension=classifier_teacher_lex.get_sentence_embedding_dimension(), num_labels=train_num_labels)

train_data_delex = SentencesDataset(nli_reader_fever_delex.get_examples('train.gz'), model=classifier_student_delex)
train_dataloader_delex = DataLoader(train_data_delex, shuffle=True, batch_size=batch_size)
train_loss_delex = losses.SoftmaxLoss(model=classifier_student_delex, sentence_embedding_dimension=classifier_student_delex.get_sentence_embedding_dimension(), num_labels=train_num_labels)

#all fever dev related data
logging.info("Reading fever dev dataset")
dev_data_fever_lex = SentencesDataset(nli_reader_fever_lex.get_examples('dev.gz'), model=classifier_teacher_lex)
dev_dataloader_fever_lex = DataLoader(dev_data_fever_lex, shuffle=False, batch_size=batch_size)
evaluator_fever_dev_lex = LabelAccuracyEvaluator(dev_dataloader_fever_lex, softmax_model = train_loss_lex, grapher=comet_value_updater, logger=LOG, name="fever-dev-lex")

dev_data_fever_delex = SentencesDataset(nli_reader_fever_delex.get_examples('dev.gz'), model=classifier_student_delex)
dev_dataloader_fever_delex = DataLoader(dev_data_fever_delex, shuffle=False, batch_size=batch_size)
evaluator_fever_dev_delex = LabelAccuracyEvaluator(dev_dataloader_fever_delex, softmax_model = train_loss_delex, grapher=comet_value_updater, logger=LOG, name="fever-dev-delex")

#all fnc dev related data
dev_data_fnc_lex = SentencesDataset(nli_reader_fnc_lex.get_examples('dev.gz'), model=classifier_teacher_lex)
dev_dataloader_fnc_lex = DataLoader(dev_data_fnc_lex, shuffle=False, batch_size=batch_size)
evaluator_fnc_lex = LabelAccuracyEvaluator(dev_dataloader_fnc_lex, softmax_model = train_loss_lex, grapher=comet_value_updater, logger=LOG, name="fnc-dev-lex")

dev_data_fnc_delex = SentencesDataset(nli_reader_fnc_delex.get_examples('dev.gz'), model=classifier_student_delex)
dev_dataloader_fnc_delex = DataLoader(dev_data_fnc_delex, shuffle=False, batch_size=batch_size)
evaluator_fnc_delex = LabelAccuracyEvaluator(dev_dataloader_fnc_delex, softmax_model = train_loss_delex, grapher=comet_value_updater, logger=LOG, name="fnc-dev-delex")


if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Configure the training
num_epochs = 25

warmup_steps = math.ceil(len(train_dataloader_lex) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))




classifier_teacher_lex.train_1teacher(args, train_objectives=[(train_dataloader_lex, train_loss_lex),(train_dataloader_delex, train_loss_delex)],
                                      evaluators = [evaluator_fever_dev_lex,evaluator_fever_dev_delex, evaluator_fnc_lex, evaluator_fnc_delex],
                                      epochs = num_epochs,
                                      evaluation_steps = 1,
                                      warmup_steps = warmup_steps,
                                      output_path = model_save_path
                                      )


