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
import torch
import numpy as np


initializer=Initializer()
command_line_args = initializer.parse_commandline_args()
args=initializer.set_parameters()

set_seed_everywhere(args.seed, args.cuda)

random_seed = args.random_seed
random.seed(random_seed)
np.random.seed(random_seed)
LOG.setLevel(args.log_level)


torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    LOG.info(f"found that cuda is available and hence setting the manual seed as {args.random_seed} ")
else:
    torch.manual_seed(args.random_seed)
    LOG.info(f"found that cuda is not available and hence setting the manual seed as {args.random_seed} ")

LOG.setLevel(args.log_level)

current_time={time.strftime("%c")}

glove_filepath_in,fever_train_input_file,fever_dev_input_file,fever_delex_train_input_file,fever_delex_dev_input_file=initializer.get_file_paths(command_line_args)
LOG.info(f"{current_time} loading glove from path:{glove_filepath_in}")


if args.reload_from_files:
    # training from a checkpoint
    dataset = RTEDataset.load_dataset_and_load_vectorizer(args.fever_lex_train,
                                                              args.vectorizer_file)
else:
    # create dataset and vectorizer
    dataset = RTEDataset.load_dataset_and_create_vocabulary_for_combined_lex_delex(fever_train_input_file, fever_dev_input_file, fever_delex_train_input_file,fever_delex_dev_input_file,args)
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
classifier = create_model(logger_object=LOG,args_in=args,num_classes_in=len(vectorizer.label_vocab)
                          ,word_vocab_embed=embeddings,word_vocab_size=num_features,wordemb_size_in=embedding_size)

train_rte=Trainer(LOG)
train_rte.train(args,classifier,dataset)
