from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.classifier_decomp_attn_works_with_rao_code import DecompAttnClassifier
from mean_teacher.model.train_rao import Trainer
from mean_teacher.scripts.initializer import Initializer
from mean_teacher.utils.utils_rao import make_embedding_matrix
from mean_teacher.utils.logger import Logger
import os
import logging
import time



initializer=Initializer()
command_line_args = initializer.parse_commandline_args()
args=initializer.set_parameters()

obj_logger=Logger(args)
LOG=obj_logger.get_logger()

current_time={time.strftime("%c")}

glove_filepath_in,fever_train_input_file,fever_dev_input_file=initializer.get_file_paths(command_line_args)
LOG.info(f"{current_time} loading glove from path:{glove_filepath_in}")


if args.reload_from_files:
    # training from a checkpoint
    dataset = RTEDataset.load_dataset_and_load_vectorizer(args.fever_lex_train,
                                                              args.vectorizer_file)
else:
    # create dataset and vectorizer
    dataset = RTEDataset.load_dataset_and_create_vocabulary(fever_train_input_file, fever_dev_input_file,args)
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

classifier = DecompAttnClassifier(len(vectorizer.claim_ev_vocab),embedding_size,args.hidden_sz, embeddings,
                  args.update_pretrained_wordemb, args.para_init, len(vectorizer.label_vocab), args.use_gpu)

train_rte=Trainer(LOG)
train_rte.train(args,classifier,dataset)
