from comet_ml import Experiment

from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.train_rao import Trainer
from mean_teacher.scripts.initializer import Initializer
from mean_teacher.utils.utils_rao import make_embedding_matrix,create_model,set_seed_everywhere

from mean_teacher.utils.logger import LOG
import time
import random
import numpy as np



# Create an comet_value_updater value for comet.ml
comet_value_updater = Experiment(api_key="XUbi4cShweB6drrJ5eAKMT6FT",
                            project_name="rte-decomp-attention", workspace="mithunpaul08",auto_output_logging="simple")
import torch


initializer=Initializer()
command_line_args = initializer.parse_commandline_args()
args=initializer.set_parameters()

# for drawing graphs on comet:
hyper_params=vars(args)
comet_value_updater.log_parameters(hyper_params)




set_seed_everywhere(args.seed, args.cuda)

random_seed = args.random_seed
random.seed(random_seed)
np.random.seed(random_seed)
LOG.setLevel(args.log_level)


torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    LOG.info(f"found that cuda is available. ALso setting the manual seed as {args.random_seed} ")
else:
    torch.manual_seed(args.random_seed)
    LOG.info(f"found that cuda is not available . ALso setting the manual seed as {args.random_seed} ")


current_time={time.strftime("%c")}

glove_filepath_in,fever_train_input_file,fever_dev_input_file=initializer.get_file_paths(command_line_args)
LOG.info(f"{current_time} loading glove from path:{glove_filepath_in}")
LOG.debug(f"value of fever_train_input_file is :{fever_train_input_file}")
LOG.debug(f"value of fever_dev_input_file is :{fever_dev_input_file}")


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

num_features=len(vectorizer.claim_ev_vocab)
classifier = create_model(logger_object=LOG,args_in=args,num_classes_in=len(vectorizer.label_vocab)
                          ,word_vocab_embed=embeddings,word_vocab_size=num_features,wordemb_size_in=embedding_size)

train_rte=Trainer()
train_rte.train(args,classifier,dataset,comet_value_updater)
