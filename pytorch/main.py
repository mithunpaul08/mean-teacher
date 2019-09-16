from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.classifier_decomp_attn_works_with_rao_code import DecompAttnClassifier
from mean_teacher.model.train_rao import Trainer
from mean_teacher.scripts.set_parameters import Initializer
from mean_teacher.utils.utils_rao import make_embedding_matrix
import os

initializer=Initializer()
command_line_args = initializer.parse_commandline_args()
args=initializer.set_parameters()
args.use_glove = True

data_dir=args.data_dir
glove_filepath_in = args.glove_filepath_local
fever_train_input_file = os.path.join(data_dir, args.fever_train_local)
fever_dev_input_file = os.path.join(data_dir, args.fever_dev_local)

if(command_line_args.run_on_server==True):
    glove_filepath_in = args.glove_filepath_server ,
    fever_train_input_file = os.path.join(data_dir, args.fever_train_server)
    fever_dev_input_file = os.path.join(data_dir, args.fever_dev_server)

print(fever_train_input_file)
print(fever_dev_input_file)



if args.reload_from_files:
    # training from a checkpoint
    dataset = RTEDataset.load_dataset_and_load_vectorizer(args.fever_lex_train,
                                                              args.vectorizer_file)
else:
    # create dataset and vectorizer
    dataset = RTEDataset.load_dataset_and_make_vectorizer(fever_train_input_file,fever_dev_input_file)
    dataset.save_vectorizer(args.vectorizer_file)
vectorizer = dataset.get_vectorizer()

# taking embedding size from user initially, but will get replaced by original embedding size if its loaded
embedding_size=args.embedding_size
# Use GloVe or randomly initialized embeddings
if args.use_glove:
    words = vectorizer.claim_ev_vocab._token_to_idx.keys()
    embeddings,embedding_size = make_embedding_matrix(glove_filepath=glove_filepath_in,
                                       words=words)
    print("Using pre-trained embeddings")
else:
    print("Not using pre-trained embeddings")
    embeddings = None

classifier = DecompAttnClassifier(len(vectorizer.claim_ev_vocab),embedding_size,args.hidden_sz, embeddings,
                  args.update_pretrained_wordemb, args.para_init, len(vectorizer.label_vocab), args.use_gpu)

train_rte=Trainer()
train_rte.train(args,classifier,dataset)
