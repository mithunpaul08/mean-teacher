from argparse import Namespace
import torch
import os
from mean_teacher.utils.utils_rao import set_seed_everywhere,make_embedding_matrix
from mean_teacher.utils.utils_rao import handle_dirs
from mean_teacher.modules.rao_datasets import RTEDataset

class Initializer():
    def set_parameters(self):
        args = Namespace(
            # Data and Path information
            frequency_cutoff=25,
            model_state_file='model.pth',
            fever_lex_train='train/fever_train_lex_3labels_11k_smartner_3labels_no_lists_evidence_not_sents.jsonl',
            fever_lex_dev='dev/fever_dev_lex_3labels_2k_no_lists_evidence_not_sents.jsonl',
            save_dir='model_storage/ch3/yelp/',
            vectorizer_file='vectorizer.json',
            # No Model hyper parameters
            # Training hyper parameters
            batch_size=128,
            early_stopping_criteria=5,
            learning_rate=0.005,
            num_epochs=200,
            seed=1337,
            # Runtime options
            catch_keyboard_interrupt=True,
            cuda=True,
            expand_filepaths_to_save_dir=True,
            reload_from_files=False,
            #End of rao's parameters

            #todo: get it from data
            num_classes=3,
            truncate_words_length=1000,
            type_of_data='plain',
            embedding_size=300,
            #parameter initialization gaussian
            para_init=0.01,
            hidden_sz=200,
            dataset='fever',
            arch='simple_MLP_embed_RTE',
            pretrained_wordemb='True',
            update_pretrained_wordemb='False',
            run_name='fever_transform',
            data_dir='../data-local/rte/fever',
            print_freq=1,
            workers=4,
            log_level='INFO',
            use_gpu=False,
            pretrained_wordemb_file='/Users/mordor/research/glove/glove.840B.300d.txt',
            use_double_optimizers=True,
            run_student_only=True,
            labels=20.0,
            consistency=1
        )

        if args.expand_filepaths_to_save_dir:
            args.vectorizer_file = os.path.join(args.save_dir,
                                                args.vectorizer_file)

            args.model_state_file = os.path.join(args.save_dir,
                                                 args.model_state_file)

            print("Expanded filepaths: ")
            print("\t{}".format(args.vectorizer_file))
            print("\t{}".format(args.model_state_file))

        # Check CUDA
        if not torch.cuda.is_available():
            args.cuda = False

        print("Using CUDA: {}".format(args.cuda))

        args.device = torch.device("cuda" if args.cuda else "cpu")

        # Set seed for reproducibility
        set_seed_everywhere(args.seed, args.cuda)
        handle_dirs(args.save_dir)

        return args



    def read_data_make_vectorizer(self,args):

        if args.reload_from_files:
            # training from a checkpoint
            dataset = RTEDataset.load_dataset_and_load_vectorizer(args.news_csv,
                                                                   args.vectorizer_file)
        else:
            # create dataset and vectorizer
            dataset = RTEDataset.load_dataset_and_make_vectorizer(args.news_csv)
            dataset.save_vectorizer(args.vectorizer_file)
        vectorizer = dataset.get_vectorizer()

        # Use GloVe or randomly initialized embeddings
        if args.use_glove:
            words = vectorizer.title_vocab._token_to_idx.keys()
            embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                               words=words)
            print("Using pre-trained embeddings")
        else:
            print("Not using pre-trained embeddings")
            embeddings = None


        if args.reload_from_files:
            # training from a checkpoint
            print("Loading dataset and vectorizer")
            dataset = RTEDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                     args.vectorizer_file)
        else:
            print("Loading dataset and creating vectorizer")
            # create dataset and vectorizer
            dataset = RTEDataset.load_dataset_and_make_vectorizer(args)
            dataset.save_vectorizer(args.vectorizer_file)

        return dataset,embeddings
