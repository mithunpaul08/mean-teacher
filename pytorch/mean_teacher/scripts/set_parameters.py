from argparse import Namespace
import torch
import os
import argparse
from mean_teacher.utils.utils_rao import set_seed_everywhere,make_embedding_matrix
from mean_teacher.utils.utils_rao import handle_dirs
from mean_teacher.modules.rao_datasets import RTEDataset

class Initializer():
    def set_parameters(self):
        args = Namespace(
            # Data and Path information
            frequency_cutoff=25,
            model_state_file='model.pth',
            # for laptop
            fever_train_local='train/fever_train_lex_3labels_11k_smartner_3labels_no_lists_evidence_not_sents.jsonl',
            fever_dev_local='dev/fever_dev_lex_3labels_2k_no_lists_evidence_not_sents.jsonl',

            #for server
            fever_train_server='train/fever_train_delex_smartner_119k_3labels_no_lists_evidence_not_sents.jsonl',
            fever_dev_server='dev/fever_dev_delexicalized_3labels_26k.jsonl',

            save_dir='model_storage/ch3/yelp/',
            vectorizer_file='vectorizer.json',
            # No Model hyper parameters
            # Training hyper parameters
            batch_size=1000,
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

            use_glove=True,
            truncate_words_length=1000,
            type_of_data='plain',
            embedding_size=300,
            #parameter initialization gaussian
            para_init=0.01,
            hidden_sz=200,
            dataset='fever',
            arch='simple_MLP_embed_RTE',
            pretrained_wordemb=True,
            update_pretrained_wordemb=False,
            run_name='fever_transform',
            data_dir_local='../data-local/rte/fever',
            data_dir_server='data-local/rte/fever',
            print_freq=1,
            workers=4,
            log_level='INFO',
            use_gpu=False,
            glove_filepath_local='/Users/mordor/research/glove/glove.840B.300d.txt',
            glove_filepath_server='/work/mithunpaul/glove/glove.840B.300d.txt',
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

    def create_parser(self):
        parser = argparse.ArgumentParser(description='PyTorch Mean-Teacher Training')
        parser.add_argument('--run_on_server', default=False, type=self.str2bool, metavar='BOOL',
                            help='exclude unlabeled examples from the training set')

        return parser

    def parse_commandline_args(self):
        return self.create_parser().parse_args()

    def str2bool(self,v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
