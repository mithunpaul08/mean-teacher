from argparse import Namespace
import torch
import os
import argparse
from mean_teacher.utils.utils_rao import set_seed_everywhere,make_embedding_matrix
from mean_teacher.utils.utils_rao import handle_dirs
from mean_teacher.modules.rao_datasets import RTEDataset

class Initializer():
    def __init__(self):
        self._args=Namespace()

    def set_parameters(self):
        args = Namespace(
            # Data and Path information
            frequency_cutoff=5,
            model_state_file='model.pth',
            # for laptop
            fever_train_local='train/fever_train_lex_3labels_400_smartner_3labels_no_lists_evidence_not_sents.jsonl',
            fever_dev_local='dev/fever_dev_lex_3labels_100_no_lists_evidence_not_sents.jsonl',

            #for server
            fever_train_server='train/fever_train_lex_3labels_119k_smartner_3labels_no_lists_evidence_not_sents.jsonl',
            fever_dev_server='dev/fever_dev_lex_3labels_26k_no_lists_evidence_not_sents.jsonl',
            data_dir_local='../data-local/rte/fever',
            data_dir_server='data-local/rte/fever',
            save_dir='model_storage/ch3/yelp/',
            vectorizer_file='vectorizer.json',
            glove_filepath_local='/Users/mordor/research/glove/glove.840B.300d.txt',
            glove_filepath_server='/work/mithunpaul/glove/glove.840B.300d.txt',


            # Training hyper parameters
            batch_size=32,
            early_stopping_criteria=5,
            learning_rate=0.005,
            num_epochs=500,
            seed=256,
            weight_decay=5e-5,
            Adagrad_init=0,

            # Runtime options
            expand_filepaths_to_save_dir=True,
            reload_from_files=False,
            max_grad_norm=5,
            #End of rao's parameters


            truncate_words_length=1000,
            embedding_size=300,
            optimizer="adagrad",
            para_init=0.01,
            hidden_sz=200,
            arch='decomp_attention',
            pretrained="false",
            update_pretrained_wordemb=False,
            cuda=True,
            workers=0,


            #workers=4,
            log_level='INFO',
            use_gpu=True
        )
        args.use_glove = True
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
        self._args=args

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

    def get_file_paths(self,command_line_args):
        '''
        decide the path of the local files based on whether we are running on server or laptop.
        #todo: move this to config file
        :return:
        '''

        data_dir = self._args.data_dir_local
        glove_filepath_in = self._args.glove_filepath_local
        fever_train_input_file = os.path.join(data_dir, self._args.fever_train_local)
        fever_dev_input_file = os.path.join(data_dir, self._args.fever_dev_local)

        if (command_line_args.run_on_server == True):
            glove_filepath_in = self._args.glove_filepath_server
            fever_train_input_file = os.path.join(self._args.data_dir_server, self._args.fever_train_server)
            fever_dev_input_file = os.path.join(self._args.data_dir_server, self._args.fever_dev_server)
        return glove_filepath_in,fever_train_input_file,fever_dev_input_file