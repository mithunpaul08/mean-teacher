from argparse import Namespace
import os

import argparse
from mean_teacher.utils.utils_rao import set_seed_everywhere,make_embedding_matrix
from mean_teacher.utils.utils_rao import handle_dirs
from mean_teacher.modules.rao_datasets import RTEDataset
import torch


class Initializer():
    def __init__(self):
        self._args=Namespace()

    def set_default_parameters(self):

        args = Namespace(
            # Data and Path information
            frequency_cutoff=5,
            model_state_file='model.pth',
            # for laptop
            fever_lex_train_local='train/fever_train_lex.jsonl',
            fever_lex_dev_local='dev/fever_dev_lex.jsonl',
            fever_delex_train_local='train/fever_train_delex.jsonl',
            fever_delex_dev_local='dev/fever_dev_delex.jsonl',

            #for server
            fever_lex_train_server='train/fever_train_lex.jsonl',
            fever_lex_dev_server='dev/fever_dev_lex.jsonl',
            fever_delex_train_server='train/fever_train_delex.jsonl',
            fever_delex_dev_server='dev/fever_dev_delex.jsonl',

            data_dir_local='data/rte/fever',
            data_dir_server='data/rte/fever',
            save_dir='model_storage/',
            vectorizer_file='vectorizer.json',
            glove_filepath_local='data/glove/glove.840B.300d.txt',
            glove_filepath_server='/work/mithunpaul/glove/glove.840B.300d.txt',


            # Training hyper parameters
            batch_size=32,
            early_stopping_criteria=5,
            learning_rate=0.005,
            num_epochs=500,
            seed=256,
            random_seed=20,
            weight_decay=5e-5,
            Adagrad_init=0,

            # Runtime options
            expand_filepaths_to_save_dir=True,
            reload_from_files=False,
            load_model_from_disk_and_test=False,
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
            ema_decay=0.99,

            use_gpu=True,
            consistency_type="mse",
            NO_LABEL=-1

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
        parser.add_argument('--create_new_comet_graph', default=True, type=self.str2bool, metavar='BOOL',
                            help='used in comet graphing to decide if this has to go into an existing graph or create a new graph')
        parser.add_argument('--run_type', default="train", type=str,
                            help='type of run. options are: train (which includes val validation also),val, test')
        parser.add_argument('--add_student', default="False", type=self.str2bool,
                            help='for experiments. eg:running one student at a time')
        parser.add_argument('--consistency_weight', default=1, type=int,
                            help='for weighted average in the loss function')
        parser.add_argument('--use_semi_supervised', default="False", type=self.str2bool,
                            help='make a certain percentage of gold labels as -1')
        parser.add_argument('--percentage_labels_for_semi_supervised', default=0, type=int,
                            help='what percentage of gold labels do you want to hide for semi supervised learning')
        parser.add_argument('--which_gpu_to_use', default=0, type=int,
                            help='if you have more than 1 gpus and you know which one you want to run this code on Eg:2')
        parser.add_argument('--log_level', default='INFO', type=str,
                            help='choice between DEBUG, INFO ,ERROR ,WARNING')
        parser.add_argument('--consistency_type', default='mse', type=str,
                            help='choice between kl,mse')
        parser.add_argument('--use_ema', default="False", type=self.str2bool,
                            help='use teacher student architecture with exponential moving average/mean teacher')




        return parser

    def parse_commandline_args(self):
        return self.create_parser().parse_args(namespace=self._args)

    def str2bool(self,v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def get_file_paths(self,command_line_args):
        data_dir = self._args.data_dir_local
        glove_filepath_in = self._args.glove_filepath_local
        fever_lex_train_input_file = os.path.join(os.getcwd(),data_dir, self._args.fever_lex_train_local)
        fever_lex_dev_input_file = os.path.join(os.getcwd(),data_dir, self._args.fever_lex_dev_local)
        fever_delex_train_input_file = os.path.join(os.getcwd(),data_dir, self._args.fever_delex_train_local)
        fever_delex_dev_input_file = os.path.join(os.getcwd(),data_dir, self._args.fever_delex_dev_local)


        return glove_filepath_in,fever_lex_train_input_file,fever_lex_dev_input_file,fever_delex_train_input_file,fever_delex_dev_input_file