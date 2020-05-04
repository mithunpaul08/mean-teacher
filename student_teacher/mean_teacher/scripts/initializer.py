from argparse import Namespace
import os

import argparse
from mean_teacher.utils.utils_rao import set_seed_everywhere,make_embedding_matrix
from mean_teacher.utils.utils_rao import handle_dirs
from mean_teacher.modules.rao_datasets import RTEDataset
import torch

logs_dir='log_dir/',

class Initializer():
    def __init__(self):
        self._args=Namespace()

    def set_default_parameters(self):

        args = Namespace(
            # Data and Path information
            frequency_cutoff=5,
            model_state_file='model',

            lex_train='fever/train/fever_train_lex.jsonl',
            lex_dev='fever/dev/fever_dev_lex.jsonl',

            #we are loading fnc dev as the test partitions now.
            # This is so that we can conduct simultaneous tests on fnc
            lex_test='fnc/dev/fnc_dev_lex.jsonl',
            delex_train= 'fever/train/fever_train_delex.jsonl',
            delex_dev='fever/dev/fever_dev_delex.jsonl',
            delex_test='fnc/dev/fnc_dev_delex.jsonl',





            data_dir='data/rte',
            logs_dir='log_dir/',
            predictions_teacher_dev_file='log_dir/predictions_teacher_dev.jsonl',
            predictions_student_dev_file="log_dir/predictions_student_dev.jsonl",
            predictions_teacher_test_file='log_dir/predictions_teacher_test.jsonl',
            predictions_student_test_file="log_dir/predictions_student_test.jsonl",



            save_dir='model_storage/',
            vectorizer_file='best_vectorizer.json',
            glove_filepath='data/glove/glove.840B.300d.txt',
            gigaword_file_path='data/gigaword/gigawordDocFreq.sorted.freq.txt',
            #pick only words from gigaward corpora which have  frequency above this value
            gw_minfreq=30,
            shuffle_data=False,


            # Training hyper parameters
            batch_size=32,
            early_stopping_criteria=5,
            learning_rate=0.005,
            num_epochs=10000,
            random_seed=676786,

            weight_decay=5e-5,
            Adagrad_init=0,
            type_of_trained_model="teacher",

            # Runtime options
            expand_filepaths_to_save_dir=True,
            reload_data_from_files=False,
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
            update_pretrained_wordemb=True,
            cuda=True,
            workers=0,
            ema_decay=0.99,
            database_to_test_with='fever',

            use_gpu=True,
            consistency_type="mse",
            NO_LABEL=-1,
            #this is used during loading a trained model and testing with it. you can choose between teacher and student.

            #will print top 10  percentage of [oanerTag, label] combination.
            #Eg:(('PERSON-c1', 'AGREE'), 51):5.57% and exit
            # This was needed to show in LREC2020 that
            #even though we overcame one bias (ben stiller effect like) we created
            #new ones based on oaner tags
            print_oaner_label_frequency=False,
            test_in_cross_domain_dataset=True


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


        # Set seed for reproducibility
        set_seed_everywhere(args.random_seed, args.cuda)
        handle_dirs(args.save_dir)
        self._args=args

        # Check CUDA
        if not torch.cuda.is_available():
            args.cuda = False
        print("Using CUDA: {}".format(args.cuda))
        args.device = torch.device("cuda" if args.cuda else "cpu")


        return args

    def create_parser(self):

        parser = argparse.ArgumentParser(description='PyTorch Mean-Teacher Training')
        parser.add_argument('--create_new_comet_graph', default=True, type=self.str2bool, metavar='BOOL',
                            help='used in comet graphing to decide if this has to go into an existing graph or create a new graph')
        parser.add_argument('--run_type', default="train", type=str,
                            help='type of run. options are: train (which includes val validation also),val, test')
        parser.add_argument('--add_student', default="False", type=self.str2bool,
                            help='for experiments like eg:running teacher only ')
        parser.add_argument('--consistency_weight', default=3, type=float,
                            help='for weighted average in the loss function')
        parser.add_argument('--use_semi_supervised', default="False", type=self.str2bool,
                            help='make a certain percentage of gold LABELS as -1')
        parser.add_argument('--percentage_labels_for_semi_supervised', default=0, type=int,
                            help='what percentage of gold LABELS do you want to hide for semi supervised learning')
        parser.add_argument('--which_gpu_to_use', default=0, type=int,
                            help='if you have more than 1 gpus and you know which one you want to run this code on Eg:2')

        parser.add_argument('--log_level', default='INFO', type=str,
                            help='choice between DEBUG, INFO ,ERROR ,WARNING')
        parser.add_argument('--consistency_type', default='mse', type=str,
                            help='choice between kl,mse')
        parser.add_argument('--use_ema', default="False", type=self.str2bool,
                            help='use teacher student architecture with exponential moving average/mean teacher')
        parser.add_argument('--lex_train_full_path', default="data/rte/fever/train/fever_train_lex.jsonl", type=str,
                            help='input file lexicalized data')
        parser.add_argument('--load_model_from_disk_and_test', default="False", type=self.str2bool,
                            help='when you have a trained model that you want to load and test using it')
        parser.add_argument('--trained_model_path', default="model_storage/best_model.pth", type=str,
                            help='')
        parser.add_argument('--use_trained_teacher_inside_student_teacher_arch', default="False", type=self.str2bool,
                            help='when you have a trained teachear model that you want to load and train student using it')
        parser.add_argument('--batch_size', default=32, type=int,
                            help='number of data points per batch. 11919 makes 10 batches of fever training data')



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

    #todo get all input file paths from command line or a shell script
    def get_file_paths(self,LOG):
        glove_filepath = self._args.glove_filepath
        gigaword_full_path = self._args.gigaword_file_path


        lex_train_full_path = os.path.join(os.getcwd(), self._args.data_dir,self._args.lex_train_full_path)
        lex_dev_full_path = os.path.join(os.getcwd(), self._args.data_dir, self._args.lex_dev)
        lex_test_full_path = os.path.join(os.getcwd(), self._args.data_dir, self._args.lex_test)

        delex_train_full_path = os.path.join(os.getcwd(), self._args.data_dir, self._args.delex_train)
        delex_dev_full_path = os.path.join(os.getcwd(), self._args.data_dir, self._args.delex_dev)




        delex_test_full_path = os.path.join(os.getcwd(), self._args.data_dir, self._args.delex_test)

        LOG.info(f" lex_train_full_path:{lex_train_full_path} ")
        LOG.info(f" lex_dev_full_path:{lex_dev_full_path} ")
        LOG.info(f" lex_test_full_path:{lex_test_full_path} ")
        LOG.info(f" delex_train_full_path:{delex_train_full_path} ")
        LOG.info(f" delex_dev_full_path:{delex_dev_full_path} ")
        LOG.info(f" delex_test_full_path:{delex_test_full_path} ")


        assert glove_filepath is not None
        assert lex_train_full_path is not None
        assert lex_dev_full_path is not None
        assert lex_test_full_path is not None
        assert delex_train_full_path is not None
        assert delex_dev_full_path is not None
        assert delex_test_full_path is not None

        assert os.path.exists(lex_train_full_path) is True
        assert os.path.exists(lex_dev_full_path) is True
        assert os.path.exists(lex_test_full_path) is True
        assert os.path.exists(lex_train_full_path) is True
        assert os.path.exists(delex_train_full_path) is True
        assert os.path.exists(delex_dev_full_path) is True
        assert os.path.exists(delex_test_full_path) is True


        return glove_filepath, lex_train_full_path, lex_dev_full_path, lex_test_full_path,delex_train_full_path, delex_dev_full_path, delex_test_full_path,gigaword_full_path





