import numpy as np
import torch
import os
import re
import mmap
from torch.utils.data import DataLoader
from tqdm import tqdm
from mean_teacher.model import architectures
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import math

# #### General utilities

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text



def generate_batches(dataset,workers,batch_size,device ,shuffle=False,
                     drop_last=True ):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """


    if(shuffle==True):
        labeled_idxs = dataset.get_all_label_indices(dataset)
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler_local = BatchSampler(sampler, batch_size, drop_last=True)
        dataloader=DataLoader(dataset,batch_sampler=batch_sampler_local,num_workers=workers,pin_memory=True)
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,pin_memory=True,drop_last=False,num_workers=workers)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def generate_batches_for_semi_supervised(dataset,percentage_labels_for_semi_supervised,workers,batch_size,device,shuffle=True,
                     drop_last=True,mask_value=-1 ):
    '''
    similar to generate_batches but will mask/replace the labels of certain certain percentage of indices with -1. a
    :param dataset:
    :param workers:
    :param batch_size:
    :param device:
    :param shuffle:
    :param drop_last:
    :return: BatchSampler
    '''


    if(shuffle==True):
        labeled_idxs = dataset.get_all_label_indices(dataset)
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler_local = BatchSampler(sampler, batch_size, drop_last=True)
        dataloader=DataLoader(dataset,batch_sampler=batch_sampler_local,num_workers=workers,pin_memory=True)
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,pin_memory=True,drop_last=False,num_workers=workers)


    count_indices_to_mask= math.ceil(batch_size* (percentage_labels_for_semi_supervised)/100)
    mask=torch.randint(0,batch_size-1,(count_indices_to_mask,))


    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            if(name=="y_target"):
                for m in mask:
                    tensor[m]=mask_value
                    out_data_dict[name] = data_dict[name].to(device)
            else:
                out_data_dict[name] = data_dict[name].to(device)

        yield out_data_dict


def get_num_lines(file_path):

    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    total_lines = get_num_lines(glove_filepath)
    with open(glove_filepath, "r") as fp:
        for index, line in tqdm(enumerate(fp),total=total_lines, desc="glove"):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """

    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings,embedding_size

def initialize_optimizers(list_models, args):

    '''
        The code for decomposable attention we use , utilizes two different optimizers
    :param model:
    :param args:
    :return:
    '''
    combined_para1= []
    combined_para2 = []
    for model in list_models:
        combined_para1  =   combined_para1  + list(model.para1)
        combined_para2  =   combined_para2  + list(model.para2)

    input_optimizer = None
    inter_atten_optimizer = None

    if args.optimizer == 'adagrad':
        input_optimizer = torch.optim.Adagrad(combined_para1, lr=args.learning_rate, weight_decay=args.weight_decay)
        inter_atten_optimizer = torch.optim.Adagrad(combined_para2, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        input_optimizer = torch.optim.Adadelta(combined_para1, lr=args.lr)
        inter_atten_optimizer = torch.optim.Adadelta(combined_para2, lr=args.lr)
    else:
        print('No Optimizer.')
        import sys
        sys.exit()
    assert input_optimizer != None
    assert inter_atten_optimizer != None

    return input_optimizer,inter_atten_optimizer

def update_optimizer_state(input_optimizer, inter_atten_optimizer,args):
    for group in input_optimizer.param_groups:
        for p in group['params']:
            state = input_optimizer.state[p]
            state['sum'] += args.Adagrad_init
    for group in inter_atten_optimizer.param_groups:
        for p in group['params']:
            state = inter_atten_optimizer.state[p]
            state['sum'] += args.Adagrad_init
    return input_optimizer, inter_atten_optimizer


def create_model(logger_object, args_in,  num_classes_in, word_vocab_embed, word_vocab_size, wordemb_size_in,ema=False,):
    logger_object.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained '
            if args_in.pretrained else '',
            ema='EMA '
            if ema else '',arch=args_in.arch))
    model_factory = architectures.__dict__[args_in.arch]
    model_params = dict(pretrained=args_in.pretrained, num_classes_in=num_classes_in)
    model_params['word_vocab_embed'] = word_vocab_embed
    model_params['word_vocab_size'] = word_vocab_size
    model_params['wordemb_size'] = wordemb_size_in
    model_params['hidden_size'] = args_in.hidden_sz
    model_params['update_pretrained_wordemb'] = args_in.update_pretrained_wordemb
    model_params['para_init'] = args_in.para_init
    model_params['use_gpu'] = args_in.use_gpu
    logger_object.debug(f"value of word_vocab_embed={word_vocab_embed}")
    logger_object.debug(f"value of word_vocab_size={word_vocab_size}")

    model = model_factory(**model_params)

    args_in.device=None
    if(args_in.use_gpu) and torch.cuda.is_available():
        logger_object.info("found that GPU is available")
        torch.cuda.set_device(args_in.which_gpu_to_use)
        args_in.device = torch.device('cuda')
        logger_object.info(f"will be using gpu number{args_in.which_gpu_to_use}")
    else:
        args_in.device = torch.device('cpu')

    model = model.to(device=args_in.device)

    if ema:
        for param in model.parameters():
            param.detach_() ##NOTE: Detaches the variable from the gradient computation, making it a leaf .. needed from EMA model

    return model