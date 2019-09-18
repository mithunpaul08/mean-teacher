import numpy as np
import torch
import os
import re
import mmap
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
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

def initialize_double_optimizers(model, args):

    '''
        The code for decomposable attention we use , utilizes two different optimizers
    :param model:
    :param args:
    :return:
    '''
    input_optimizer = None
    inter_atten_optimizer = None
    para1 = model.para1
    para2 = model.para2
    if args.optimizer == 'adagrad':
        input_optimizer = torch.optim.Adagrad(para1, lr=args.learning_rate, weight_decay=args.weight_decay)
        inter_atten_optimizer = torch.optim.Adagrad(para2, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        input_optimizer = torch.optim.Adadelta(para1, lr=args.lr)
        inter_atten_optimizer = torch.optim.Adadelta(para2, lr=args.lr)
    else:
        #LOG.info('No Optimizer.')
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