import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import export, parameter_count



@export
def simple_MLP_embed_RTE(word_vocab_size, num_classes, wordemb_size, pretrained=True, word_vocab_embed=None, hidden_size=200, update_pretrained_wordemb=False):

    model = FeedForwardMLPEmbed_RTE(word_vocab_size, wordemb_size, hidden_size, num_classes, word_vocab_embed, update_pretrained_wordemb)
    return model

class FeedForwardMLPEmbed_RTE(nn.Module):
    def __init__(self, word_vocab_size, embedding_size, hidden_sz, output_sz, word_vocab_embed, update_pretrained_wordemb):
        super().__init__()
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(word_vocab_size, embedding_size)
        print(f"inside architectures.py line 26 at 1 value of self.embeddings.weight is {self.embeddings.weight.shape} ")


        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222
        if word_vocab_embed is not None:  # Pre-initalize the embedding layer from a vector loaded from word2vec/glove/or such
            print("Using a pre-initialized word-embedding vector .. loaded from disk")
            self.embeddings.weight = nn.Parameter(torch.from_numpy(word_vocab_embed))
            print(f"at 2 value of self.embeddings.weight is {self.embeddings.weight} ")

            if update_pretrained_wordemb is False:
                # NOTE: do not update the emebddings
                # https://discuss.pytorch.org/t/how-to-exclude-embedding-layer-from-model-parameters/1283
                print("NOT UPDATING the word embeddings ....")
                self.embeddings.weight.detach_()
            else:
                print("UPDATING the word embeddings ....")
                print(f"at 2 value of self.embeddings.weight is {self.embeddings.weight} ")
                sys.exit(1)


        #todo: pass them from somewhere...maybe command line or config file
        self.NUM_CLASSES = 3
        self.CODE_PRCT_DROPOUT, self.COMM_PRCT_DROPOUT = 0.1, 0.1
        self.CODE_HD_SZ, self.COMM_HD_SZ = 50,50
        self.CODE_NUM_LAYERS, self.COMM_NUM_LAYERS = 2, 2


        # Creates a bidirectional LSTM for the code input
        self.lstm = nn.LSTM(embedding_size,  # Size of the code embedding
                                 self.CODE_HD_SZ,  # Size of the hidden layer
                                 num_layers=self.CODE_NUM_LAYERS,
                                 dropout=self.CODE_PRCT_DROPOUT,
                                 batch_first=True,
                                 bidirectional=True)

        # Size of the concatenated output from the 2 LSTMs
        self.CONCAT_SIZE = (self.CODE_HD_SZ + self.COMM_HD_SZ) * 2

        # FFNN layer to transform LSTM output into class predictions
        self.lstm2hidden = nn.Linear(self.CONCAT_SIZE, 50)
        self.hidden2label = nn.Linear(50, self.NUM_CLASSES)

        #todo: might have to add a softmax. look at what loss function you are using.-CrossEntropyLoss




    def forward(self, claim, evidence, claim_lengths, evidence_lengths):

        # keep track of how code and comm were sorted so that we can unsort them later
        # because packing requires them to be in descending order
        claim_lengths, claim_sort_order = claim_lengths.sort(descending=True)
        evidence_lengths, ev_sort_order = evidence_lengths.sort(descending=True)
        claim_inv_order = claim_sort_order.sort()[1]
        ev_inv_order = ev_sort_order.sort()[1]

        # Encode the batch input using word embeddings
        claim_encoding = self.embeddings(claim[claim_sort_order])
        ev_encoding = self.embeddings(evidence[ev_sort_order])

        # pack padded input
        claim_enc_pack = torch.nn.utils.rnn.pack_padded_sequence(claim_encoding, claim_lengths, batch_first=True)
        evidence_enc_pack = torch.nn.utils.rnn.pack_padded_sequence(ev_encoding, evidence_lengths, batch_first=True)

        # Run the LSTMs over the packed input
        #:claim_h_n hidden states at each word- will be used later when we have to get output of bilstm.

        #claim_c_n = context states at each word
        claim_enc_pad, (claim_h_n, claim_c_n) = self.lstm(claim_enc_pack)
        ev_enc_pad, (ev_h_n, ev_c_n) = self.lstm(evidence_enc_pack)

        # back to padding
        code_vecs, _ = torch.nn.utils.rnn.pad_packed_sequence(claim_enc_pad, batch_first=True)
        comm_vecs, _ = torch.nn.utils.rnn.pad_packed_sequence(ev_enc_pad, batch_first=True)

        # Concatenate the final output from both LSTMs
        # therefore claim_h_n[0]= hidden states at the end of forward lstm pass.
        # therefore claim_h_n[1]= hidden states at the end of backward lstm pass.
        recurrent_vecs = torch.cat((claim_h_n[0, claim_inv_order], claim_h_n[1, claim_inv_order],
                                    ev_h_n[0, ev_inv_order], ev_h_n[1, ev_inv_order]), 1)

        # Transform recurrent output vector into a class prediction vector
        y = F.relu(self.lstm2hidden(recurrent_vecs))
        y = self.hidden2label(y)

        return y

        #ask becky: entity in my case is claim. why does it have a max of 18719: i thought max of evidence was 18719. Max of claim was some 20 something
        #claim_embed = torch.mean(self.claim_embeddings(claim), 1)             # Note: Average the word-embeddings

        #ajay's code was like this, but was giving me dimension erorr while concatenation. I think he used this pattern_flattened, since his pattern has
        #two parts, via the student and teacher part...right now am going to just directly use pattern. mithun
        # pattern_flattened = evidence.view(evidence.size()[0], -1)                  # Note: Flatten the list of list of words into a list of words
        # evidence_embed = torch.mean((self.evidence_embeddings(pattern_flattened)), 1)  # Note: Average the words in every pattern in the list of patterns
       # evidence_embed = torch.mean(self.evidence_embeddings(evidence), 1)

        # print("claim_embed.size()")
        # print (claim_embed.size())
        # print (evidence_embed.size())
        # print(claim_embed.shape)
        # print(evidence_embed.shape)
        ## concatenate entity and pattern embeddings
        # concatenated = torch.cat([claim_embed, evidence_embed], 1)
        # res = self.layer1(concatenated)
        # res = self.activation(res)
        # res = self.layer2(res)
        # print (res)
        # print (res.shape)
        # res = self.softmax(res) ## IMPT NOTE: Removing the softmax from here as it is done in the loss function
        # print ("After softmax : " + str(res))
        #return res

