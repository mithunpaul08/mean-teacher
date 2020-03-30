from collections import Counter
from .vocabulary import Vocabulary,SequenceVocabulary
import numpy as np
import string
import re

# ### The Vectorizer


class VectorizerWithEmbedding(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, claim_ev_vocab, labels_vocab):
        self.claim_ev_vocab = claim_ev_vocab
        self.label_vocab = labels_vocab

    def vectorize(self, input_sentence, vector_length=-1):
        """
        Args:
            input_sentence (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        indices = [self.claim_ev_vocab.begin_seq_index]
        indices.extend(self.claim_ev_vocab.lookup_token(token)
                       for token in input_sentence.split(" "))
        indices.append(self.claim_ev_vocab.end_seq_index)

        #if we have not found or are providing the length of the input with maximum length.
        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.claim_ev_vocab.mask_index

        return out_vector

    def get_oanertag_label_frequency(self,oaner_label_freq,label,sentence):
        for word in sentence.split(" "):
            match=re.search(r'[A-Z]+-[c,e][0-9]+', word)
            if(match):
                oanertag=match.group()
                oaner_label = (oanertag,label)
                if oaner_label in oaner_label_freq:
                    oaner_label_freq[oaner_label]+=1
                else:
                    oaner_label_freq[oaner_label] = 1



    def update_word_count(self, sentence,word_counts):
            for word in sentence.split(" "):
                #if word not in string.punctuation:
                    word_counts[word] += 1
            return word_counts


    def get_oanertag_label_percentages(self,claim_ev_delex):
        oaner_label_freq = {}
        for index, row in (claim_ev_delex.iterrows()):
            self.get_oanertag_label_frequency(self, oaner_label_freq, row.label, row.claim)
            self.get_oanertag_label_frequency( self,oaner_label_freq, row.label, row.evidence)
        total = 0
        for x in (sorted(oaner_label_freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)):
            total += x[1]
        print(f"total={total}")
        for index, x in enumerate((sorted(oaner_label_freq.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))):
            val = round((x[1] * 100 / total), ndigits=2)
            print(f"{x}:{val}%")
            if (index > 100):
                import sys
                sys.exit(1)

    @classmethod
    def create_vocabulary(cls, claim_ev_lex, claim_ev_delex, args):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            claim_ev_lex (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """

        claim_ev_vocab = SequenceVocabulary()
        word_counts = Counter()
        for claim in (claim_ev_lex.claim):
            word_counts=cls.update_word_count(cls,claim,word_counts)
        for ev in (claim_ev_lex.evidence):
            word_counts=cls.update_word_count(cls, ev,word_counts)

        if(args.print_oaner_label_frequency==True):
            cls.get_oanertag_label_percentages(cls,claim_ev_delex)

        for claim in (claim_ev_delex.claim):
            word_counts=cls.update_word_count(cls,claim,word_counts)
        for ev in (claim_ev_delex.evidence):
            word_counts=cls.update_word_count(cls, ev,word_counts)


        for word, count in word_counts.items():
            # removing cutoff for the time being- to check if it increases accuracy
            # if count > cutoff:
                claim_ev_vocab.add_token(word)

        labels_vocab = Vocabulary(add_unk=False)
        for label in sorted(set(claim_ev_lex.label)):
            labels_vocab.add_token(label)

        return cls(claim_ev_vocab, labels_vocab)

    @classmethod
    def from_serializable(cls, contents):
        claim_ev_vocab_ser = SequenceVocabulary.from_serializable(contents['claim_ev_vocab'])
        label_vocab_ser = SequenceVocabulary.from_serializable(contents['label_vocab'])
        return cls(claim_ev_vocab=claim_ev_vocab_ser, labels_vocab=label_vocab_ser)

    def to_serializable(self):
        return {'claim_ev_vocab': self.claim_ev_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}