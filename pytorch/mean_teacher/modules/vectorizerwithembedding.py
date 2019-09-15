from collections import Counter
from .vocabulary import Vocabulary,SequenceVocabulary
import numpy as np
import string


# ### The Vectorizer


class VectorizerWithEmbedding(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, claim_ev_vocab, labels_vocab):
        self.claim_ev_vocab = claim_ev_vocab
        self.label_vocab = labels_vocab

    def vectorize(self, title, vector_length=-1):
        """
        Args:
            title (str): the string of words separated by a space
            vector_length (int): an argument for forcing the length of index vector
        Returns:
            the vetorized title (numpy.array)
        """
        indices = [self.claim_ev_vocab.begin_seq_index]
        indices.extend(self.claim_ev_vocab.lookup_token(token)
                       for token in title.split(" "))
        indices.append(self.claim_ev_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.claim_ev_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, claim_ev_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            claim_ev_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """

        claim_ev_vocab = SequenceVocabulary()
        word_counts = Counter()
        for claim, ev in zip(claim_ev_df.claim, claim_ev_df.evidence):
            combined_claim_ev = claim + ev
            for word in combined_claim_ev.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1


        for word, count in word_counts.items():
            if count > cutoff:
                claim_ev_vocab.add_token(word)

        labels_vocab = Vocabulary(add_unk=False)
        for label in sorted(set(claim_ev_df.label)):
            labels_vocab.add_token(label)

        return cls(claim_ev_vocab, labels_vocab)

    @classmethod
    def from_serializable(cls, contents):
        claim_ev_vocab_ser = SequenceVocabulary.from_serializable(contents['claim_ev_vocab_ser'])
        label_vocab_ser = SequenceVocabulary.from_serializable(contents['label_vocab_ser'])
        return cls(claim_ev_vocab=claim_ev_vocab_ser, labels_vocab=label_vocab_ser)

    def to_serializable(self):
        return {'claim_ev_vocab': self.claim_ev_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}