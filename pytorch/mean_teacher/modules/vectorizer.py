from collections import Counter
from .vocabulary import Vocabulary
import numpy as np
import string


# ### The Vectorizer


class Vectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, claim_ev_vocab, labels_vocab):
        """
        Args:
            claim_ev_vocab (Vocabulary): maps words to integers
            labels_vocab (Vocabulary): maps class labels to integers
        """
        self.claim_ev_vocab = claim_ev_vocab
        self.label_vocab = labels_vocab

    def vectorize(self, review):
        """Create a collapsed one-hit vector for the review

        Args:
            review (str): the review
        Returns:
            one_hot (np.ndarray): the collapsed one-hot encoding
        """
        one_hot = np.zeros(len(self.claim_ev_vocab), dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.claim_ev_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, claim_ev_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            claim_ev_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """
        claim_ev_vocab = Vocabulary(add_unk=True)
        labels_vocab = Vocabulary(add_unk=False)

        # Add ratings
        for label in sorted(set(claim_ev_df.label)):
            labels_vocab.add_token(label)

        # Add top words if count > provided count
        word_counts = Counter()
        for claim,ev in zip(claim_ev_df.claim,claim_ev_df.evidence):
            combined_claim_ev=claim+ev
            for word in combined_claim_ev.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                claim_ev_vocab.add_token(word)

        return cls(claim_ev_vocab, labels_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a ReviewVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents['claim_ev_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['label_vocab'])

        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        """Create the serializable dictionary for caching

        Returns:
            contents (dict): the serializable dictionary
        """
        return {'claim_ev_vocab': self.claim_ev_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}