from collections import Counter
from .vocabulary import Vocabulary,SequenceVocabulary
import numpy as np
import string
import re
from mean_teacher.utils.logger import Logger


# ### The Vectorizer
LABELS=["AGREE", "DISAGREE", "DISCUSS", "UNRELATED"]
gigaword_freq={}
logger_client=Logger()
LOG=logger_client.initialize_logger()

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
            cutoff (int): the parameter for frequency-based filteringgenerate_batches_without_sampler
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

        # update@april 11th 2020: mihai said create a vocabulary which is a union of training data and  top n freq words from gigaword.
        #  as per mihai this enhances/is usefull to reduce
        # the number of uNK words in dev/test partitions- especially when either of those tend to be cross-domain.
        #  though i still think its cheating- mithun
        assert len(gigaword_freq.items()) > 0

        singletons=0
        singletons_list=[]
        for word,count in word_counts.items():
            if count==1:
                singletons+=1
                singletons_list.append(word)


        LOG.info(
            f"total number of singletons in Training alone before merging with gigaword is {(singletons)}")

        LOG.info(f"going to merge gigaword vocabulary with training vocabulary. length of training vocab now  is {len(word_counts)}")
        for word, count in gigaword_freq.items():
            #if the word exists in gigaword, replace the frequency with that in gigaword, not in training data. This is useful when we have to pick singleton words
            #i.e words that occur only once in training data (either has freq =1 in gigaword, or doesn't exist in gigaword)
            if(word not in word_counts):
                word_counts[word]=1
            else:
                word_counts[word] = count
        LOG.info(f"after merging gigaword vocabulary with training vocabulary. length of training vocab now   is {len(word_counts)}")
        LOG.info(f"frequency of the word 'the'  is {word_counts['the']}. note: must be 9615720 since that is the freq in gigaword")
        LOG.info(
            f"frequency of the word 'Roman'  is {word_counts['Roman']}. note: must be 1.")

        singletons = 0
        for word, count in word_counts.items():
            if count == 1:
                singletons += 1
        LOG.info(
            f"total number of singletons in both gigaword and training data together is {(singletons)}")


        for word, count in word_counts.items():
                claim_ev_vocab.add_token(word)


        singletons=0
        for singleton in singletons_list:

            if (word_counts[singleton])==1:
                singletons += 1

        LOG.info(
            f"total number of singletons that are only Training after merging with gigaword is {(singletons)}")

        import sys
        sys.exit(1)

        claim_ev_vocab.add_word_frequency(word_counts)
        labels_vocab = Vocabulary(add_unk=False)
        for label in sorted(set(claim_ev_lex.label)):
            labels_vocab.add_token(label)

        return cls(claim_ev_vocab, labels_vocab)

    @classmethod
    def from_serializable(cls, contents):
        claim_ev_vocab_ser = SequenceVocabulary.from_serializable(contents['claim_ev_vocab'])
        label_vocab_ser = Vocabulary.from_serializable(contents['label_vocab'])
        return cls(claim_ev_vocab=claim_ev_vocab_ser, labels_vocab=label_vocab_ser)

    def to_serializable(self):
        return {'claim_ev_vocab': self.claim_ev_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}