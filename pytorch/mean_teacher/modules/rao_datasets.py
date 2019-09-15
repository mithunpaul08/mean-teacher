# ### The Dataset

import json
from torch.utils.data import Dataset, DataLoader
from  .vectorizerwithembedding import VectorizerWithEmbedding
import pandas as pd
from mean_teacher.utils.utils_valpola import export
import os

# @export
# def fever():
#
#     if RTEDataset.WORD_NOISE_TYPE in ['drop', 'replace']:
#         addNoise = data.RandomPatternWordNoise(RTEDataset.NUM_WORDS_TO_REPLACE, RTEDataset.OOV, RTEDataset.WORD_NOISE_TYPE)
#     else:
#         assert False, "Unknown type of noise {}".format(RTEDataset.WORD_NOISE_TYPE)
#
#     return {
#         'train_transformation': None,
#         'eval_transformation': None,
#     }

class RTEDataset(Dataset):
    def __init__(self, claims_evidences_df, vectorizer):
        """
        Args:
            claims_evidences_df (pandas.DataFrame): the dataset
            vectorizer (VectorizerWithEmbedding): vectorizer instantiated from dataset
        """
        self.claims_ev_df = claims_evidences_df
        self._vectorizer = vectorizer

        self.train_df = self.claims_ev_df[self.claims_ev_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.claims_ev_df[self.claims_ev_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.claims_ev_df[self.claims_ev_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')


    @classmethod
    def load_dataset_and_make_vectorizer(cls, args):
        """Load dataset and make a new vectorizer from scratch

        Args:
            args (str): all arguments which were create initially.
        Returns:
            an instance of ReviewDataset
        """
        fever_lex_train_df = pd.read_json(os.path.join(args.data_dir,args.fever_lex_train), lines=True)
        fever_lex_train_df['split'] = "train"

        fever_lex_dev_df = pd.read_json(os.path.join(args.data_dir,args.fever_lex_dev), lines=True)
        fever_lex_dev_df['split'] = "val"

        frames = [fever_lex_train_df, fever_lex_dev_df]
        combined_train_dev_test_with_split_column_df = pd.concat(frames)

        # todo: uncomment/call and check the function replace_if_PERSON_C1_format has any effect on claims and evidence sentences-mainpulate dataframe
        return cls(combined_train_dev_test_with_split_column_df, VectorizerWithEmbedding.from_dataframe(fever_lex_train_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, input_file, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            input_file (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of ReviewDataset
        """
        print(f"just before reading file {input_file}")
        review_df = cls.read_rte_data(input_file)

        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(review_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of ReviewVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return VectorizerWithEmbedding.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        combined_claim_evidence=row.claim+row.evidence
        claim_evidence_vector = \
            self._vectorizer.vectorize(combined_claim_evidence)

        label_index = \
            self._vectorizer.label_vocab.lookup_token(row.label)

        return {'x_data': claim_evidence_vector,
                'y_target': label_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


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