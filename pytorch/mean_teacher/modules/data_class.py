# ### The Dataset

import json
from torch.utils.data import Dataset, DataLoader
from  .vectorizer import ReviewVectorizer
import pandas as pd
from mean_teacher.utils import export

@export
def fever():

    if RTEDataset2.WORD_NOISE_TYPE in ['drop', 'replace']:
        addNoise = data.RandomPatternWordNoise(RTEDataset2.NUM_WORDS_TO_REPLACE, RTEDataset2.OOV, RTEDataset2.WORD_NOISE_TYPE)
    else:
        assert False, "Unknown type of noise {}".format(RTEDataset2.WORD_NOISE_TYPE)

    return {
        'train_transformation': None,
        'eval_transformation': None,
    }

class RTEDataset2(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        Args:
            review_df (pandas.DataFrame): the dataset
            vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    def replace_if_PERSON_C1_format(sent, args):
        sent_replaced = ""
        regex = re.compile('([A-Z]+)(-)([ce])([0-99])')
        if (args.type_of_data == "ner_replaced" and regex.search(sent)):
            sent_replaced = regex.sub(r'\1\3\4', sent)
        else:
            sent_replaced = sent
        return sent_replaced

    def get_num_lines(file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    @classmethod
    def load_dataset_and_make_vectorizer(cls, args):
        """Load dataset and make a new vectorizer from scratch

        Args:
            args (str): all arguments which were create initially.
        Returns:
            an instance of ReviewDataset
        """
        fever_lex_train_df = pd.read_json(args.train_input_file, lines=True)
        fever_lex_train_df['split'] = "train"

        fever_lex_dev_df = pd.read_json(args.dev_input_file.jsonl, lines=True)
        fever_lex_dev_df['split'] = "val"

        frames = [fever_lex_train_df, fever_lex_dev_df]
        combined_train_dev_test_with_split_column_df = pd.concat(frames)

        # todo: uncomment/call and check the function replace_if_PERSON_C1_format has any effect on claims and evidence sentences-mainpulate dataframe
        return cls(combined_train_dev_test_with_split_column_df, ReviewVectorizer.from_dataframe(fever_lex_train_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            review_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of ReviewDataset
        """
        print(f"just before reading file {review_csv}")
        review_df = cls.read_rte_data(review_csv)

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
            return ReviewVectorizer.from_serializable(json.load(fp))

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
            self._vectorizer.rating_vocab.lookup_token(row.label)

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