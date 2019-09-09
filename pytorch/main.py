from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.classifier import FFNNClassifier
from mean_teacher.model.train import Trainer
from mean_teacher.scripts.set_parameters import Initializer

rte=Initializer()
args=rte.set_parameters()
dataset=rte.read_data_make_vectorizer(args)
vectorizer = dataset.get_vectorizer()
classifier = FFNNClassifier(num_features=len(vectorizer.claim_ev_vocab))
train_rte=Trainer()
train_rte.train(args)
