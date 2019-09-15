from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.model.classifier import DecompAttnClassifier,FFNNClassifier
from mean_teacher.model.train import Trainer
from mean_teacher.scripts.set_parameters import Initializer

rte=Initializer()
args=rte.set_parameters()
dataset, embeddings=rte.read_data_make_vectorizer(args)
vectorizer = dataset.get_vectorizer()
classifier = DecompAttnClassifier(len(vectorizer.claim_ev_vocab),args.embedding_size,args.hidden_sz, embeddings,
                  args.update_pretrained_wordemb, args.para_init, args.num_classes, args.use_gpu)
train_rte=Trainer()
train_rte.train(args)
