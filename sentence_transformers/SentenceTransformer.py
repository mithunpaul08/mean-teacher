import json
import logging
import os
import shutil
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type
from zipfile import ZipFile

import numpy as np
import transformers
import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from . import __DOWNLOAD_SERVER__
from .evaluation import SentenceEvaluator
from .util import import_from_string, batch_to_device, http_get
from . import __version__

#import related to student teacher architecture
from student_teacher.mean_teacher.utils import losses
import git

NO_LABEL=-1


if torch.cuda.is_available():
    class_loss_func = nn.CrossEntropyLoss(ignore_index=NO_LABEL).cuda()
else:
    class_loss_func = nn.CrossEntropyLoss(ignore_index=NO_LABEL).cpu()


class SentenceTransformer(nn.Sequential):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: str = None):
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        if model_name_or_path is not None and model_name_or_path != "":
            logging.info("Load pretrained SentenceTransformer: {}".format(model_name_or_path))

            if '/' not in model_name_or_path and '\\' not in model_name_or_path and not os.path.isdir(model_name_or_path):
                logging.info("Did not find a '/' or '\\' in the name. Assume to download model from server.")
                model_name_or_path = __DOWNLOAD_SERVER__ + model_name_or_path + '.zip'

            if model_name_or_path.startswith('http://') or model_name_or_path.startswith('https://'):
                model_url = model_name_or_path
                folder_name = model_url.replace("https://", "").replace("http://", "").replace("/", "_")[:250]

                try:
                    from torch.hub import _get_torch_home
                    torch_cache_home = _get_torch_home()
                except ImportError:
                    torch_cache_home = os.path.expanduser(
                        os.getenv('TORCH_HOME', os.path.join(
                            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
                default_cache_path = os.path.join(torch_cache_home, 'sentence_transformers')
                model_path = os.path.join(default_cache_path, folder_name)
                os.makedirs(model_path, exist_ok=True)

                if not os.listdir(model_path):
                    if model_url[-1] == "/":
                        model_url = model_url[:-1]
                    logging.info("Downloading sentence transformer model from {} and saving it at {}".format(model_url, model_path))
                    try:
                        zip_save_path = os.path.join(model_path, 'model.zip')
                        http_get(model_url, zip_save_path)
                        with ZipFile(zip_save_path, 'r') as zip:
                            zip.extractall(model_path)
                    except Exception as e:
                        shutil.rmtree(model_path)
                        raise e
            else:
                model_path = model_name_or_path

            #### Load from disk
            if model_path is not None:
                logging.info("Load SentenceTransformer from folder: {}".format(model_path))

                if os.path.exists(os.path.join(model_path, 'config.json')):
                    with open(os.path.join(model_path, 'config.json')) as fIn:
                        config = json.load(fIn)
                        if config['__version__'] > __version__:
                            logging.warning("You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(config['__version__'], __version__))

                with open(os.path.join(model_path, 'modules.json')) as fIn:
                    contained_modules = json.load(fIn)

                modules = OrderedDict()
                for module_config in contained_modules:
                    module_class = import_from_string(module_config['type'])
                    module = module_class.load(os.path.join(model_path, module_config['path']))
                    modules[module_config['name']] = module


        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info("Use pytorch device: {}".format(device))

        self.device = torch.device(device)
        self.to(device)

    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None, output_value: str = 'sentence_embedding', convert_to_numpy: bool = True) -> List[ndarray]:
        """
        Computes sentence embeddings

        :param sentences:
           the sentences to embed
        :param batch_size:
           the batch size used for the computation
        :param show_progress_bar:
            Output a progress bar when encode sentences
        :param output_value:
            Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings
            to get wordpiece token embeddings.
        :param convert_to_numpy:
            If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :return:
           Depending on convert_to_numpy, either a list of numpy vectors or a list of pytorch tensors
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                tokens = self.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                #features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.device)
                features[feature_name] = torch.cat(features[feature_name]).to(self.device)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                if convert_to_numpy:
                    embeddings = embeddings.to('cpu').numpy()

                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]

        return all_embeddings

    def get_max_seq_length(self):
        if hasattr(self._first_module(), 'max_seq_length'):
            return self._first_module().max_seq_length

        return None

    def tokenize(self, text):
        return self._first_module().tokenize(text)

    def get_sentence_features(self, *features):
        return self._first_module().get_sentence_features(*features)

    def get_sentence_embedding_dimension(self):
        return self._last_module().get_sentence_embedding_dimension()

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def save(self, path):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders
        """
        if path is None:
            return

        logging.info("Save model to {}".format(path))
        contained_modules = []

        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, str(idx)+"_"+type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(path, 'modules.json'), 'w') as fOut:
            json.dump(contained_modules, fOut, indent=2)

        with open(os.path.join(path, 'config.json'), 'w') as fOut:
            json.dump({'__version__': __version__}, fOut, indent=2)

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0][0])

        labels = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label in batch:
            labels.append(label)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        features = []
        for idx in range(num_texts):
            max_len = max_seq_len[idx]
            feature_lists = {}

            for text in paired_texts[idx]:
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []

                    feature_lists[feature_name].append(sentence_features[feature_name])


            for feature_name in feature_lists:
                #feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))
                feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

            features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}



    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object ]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            fp16: bool = False,
            fp16_opt_level: str = 'O1',
            local_rank: int = -1
            ):
        """
        Train the model with the given training objective

        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param weight_decay:
        :param scheduler:
        :param warmup_steps:
        :param optimizer:
        :param evaluation_steps:
        :param output_path:
        :param save_best_model:
        :param max_grad_norm:
        :param fp16:
        :param fp16_opt_level:
        :param local_rank:
        :param train_objectives:
            Tuples of DataLoader and LossConfig
        :param evaluator:
        :param epochs:
        :param steps_per_epoch: Train for x steps in each epoch. If set to None, the length of the dataset will be used
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if os.listdir(output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    output_path))

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        device = self.device

        for loss_model in loss_models:
            loss_model.to(device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = num_train_steps
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            for train_idx in range(len(loss_models)):
                model, optimizer = amp.initialize(loss_models[train_idx], optimizers[train_idx], opt_level=fp16_opt_level)
                loss_models[train_idx] = model
                optimizers[train_idx] = optimizer

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="batches", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        #logging.info("Restart data_iterator")
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = batch_to_device(data, self.device)
                    loss_value = loss_model(features, labels)

                    if fp16:
                        with amp.scale_loss(loss_value, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

    def evaluate(self, evaluator: SentenceEvaluator, output_path: str = None):
        """
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if score > self.best_score and save_best_model:
                self.save(output_path)
                self.best_score = score


    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    # code used in uofa student teacher architecture
    def train_1teacher(self, args_in,
                       train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
                       evaluators: Iterable[SentenceEvaluator],
                       epochs: int = 1,
                       steps_per_epoch=None,
                       scheduler: str = 'WarmupLinear',
                       warmup_steps: int = 10000,
                       optimizer_class: Type[Optimizer] = transformers.AdamW,
                       optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
                       weight_decay: float = 0.01,
                       evaluation_steps: int = 0,
                       output_path: str = None,
                       save_best_model: bool = True,
                       max_grad_norm: float = 1,
                       fp16: bool = False,
                       fp16_opt_level: str = 'O1',
                       local_rank: int = -1
                       ):
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if os.listdir(output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    output_path))
        #the loss function to be used in consistency loss
        if args_in.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif args_in.consistency_type == 'kl':
            consistency_criterion = losses.softmax_kl_loss

        #code from https://github.com/UKPLab/sentence-transformers
        dataloaders = [dataloader for dataloader, _ in train_objectives]
        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        device = self.device
        self.best_score = -9999999
        for loss_model in loss_models:
            loss_model.to(device)
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * args_in.num_epochs)
        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = num_train_steps
            if local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                t_total=t_total)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

            if fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                for train_idx in range(len(loss_models)):
                    model, optimizer = amp.initialize(loss_models[train_idx], optimizers[train_idx],
                                                      opt_level=fp16_opt_level)
                    loss_models[train_idx] = model
                    optimizers[train_idx] = optimizer

        ###########################end of #code from https://github.com/UKPLab/sentence-transformers

        train_state_in = self.make_train_state(args_in)

        # empty out the predictions files once before all epochs . writing of predictions to disk will happen at early stopping
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        predictions_teacher_dev_file_full_path = (args_in.predictions_teacher_dev_file+ '_' + sha + '.jsonl')


        with open(predictions_teacher_dev_file_full_path, 'w') as outfile:
            outfile.write("")
        with open(args_in.predictions_student_dev_file, 'w') as outfile:
            outfile.write("")
        with open(args_in.predictions_teacher_test_file, 'w') as outfile:
            outfile.write("")
        with open(args_in.predictions_student_test_file, 'w') as outfile:
            outfile.write("")


        try:


            global_step = 0
            data_iterators = [iter(dataloader) for dataloader in dataloaders]

            num_train_objectives = len(train_objectives)

            for epoch in trange(epochs, desc="Epoch"):
                training_steps = 0

                for loss_model in loss_models:
                    loss_model.zero_grad()
                    loss_model.train()


                #for each batch
                for _ in trange(steps_per_epoch, desc="training_batches", smoothing=0.05):

                    classifier_teacher_lex=  loss_models[0]
                    classifier_student_delex = loss_models[1]
                    classifier_teacher_lex_ema = loss_models[2]
                    classifier_student_delex_ema = loss_models[3]

                    optimizer_teacher_lex = optimizers[0]
                    optimizer_student_delex = optimizers[1]
                    optimizer_teacher_lex_ema = optimizers[2]
                    optimizer_student_delex_ema = optimizers[3]

                    scheduler_teacher_lex = schedulers[0]
                    scheduler_student_delex = schedulers[1]
                    scheduler_teacher_lex_ema = schedulers[2]
                    scheduler_student_delex_ema = schedulers[3]


                    #teacher models will have same iterator + all student models will have same iterator
                    data_iterator_teachers_lex = data_iterators[0]
                    data_iterator_students_delex = data_iterators[1]





                    try:
                        data_teachers_lex = next(data_iterator_teachers_lex)
                        data_teachers_delex = next(data_iterator_students_delex)
                    except StopIteration:
                        logging.info("error in data_iterator")
                        # data_iterator = iter(dataloaders[train_idx])
                        # data_iterators[train_idx] = data_iterator
                        # data = next(data_iterator)


                    assert data_teachers_lex is not None
                    assert data_teachers_delex is not None
                    #predictions using all teacher models
                    features, labels = batch_to_device(data_teachers_lex, self.device)
                    predictions_lex_teacher = classifier_teacher_lex(features, labels)
                    predictions_lex_teacher_ema = classifier_teacher_lex_ema(features, labels)
                    class_loss_lex_teacher = class_loss_func(predictions_lex_teacher, labels)
                    class_loss_lex_teacher_ema = class_loss_func(predictions_lex_teacher_ema, labels)

                    # predictions using all student models
                    features, labels = batch_to_device(data_teachers_delex, self.device)
                    predictions_delex_student = classifier_student_delex(features, labels)
                    predictions_delex_student_ema = classifier_student_delex_ema(features, labels)
                    class_loss_delex_student = class_loss_func(predictions_delex_student, labels)
                    class_loss_delex_student_ema = class_loss_func(predictions_delex_student_ema, labels)


                    combined_class_loss =  class_loss_lex_teacher+class_loss_lex_teacher_ema+class_loss_delex_student+class_loss_delex_student_ema


                    consistency_loss_delex_student_lex_teacher = consistency_criterion(predictions_delex_student,predictions_lex_teacher )
                    consistency_loss_delex_student_lex_teacher_ema = consistency_criterion(predictions_delex_student,predictions_lex_teacher_ema)
                    consistency_loss_delex_student_delex_student_ema = consistency_criterion(predictions_delex_student, predictions_delex_student_ema)
                    combined_consistency_loss=(0.5)*consistency_loss_delex_student_lex_teacher+\
                                              (0.5)*consistency_loss_delex_student_lex_teacher_ema+\
                                              (6)*consistency_loss_delex_student_delex_student_ema



                    # combined loss is the sum of  classification losses and  consistency losses
                    combined_loss = (args_in.consistency_weight * combined_consistency_loss) + (combined_class_loss)
                    if fp16:
                        with amp.scale_loss(combined_loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                    else:
                        combined_loss.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                    #todo: will have to combine the optimizers and do one single step

                    optimizer_teacher_lex.step()
                    optimizer_student_delex.step()
                    optimizer_teacher_lex_ema.step()
                    optimizer_student_delex_ema.step()

                    scheduler_teacher_lex.step()
                    scheduler_student_delex.step()
                    scheduler_teacher_lex_ema.step()
                    scheduler_student_delex_ema.step()

                    optimizer_teacher_lex.zero_grad()
                    optimizer_student_delex.zero_grad()
                    optimizer_teacher_lex_ema.zero_grad()
                    optimizer_student_delex_ema.zero_grad()



                    training_steps += 1
                    global_step += 1

                    #  in ema mode, one model is the exponential moving average of the other. that calculation is done here
                    # eg: classifier_student_delex_ema is the ema of classifier_student_delex
                    self.update_ema_variables(classifier_student_delex, classifier_student_delex_ema, args_in.ema_decay,
                                              global_step)
                    self.update_ema_variables(classifier_teacher_lex, classifier_teacher_lex_ema, args_in.ema_decay,
                                              global_step)

                # for printing training accuracy etc
                # if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                #     self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                #     for loss_model in loss_models:
                #         loss_model.zero_grad()
                #         loss_model.train()

                #run evaluation on dev at the end of every epoch, not batch
                for evaluator in evaluators:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)





        except KeyboardInterrupt:
            print("Exiting loop")


    def make_train_state(self,args):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate': args.learning_rate,
                'epoch_index': 0,
                'train_loss_teacher_lex': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1,
                'model_filename': args.model_state_file}

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)