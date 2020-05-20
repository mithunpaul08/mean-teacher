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

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
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
    def train_1teacher(self, args_in, dataset, comet_value_updater, vectorizer,
                       train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
                       epochs: int = 1,
                       steps_per_epoch=None,
                       scheduler: str = 'WarmupLinear',
                       warmup_steps: int = 10000,
                       optimizer_class: Type[Optimizer] = transformers.AdamW,
                       optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
                       weight_decay: float = 0.01,
                       fp16: bool = False,
                       fp16_opt_level: str = 'O1',
                       local_rank: int = -1
                       ):

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

                for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                    for train_idx in range(num_train_objectives):
                        loss_model = loss_models[train_idx]
                        optimizer = optimizers[train_idx]
                        scheduler = schedulers[train_idx]
                        data_iterator = data_iterators[train_idx]

                        try:
                            data = next(data_iterator)
                        except StopIteration:
                            # logging.info("Restart data_iterator")
                            data_iterator = iter(dataloaders[train_idx])
                            data_iterators[train_idx] = data_iterator
                            data = next(data_iterator)

                        features, labels = batch_to_device(data, self.device)
                        predictions = loss_model(features, labels)

                        # compute the classification loss of teacher running over lexicalized data class_loss_lex
                        # note: we are not adding the classification of the two ema models becuase there is no backpropagation in them
                        class_loss_lex = class_loss_func(predictions, labels)

                        loss_t_lex = class_loss_lex.item()

            # Iterate over training dataset
            for epoch_index in range(args_in.num_epochs):
                train_state_in['epoch_index'] = epoch_index
                dataset.set_split('train_lex')
                dataset_lex= copy.deepcopy(dataset)



                batch_generator_lex_data=None
                #WHEN use_semi_supervised is turned on, only part of the gold LABELS will be given to the classifier. Rest all will be masked.
                if(args_in.use_semi_supervised==True):
                    assert args_in.percentage_labels_for_semi_supervised > 0
                    batch_generator_lex_data = generate_batches_for_semi_supervised(dataset_lex, args_in.percentage_labels_for_semi_supervised, workers=args_in.workers, batch_size=args_in.batch_size,
                                                        device=args_in.device,mask_value=args_in.NO_LABEL )
                else:
                    batch_generator_lex_data = generate_batches_without_sampler(dataset_lex, workers=args_in.workers, batch_size=args_in.batch_size,device=args_in.device,shuffle=args_in.shuffle_data)


                no_of_batches_lex = int(len(dataset)/args_in.batch_size)

                assert batch_generator_lex_data is not None
                batch_generator_delex_data = None


                #do batch generation for delex data
                dataset.set_split('train_delex')
                dataset_delex = copy.deepcopy(dataset)
                if (args_in.use_semi_supervised == True):
                    assert args_in.percentage_labels_for_semi_supervised > 0
                    batch_generator_delex_data = generate_batches_for_semi_supervised(dataset_delex,
                                                                            args_in.percentage_labels_for_semi_supervised,
                                                                            workers=args_in.workers,
                                                                            batch_size=args_in.batch_size,
                                                                            device=args_in.device,mask_value=args_in.NO_LABEL  )

                else:
                    batch_generator_delex_data = generate_batches_without_sampler(dataset_delex, workers=args_in.workers, batch_size=args_in.batch_size,
                                                        device=args_in.device,shuffle=args_in.shuffle_data)

                assert batch_generator_delex_data is not None

                no_of_batches_delex = int(len(dataset) / args_in.batch_size)

                running_consistency_loss = 0.0


                running_loss_lex = 0.0
                running_acc_lex = 0.0
                running_acc_lex_ema = 0.0
                running_loss_delex = 0.0
                running_acc_delex = 0.0
                running_acc_delex_ema = 0.0
                classifier_teacher_lex.train_multiple_teachers_1student()
                classifier_student_delex.train_multiple_teachers_1student()



                total_right_predictions_teacher_lex=0
                total_right_predictions_teacher_lex_ema = 0
                total_right_predictions_student_delex = 0
                total_right_predictions_student_delex_ema = 0
                total_gold_label_count=0


                combined_data_generators = zip(batch_generator_lex_data, batch_generator_delex_data)

                assert combined_data_generators is not None

                for batch_index, (batch_dict_lex,batch_dict_delex) in enumerate(tqdm(combined_data_generators,desc="training_batches",total=no_of_batches_delex)):


                    # --------------------------------------
                    # step 1. zero the gradients
                    input_optimizer.zero_grad()
                    inter_atten_optimizer.zero_grad()



                    #initializing initial state of the optimizer to start from 0. This should be learned/tuned hyper parameter.
                    # remove if not having any effect/improvement
                    if epoch_index == 0 and args_in.optimizer == 'adagrad':
                        update_optimizer_state(input_optimizer, inter_atten_optimizer, args_in)




                    # step 2. compute the output


                    y_pred_lex = classifier_teacher_lex(batch_dict_lex['x_claim'], batch_dict_lex['x_evidence'])
                    y_pred_lex_ema = classifier_teacher_lex_ema(batch_dict_lex['x_claim'], batch_dict_lex['x_evidence'])





                    assert y_pred_lex is not None
                    assert len(y_pred_lex) > 0


                    total_gold_label_count=total_gold_label_count+len(batch_dict_lex['y_target'])


                    # compute the classification loss of teacher running over lexicalized data class_loss_lex
                    #note: we are not adding the classification of the two ema models becuase there is no backpropagation in them
                    class_loss_lex = class_loss_func(y_pred_lex, batch_dict_lex['y_target'])

                    loss_t_lex = class_loss_lex.item()
                    running_loss_lex += (loss_t_lex - running_loss_lex) / (batch_index + 1)
                    self._LOG.debug(f"loss_t_lex={loss_t_lex}\trunning_loss_lex={running_loss_lex}")


                    combined_class_loss = class_loss_lex

                    consistency_loss=0
                    class_loss_delex=None



                    #all student classifier related prediction code (the one which feeds off delexicalized data).
                    y_pred_delex = classifier_student_delex(batch_dict_delex['x_claim'], batch_dict_delex['x_evidence'])
                    class_loss_delex = class_loss_func(y_pred_delex, batch_dict_delex['y_target'])
                    loss_t_delex = class_loss_delex.item()
                    running_loss_delex += (loss_t_delex - running_loss_delex) / (batch_index + 1)

                    y_pred_delex_ema = classifier_student_delex_ema(batch_dict_delex['x_claim'],
                                                                  batch_dict_delex['x_evidence'])

                    consistency_loss_delexstudent_lexteacher = consistency_criterion(y_pred_lex, y_pred_delex)
                    consistency_loss_delexstudent_lexTeacherEma = consistency_criterion(y_pred_lex_ema, y_pred_delex)
                    consistency_loss_delexstudent_delexStudentEma = consistency_criterion(y_pred_delex_ema, y_pred_delex)

                    consistency_loss=(5)*consistency_loss_delexstudent_lexteacher+\
                                       (0.5)*consistency_loss_delexstudent_lexTeacherEma+\
                                         (0.5)*consistency_loss_delexstudent_delexStudentEma
                    consistency_loss_value = consistency_loss.item()
                    running_consistency_loss += (consistency_loss_value - running_consistency_loss) / (batch_index + 1)

                    #since there is no backpropagation on ema model, dont add its classification loss. Even if you add
                    # it wont back propagate since the model is defined that way.
                    combined_class_loss = class_loss_delex + class_loss_lex


                    #combined loss is the sum of  classification losses and  consistency losses
                    combined_loss = (args_in.consistency_weight * consistency_loss) + (combined_class_loss)
                    combined_loss.backward()







                    # step 5. use optimizer to take gradient step
                    #optimizer.step()
                    input_optimizer.step()
                    inter_atten_optimizer.step()
                    global_variables.global_step += 1

                    #  in ema mode, one model is the exponential moving average of the other. that calculation is done here
                    #eg: classifier_student_delex_ema is the ema of classifier_student_delex
                    self.update_ema_variables(classifier_student_delex, classifier_student_delex_ema, args_in.ema_decay, global_variables.global_step)
                    self.update_ema_variables(classifier_teacher_lex, classifier_teacher_lex_ema, args_in.ema_decay,
                                              global_variables.global_step)



                    # -----------------------------------------

                    # compute the accuracy for teacher-lex
                    y_pred_labels_lex_sf = F.softmax(y_pred_lex, dim=1)
                    count_of_right_predictions_teacher_lex_per_batch, acc_t_lex, teacher_predictions_by_label_class = self.compute_accuracy(
                        y_pred_labels_lex_sf, batch_dict_lex['y_target'])
                    total_right_predictions_teacher_lex = total_right_predictions_teacher_lex + count_of_right_predictions_teacher_lex_per_batch
                    running_acc_lex += (acc_t_lex - running_acc_lex) / (batch_index + 1)

                    # compute the accuracy for teacher-lex-ema
                    y_pred_labels_lex_ema_sf = F.softmax(y_pred_lex_ema, dim=1)
                    count_of_right_predictions_teacher_lex_ema_per_batch, acc_t_lex_ema, teacher_ema_predictions_by_label_class = self.compute_accuracy(
                        y_pred_labels_lex_ema_sf, batch_dict_lex['y_target'])
                    total_right_predictions_teacher_lex_ema = total_right_predictions_teacher_lex_ema + count_of_right_predictions_teacher_lex_ema_per_batch
                    running_acc_lex_ema += (acc_t_lex_ema - running_acc_lex_ema) / (batch_index + 1)


                    # compute the accuracy for student-delex
                    y_pred_labels_delex_sf = F.softmax(y_pred_delex, dim=1)
                    count_of_right_predictions_student_delex_per_batch, acc_t_delex, student_predictions_by_label_class = self.compute_accuracy(
                        y_pred_labels_delex_sf,batch_dict_delex['y_target'])
                    total_right_predictions_student_delex = total_right_predictions_student_delex + count_of_right_predictions_student_delex_per_batch
                    running_acc_delex += (acc_t_delex - running_acc_delex) / (batch_index + 1)
                    #comet_value_updater.log_confusion_matrix(batch_dict_delex['y_target'],student_predictions_by_label_class)


                    # compute the accuracy for student-delex-ema
                    y_pred_labels_delex_ema_sf = F.softmax(y_pred_delex_ema, dim=1)
                    count_of_right_predictions_student_delex_ema_per_batch, acc_t_delex_ema, student_ema_predictions_by_label_class = self.compute_accuracy(
                        y_pred_labels_delex_ema_sf, batch_dict_delex['y_target'])
                    total_right_predictions_student_delex_ema = total_right_predictions_student_delex_ema + count_of_right_predictions_student_delex_ema_per_batch
                    running_acc_delex_ema += (acc_t_delex_ema - running_acc_delex_ema) / (batch_index + 1)






                    self._LOG.debug(
                        f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                        f"classification_loss_lex:{round(running_loss_lex,2)}\t classification_loss_delex:{round(running_loss_delex,2)} "
                        f"\t consistencyloss:{round(running_consistency_loss,6)}"
                        f" \t running_acc_lex:{round(running_acc_lex,4) }  \t running_acc_delex:{round(running_acc_delex,4)}   ")


                    total_right_predictions_teacher_lex=total_right_predictions_teacher_lex+count_of_right_predictions_teacher_lex_per_batch
                    running_acc_lex += (acc_t_lex - running_acc_lex) / (batch_index + 1)



                    #Accuracy calculation for student model

                    total_right_predictions_student_delex=total_right_predictions_student_delex+count_of_right_predictions_student_delex_per_batch
                    running_acc_delex += (acc_t_delex - running_acc_delex) / (batch_index + 1)
                    self._LOG.debug(
                        f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                        f"classification_loss_lex:{round(running_loss_lex,2)}\t classification_loss_delex:{round(running_loss_delex,2)} "
                        f"\t consistencyloss:{round(running_consistency_loss,6)}"
                        f" \t running_acc_lex:{round(running_acc_lex,4) }  \t running_acc_delex:{round(running_acc_delex,4)}   ")


                    assert len(teacher_predictions_by_label_class)>0
                    assert len(batch_dict_lex['y_target']) > 0


                    assert len(student_predictions_by_label_class) > 0



                    comet_value_updater.log_metric(
                            "training accuracy of lex teacher  across batches",
                            running_acc_lex,
                            step=batch_index)




                    teacher_lex_same_as_gold, \
                    student_delex_same_as_gold,\
                    student_teacher_match, \
                    student_teacher_match_but_not_same_as_gold, \
                    student_teacher_match_and_same_as_gold, \
                    student_delex_same_as_gold_but_teacher_is_different, \
                    teacher_lex_same_as_gold_but_student_is_different   =   self.calculate_label_overlap_between_teacher_and_student_predictions(teacher_predictions_by_label_class,student_predictions_by_label_class,batch_dict_lex['y_target'])


                    teacher_lex_same_as_gold_percent = self.calculate_percentage(teacher_lex_same_as_gold, args_in.batch_size)
                    student_delex_same_as_gold_percent = self.calculate_percentage(student_delex_same_as_gold, args_in.batch_size)
                    student_teacher_match_percent = self.calculate_percentage(student_teacher_match, args_in.batch_size)
                    student_teacher_match_but_not_same_as_gold_percent = self.calculate_percentage(
                        student_teacher_match_but_not_same_as_gold, args_in.batch_size)
                    student_teacher_match_and_same_as_gold_percent = self.calculate_percentage(
                        student_teacher_match_and_same_as_gold, args_in.batch_size)
                    student_delex_same_as_gold_but_teacher_is_different_percent = self.calculate_percentage(
                        student_delex_same_as_gold_but_teacher_is_different, args_in.batch_size)
                    teacher_lex_same_as_gold_but_student_is_different_percent = self.calculate_percentage(teacher_lex_same_as_gold_but_student_is_different, args_in.batch_size)

                    if (comet_value_updater is not None):

                        comet_value_updater.log_metric("student_delex_same_as_gold_but_teacher_is_different_percent  per batch",
                                                       student_delex_same_as_gold_but_teacher_is_different_percent,
                                                       step=batch_index)
                        comet_value_updater.log_metric("teacher_lex_same_as_gold_but_student_is_different_percent  per batch",
                                                       teacher_lex_same_as_gold_but_student_is_different_percent,
                                                       step=batch_index)


                    if (comet_value_updater is not None):
                            comet_value_updater.log_metric(
                                "teacher training accuracy  per batch",
                                running_acc_lex,
                                step=batch_index)


                self._LOG.info(
                    f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                    f"classification_loss_lex:{round(running_loss_lex,2)}\t classification_loss_delex:{round(running_loss_delex,2)} "
                    f"\t consistencyloss:{round(running_consistency_loss,6)}"
                    f" \t running_acc_lex:{round(running_acc_lex,4) }  \t running_acc_delex:{round(running_acc_delex,4)}  ")


                train_state_in['train_acc'].append(running_acc_lex)
                train_state_in['train_loss'].append(running_loss_lex)



                self.number_of_datapoints = total_gold_label_count
                accuracy_teacher_model_by_per_batch_prediction = self.calculate_percentage(total_right_predictions_teacher_lex,self.number_of_datapoints)
                if (args_in.add_student == True):
                    accuracy_student_model_by_per_batch_prediction = self.calculate_percentage(
                    total_right_predictions_student_delex, self.number_of_datapoints)



                self._LOG.info(
                    f"running_acc_lex training by old method at the end of {epoch_index}:{running_acc_lex}")
                self._LOG.info(
                    f"accuracy_teacher_model_by_per_batch_prediction at the end of epoch{epoch_index}:{accuracy_teacher_model_by_per_batch_prediction}")
                if (args_in.add_student == True):
                    self._LOG.info(
                    f"acc_t_delex by old method {epoch_index}:{running_acc_delex}")

                    self._LOG.info(
                    f"accuracy_student_model_by_per_batch_prediction method at the end of epoch{epoch_index}:{ accuracy_student_model_by_per_batch_prediction}")




                self._LOG.info(
                    f"epoch:{epoch_index}")

                if (args_in.add_student == True):
                    self._LOG.debug(f" teacher_lex_same_as_gold_percent:{teacher_lex_same_as_gold_percent}")
                    self._LOG.debug(f" student_delex_same_as_gold_percent:{student_delex_same_as_gold_percent}")
                    self._LOG.debug(f" student_teacher_match_percent:{student_teacher_match_percent}")
                    self._LOG.debug(f" student_teacher_match_but_not_same_as_gold_percent:{student_teacher_match_but_not_same_as_gold_percent}")
                    self._LOG.debug(f" student_teacher_match_and_same_as_gold_percent:{student_teacher_match_and_same_as_gold_percent}")
                    self._LOG.debug(f" student_delex_same_as_gold_but_teacher_is_different_percent:{student_delex_same_as_gold_but_teacher_is_different_percent}")
                    self._LOG.debug(f" teacher_lex_same_as_gold_but_student_is_different_percent:{teacher_lex_same_as_gold_but_student_is_different_percent}")


                if (comet_value_updater is not None):
                    comet_value_updater.log_metric("training accuracy of teacher model per epoch", running_acc_lex,
                                                   step=epoch_index)
                    comet_value_updater.log_metric("training accuracy of teacher ema model per epoch", running_acc_lex_ema,
                                                   step=epoch_index)

                    comet_value_updater.log_metric("training accuracy of student model per epoch", running_acc_delex,
                                                   step=epoch_index)

                    comet_value_updater.log_metric("training accuracy of student ema model per epoch", running_acc_delex_ema,
                                                   step=epoch_index)


                    comet_value_updater.log_metric(
                        "teacher_lex_same_as_gold_but_student_is_different_percent per global step",
                        teacher_lex_same_as_gold_but_student_is_different_percent,
                        step=global_variables.global_step)
                    comet_value_updater.log_metric("consistency_loss per epoch",
                                                   running_consistency_loss,
                                                   step=epoch_index)

                    comet_value_updater.log_metric("training accuracy of student model per epoch", running_acc_delex,
                                                   step=epoch_index)
                    comet_value_updater.log_metric("training accuracy of teacher model per epoch", running_acc_lex,
                                                   step=epoch_index)


                # Iterate over val dataset and check on dev using the intended trained model, which usually is the student delex model
                dataset.set_split('val_delex')
                classifier_student_delex.eval()
                predictions_by_student_model_on_dev=[]
                predictions_by_student_ema_model_on_dev = []
                predictions_by_teacher_model_on_dev = []
                predictions_by_teacher_ema_model_on_dev = []




                # Evaluate on dev patition using the intended trained model

                #evaluate using student-delex

                dataset.set_split('val_delex')
                classifier_student_delex.eval()
                running_acc_val_student, running_loss_val_student, microf1_student_dev_without_unrelated_class, \
                microf1_student_dev_with_only_unrelated_class, fnc_score_student_dev = self.eval(
                    classifier_student_delex, args_in, dataset, epoch_index, vectorizer,
                    predictions_by_student_model_on_dev, "student_delex_on_dev")

                # evaluate using student-delex-ema
                dataset.set_split('val_delex')
                classifier_student_delex_ema.eval()

                running_acc_val_student_ema, running_loss_val_student_ema, \
                microf1_student_dev_ema_without_unrelated_class, microf1_student_dev_ema_with_only_unrelated_class, fnc_score_student_dev_ema \
                    = self.eval(
                    classifier_student_delex_ema, args_in, dataset, epoch_index, vectorizer,
                    predictions_by_student_ema_model_on_dev, "student_delex_ema_on_dev")




                # evaluate using teacher-lex
                dataset.set_split('val_lex')
                classifier_teacher_lex.eval()
                running_acc_val_teacher, running_loss_val_teacher,\
                microf1_teacher_dev_without_unrelated_class, microf1_teacher_dev_with_only_unrelated_class, \
                fnc_score_teacher_dev =self.eval(classifier_teacher_lex, args_in, dataset,
                                                                              epoch_index, vectorizer,
                                                 predictions_by_teacher_model_on_dev, "teacher_lex_on_dev")

                # evaluate using teacher-lex-ema
                dataset.set_split('val_lex')
                classifier_teacher_lex_ema.eval()
                running_acc_val_teacher_ema, running_loss_val_teacher_ema ,\
                microf1_teacher_dev_ema_without_unrelated_class, microf1_teacher_dev_ema_with_only_unrelated_class, fnc_score_teacher_dev_ema\
                    = self.eval(classifier_teacher_lex_ema,
                                                                                      args_in, dataset,
                                                                                      epoch_index, vectorizer,
                                predictions_by_teacher_ema_model_on_dev, "teacher_lex_ema_on_dev")



                # Do early stopping based on when the dev accuracy drops from its best for patience (as defined in initializer.py)
                train_state_in['val_loss'].append(running_loss_val_student)
                train_state_in['val_acc'].append(running_acc_val_student)
                train_state_in = self.update_train_state(args=args_in,
                                                         models=[classifier_student_delex, classifier_teacher_lex],
                                                         train_state=train_state_in)

                assert comet_value_updater is not None

                comet_value_updater.log_metric("acc_dev_per_epoch_using_student_model", running_acc_val_student, step=epoch_index)
                comet_value_updater.log_metric("acc_dev_per_epoch_using_teacher_model", running_acc_val_teacher, step=epoch_index)
                comet_value_updater.log_metric("acc_dev_per_epoch_using_student_ema_model", running_acc_val_student_ema,
                                               step=epoch_index)
                comet_value_updater.log_metric("acc_dev_per_epoch_using_teacher_ema_model", running_acc_val_teacher_ema,
                                               step=epoch_index)

                # also test it on a third dataset which is usually cross domain on fnc
                args_in.database_to_test_with="fnc"


                dataset.set_split('test_delex')
                predictions_by_student_model_on_test_partition = []
                predictions_by_student_ema_model_on_test_partition = []
                predictions_by_teacher_model_on_test_partition = []
                predictions_by_teacher_ema_model_on_test_partition = []

                classifier_student_delex.eval()
                self._LOG.info("classifier_student_delex model on test_delex")
                running_acc_test_student, running_loss_test_student,microf1_student_test_without_unrelated_class,\
                microf1_student_test_with_only_unrelated_class, fnc_score_student_test= self.eval(classifier_student_delex, args_in,
                dataset, epoch_index,vectorizer
                ,predictions_by_student_model_on_test_partition,"student_delex_on_test")

                dataset.set_split('test_delex')
                self._LOG.info("classifier_student_delex_ema model on test_delex")
                classifier_student_delex_ema.eval()
                running_acc_test_student_ema, running_loss_test_student_ema, microf1_student_ema_test_without_unrelated_class, \
                microf1_student_ema_test_with_only_unrelated_class, fnc_score_student_ema_test = self.eval(
                    classifier_student_delex_ema, args_in,
                    dataset, epoch_index, vectorizer, predictions_by_student_ema_model_on_test_partition,
                    "student_delex_ema_on_test")

                dataset.set_split('test_lex')
                classifier_teacher_lex.eval()
                self._LOG.info("classifier_teacher_lex model on test_lex")
                running_acc_test_teacher, running_loss_test_teacher,microf1_teacher_test_without_unrelated_class, \
                microf1_teacher_test_with_only_unrelated_class, fnc_score_teacher_test = self.eval(classifier_teacher_lex, args_in,
                                                                                dataset, epoch_index, vectorizer,
                                                                                                   predictions_by_teacher_model_on_test_partition,"teacher_lex_on_test")

                dataset.set_split('test_lex')
                self._LOG.info("microf1_teacher_test_ema_without_unrelated_class model on test_lex")
                running_acc_test_teacher_ema, running_loss_test_teacher_ema, microf1_teacher_test_ema_without_unrelated_class, \
                microf1_teacher_test_ema_with_only_unrelated_class, fnc_score_teacher_test_ema = self.eval(
                    classifier_teacher_lex_ema, args_in,
                    dataset, epoch_index, vectorizer
                ,predictions_by_teacher_ema_model_on_test_partition, "teacher_lex_ema_on_test")

                comet_value_updater.log_metric("plain_acc_test_student", running_acc_test_student,step=epoch_index)
                comet_value_updater.log_metric("microf1_student_test_without_unrelated_class", microf1_student_test_without_unrelated_class,step=epoch_index)
                comet_value_updater.log_metric("microf1_student_test_with_only_unrelated_class", microf1_student_test_with_only_unrelated_class, step=epoch_index)
                comet_value_updater.log_metric("fnc_score_student_on_test_partition", fnc_score_student_test,step=epoch_index)

                comet_value_updater.log_metric("plain_acc_test_student_ema", running_acc_test_student_ema, step=epoch_index)
                comet_value_updater.log_metric("microf1_student_test_ema_without_unrelated_class",microf1_student_ema_test_without_unrelated_class, step=epoch_index)
                comet_value_updater.log_metric("microf1_student_test_ema_with_only_unrelated_class",microf1_student_ema_test_with_only_unrelated_class, step=epoch_index)
                comet_value_updater.log_metric("fnc_score_student_ema_on_test_partition", fnc_score_student_ema_test,step=epoch_index)


                comet_value_updater.log_metric("plain_acc_test_teacher", running_acc_test_teacher,step=epoch_index)
                comet_value_updater.log_metric("fnc_score_teacher_on_test_partition", fnc_score_teacher_test,step=epoch_index)
                comet_value_updater.log_metric("microf1_teacher_test_with_only_unrelated_class", microf1_teacher_test_with_only_unrelated_class,step=epoch_index)
                comet_value_updater.log_metric("microf1_teacher_test_without_unrelated_class", microf1_teacher_test_without_unrelated_class,step=epoch_index)


                comet_value_updater.log_metric("plain_acc_test_teacher_ema", running_acc_test_teacher_ema, step=epoch_index)
                comet_value_updater.log_metric("fnc_score_teacher_on_test_partition_ema", fnc_score_teacher_test_ema, step=epoch_index)
                comet_value_updater.log_metric("microf1_teacher_test_ema_with_only_unrelated_class",microf1_teacher_test_ema_with_only_unrelated_class , step=epoch_index)
                comet_value_updater.log_metric("microf1_teacher_test_ema_without_unrelated_class", microf1_teacher_test_ema_without_unrelated_class , step=epoch_index)



                #resetting args_in.database_to_test_with to make sure the values don't persist across epochs
                args_in.database_to_test_with = "dummy"
                dataset.set_split('val_lex')

                if train_state_in['stop_early']:
                    ## whenever you hit early stopping just store all the data and predictions at that point to disk for debug purposes

                    with open(predictions_teacher_dev_file_full_path, 'w') as outfile:
                        outfile.write("")
                    with open(args_in.predictions_student_dev_file, 'w') as outfile:
                        outfile.write("")
                    with open(args_in.predictions_teacher_test_file, 'w') as outfile:
                        outfile.write("")
                    with open(args_in.predictions_student_test_file, 'w') as outfile:
                        outfile.write("")

                    assert len(predictions_by_student_model_on_dev) > 0
                    assert len(predictions_by_teacher_model_on_dev) > 0
                    assert len(predictions_by_student_model_on_test_partition) > 0
                    assert len(predictions_by_teacher_model_on_test_partition) > 0

                    self.write_dict_as_json(args_in.predictions_student_dev_file, predictions_by_student_model_on_dev)
                    self.write_dict_as_json(predictions_teacher_dev_file_full_path, predictions_by_teacher_model_on_dev)
                    self.write_dict_as_json(args_in.predictions_student_test_file, predictions_by_student_model_on_test_partition)
                    self.write_dict_as_json(args_in.predictions_teacher_test_file, predictions_by_teacher_model_on_test_partition)

                    break

                if (args_in.add_student == True):
                    self._LOG.info(
                    f" accuracy on dev partition by student:{round(running_acc_val_student,2)} ")
                    self._LOG.info(
                        f" accuracy on test partition by student:{round(running_acc_test_student,2)} ")

                    self._LOG.info(
                    f" accuracy on dev partition by teacher:{round(running_acc_val_teacher,2)} ")

                self._LOG.info(
                    f" accuracy on test partition by teacher:{round(running_acc_test_teacher,2)} ")


                self._LOG.info(
                    f"****************end of epoch {epoch_index}*********************")
            print("****************end of all epochs*********************")
            self._LOG.info(
            f"****************end of all epochs*********************")

        except KeyboardInterrupt:
            print("Exiting loop")


    def make_train_state(self,args):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate': args.learning_rate,
                'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1,
                'model_filename': args.model_state_file}