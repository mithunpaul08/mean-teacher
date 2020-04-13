from mean_teacher.modules.rao_datasets import RTEDataset
from mean_teacher.utils.utils_rao import generate_batches,initialize_optimizers,update_optimizer_state,generate_batches_for_semi_supervised,generate_batches_without_sampler,make_embedding_matrix
from mean_teacher.utils import losses
import time
import torch
import torch.nn as nn
from tqdm import tqdm,tqdm_notebook
from torch.nn import functional as F
import os
from mean_teacher.utils import global_variables
from torch.utils.data import DataLoader
import copy
from mean_teacher.scorers.fnc_scorer import report_score
from sklearn import metrics
import git
from mean_teacher.modules.vectorizer_with_embedding import LABELS
import json

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

NO_LABEL=-1

if torch.cuda.is_available():
    class_loss_func = nn.CrossEntropyLoss(ignore_index=NO_LABEL).cuda()
else:
    class_loss_func = nn.CrossEntropyLoss(ignore_index=NO_LABEL).cpu()

class Trainer():
    def __init__(self,LOG):
        self._LOG=LOG
        self._current_time={time.strftime("%c")}
        self.number_of_datapoints=0

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

    def update_train_state(self, args, models, train_state):
        """Handle the training state updates.
        Components:
         - Early Stopping: Prevent overfitting.
         - Model Checkpoint: Model is saved if the model is better
        :param args: main arguments
        :param model: model to train
        :param train_state: a dictionary representing the training state values
        :returns:
            a new train_state
        """

        # Save one model at least
        if train_state['epoch_index'] == 0:
            for index,model in enumerate(models):
                model_type = "student"
                if index>0:
                    model_type = "teacher"
                torch.save(model.state_dict(), train_state['model_filename'] + model_type + "_e" + str(train_state['epoch_index']) + "_"+ sha+".pth")
            train_state['stop_early'] = False
            assert type(train_state['val_acc']) is list
            all_val_acc_length = len(train_state['val_acc'])
            assert all_val_acc_length > 0
            acc_current_epoch = train_state['val_acc'][all_val_acc_length - 1]
            train_state['early_stopping_best_val'] = acc_current_epoch

        # Save model if performance improved
        elif train_state['epoch_index'] >= 1:
            loss_tm1, acc_current_epoch = train_state['val_acc'][-2:]

            # If accuracy decreased
            if acc_current_epoch <= train_state['early_stopping_best_val']:
                # increase patience counter
                train_state['early_stopping_step'] += 1
                self._LOG.info(f"found that acc_current_epoch  {acc_current_epoch} is less than or equal to the best dev "
                         f"accuracy value so far which is"
                         f" {train_state['early_stopping_best_val']}. "
                         f"Increasing patience total value. "
                         f"of patience now is {train_state['early_stopping_step']}")
            # accuracy increased
            else:
                # Save the best model from the list of models passed to this. which is usually in the order, [student,teacher]
                for index, model in enumerate(models):
                    model_type = "student"
                    if index > 0:
                        model_type = "teacher"
                    torch.save(model.state_dict(), train_state['model_filename']+"_best_"+model_type + "_"+ sha+ ".pth")
                self._LOG.info(
                    f"found that acc_current_epoch loss {acc_current_epoch} is more than the best accuracy so far which is "
                    f"{train_state['early_stopping_best_val']}.resetting patience=0")
                # Reset early stopping step
                train_state['early_stopping_step'] = 0
                train_state['early_stopping_best_val'] = acc_current_epoch

            # Stop early ?
            train_state['stop_early'] = \
                train_state['early_stopping_step'] >= args.early_stopping_criteria

        return train_state

    def accuracy_fever(self,predicted_labels, gold_labels,no_of_batches_lex):
        m = nn.Softmax()
        output_sftmax = m(predicted_labels)
        labeled_minibatch_size = no_of_batches_lex
        _, pred = output_sftmax.topk(1, 1, True, True)

        # gold LABELS and predictions are in transposes (eg:1x15 vs 15x1). so take a transpose to correct it.
        pred_t = pred.t()
        correct = pred_t.eq(gold_labels.view(1, -1).expand_as(pred_t))

        # take sum because in correct_k all the LABELS that match are now denoted by 1. So the sum means, total number of correct answers
        correct_k = correct.sum(1)
        correct_k_float = float(correct_k.data.item())
        labeled_minibatch_size_f = float(labeled_minibatch_size)
        result2 = (correct_k_float / labeled_minibatch_size_f) * 100

        return result2

    def calculate_micro_f1(self,y_pred, y_target):
        assert len(y_pred) == len(y_target), "lengths are different {len(y_pred)}"
        labels_to_include =[]
        for index,l in enumerate(y_target):
            if not (l==3):
                labels_to_include.append(index)
        mf1=metrics.f1_score(y_target,y_pred, average='micro', labels=labels_to_include)
        return mf1

    def compute_accuracy(self,y_pred, y_target):
        assert len(y_pred)==len(y_target)
        _, y_pred_classes = y_pred.max(dim=1)

        n_correct = torch.eq(y_pred_classes, y_target).sum().item()
        accuracy=n_correct / len(y_target) * 100
        return n_correct,accuracy,y_pred_classes

    def get_learning_rate(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def calculate_argmax_list(self, logit):
        list_labels_pred = []
        for tensor in logit:
            values, indices = torch.max(tensor, 0)
            list_labels_pred.append(indices.data.item())
        return list_labels_pred

    def calculate_percentage(self, numerator,denominator):
        return (100 * numerator / denominator)

    def predict(self,dataset,args_in,classifier,vocab):
        batch_generator_total = generate_batches(dataset, batch_size=args_in.batch_size,
                                                 device=args_in.device,workers=0)

        import math
        no_of_batches=math.floor(len(dataset)/args_in.batch_size)
        predicted_labels=[]
        gold_labels=[]
        for batch_dict_lex in tqdm(batch_generator_total,desc="predicting on training",total=no_of_batches):
            y_pred_lex = classifier(batch_dict_lex['x_claim'], batch_dict_lex['x_evidence'])
            y_pred_labels_lex_sf = F.softmax(y_pred_lex, dim=1)
            _, y_pred_indices = y_pred_labels_lex_sf.max(dim=1)
            predicted_labels.append([vocab.lookup_index(y) for y in y_pred_indices.tolist()])
            gold=batch_dict_lex['y_target']
            gold_labels.append([vocab.lookup_index(y) for y in gold.tolist()])
        predicted_labels_flat_list = [item for sublist in predicted_labels for item in sublist]
        gold_labels_flat_list = [item for sublist in gold_labels for item in sublist]
        return predicted_labels_flat_list,gold_labels_flat_list

    def calculate_label_overlap_between_teacher_and_student_predictions(self,teacher_lex_predictions,student_delex_predictions,gold_labels):
        teacher_lex_same_as_gold = 0
        student_delex_same_as_gold = 0
        student_teacher_match = 0
        student_teacher_match_but_not_same_as_gold = 0
        student_teacher_match_and_same_as_gold = 0
        student_delex_same_as_gold_but_teacher_is_different = 0
        teacher_lex_same_as_gold_but_student_is_different = 0
        assert len(student_delex_predictions)== len(teacher_lex_predictions) == len(gold_labels)
        for student, teacher, gold in (zip(student_delex_predictions, teacher_lex_predictions, gold_labels)):
            if teacher == gold:
                teacher_lex_same_as_gold += 1
                if not student == teacher:
                    teacher_lex_same_as_gold_but_student_is_different += 1
            if student == gold:
                student_delex_same_as_gold += 1
                if not student == teacher:
                    student_delex_same_as_gold_but_teacher_is_different += 1

            if teacher == student:
                student_teacher_match += 1
                if not teacher == gold:
                    student_teacher_match_but_not_same_as_gold += 1
                else:
                    student_teacher_match_and_same_as_gold += 1

        return teacher_lex_same_as_gold , \
               student_delex_same_as_gold,\
        student_teacher_match ,\
        student_teacher_match_but_not_same_as_gold ,\
        student_teacher_match_and_same_as_gold ,\
        student_delex_same_as_gold_but_teacher_is_different,\
        teacher_lex_same_as_gold_but_student_is_different

    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def get_argmax(self,predicted_labels):
        m = nn.Softmax()
        output_sftmax = m(predicted_labels)
        _, pred = output_sftmax.topk(1, 1, True, True)
        prediction_transpose=pred.t()
        return prediction_transpose[0]

    def look_up_plain_text_datapoint_using_vectorizer(self, datapoint_indices,vectorizer):
        '''

        :param vectorizer:
        :param predictions_index_labels:
        :return:
        '''
        datapoint_list=[]
        for e in datapoint_indices:
            word=vectorizer.claim_ev_vocab.lookup_index(e)
            if word not in ["<BEGIN>","<END>","<MASK>"]:
                datapoint_list.append(word)
        datapoint_str=" ".join(datapoint_list)
        return datapoint_str

    def get_label_strings_given_vectorizer(self, vectorizer, predictions_index_labels):
        labels_str=[]
        for e in predictions_index_labels:
            labels_str.append(vectorizer.label_vocab.lookup_index(e.item()).lower())
        return labels_str

    def get_label_string_given_vectorizer(self, vectorizer, label):
        return    (vectorizer.label_vocab.lookup_index(label))


    def get_label_strings_given_list(self, labels_tensor):

        labels_str=[]
        for e in labels_tensor:
            labels_str.append(LABELS[e.item()].lower())
        return labels_str

    def load_model_and_eval(self,args_in,classifier,dataset,split_to_test,vectorizer):
        if (args_in.load_model_from_disk_and_test):
                assert os.path.exists(args_in.trained_model_path) is True
                assert os.path.isfile(args_in.trained_model_path) is True
                if os.path.getsize(args_in.trained_model_path) > 0:
                    classifier.load_state_dict(torch.load(args_in.trained_model_path,map_location=torch.device(args_in.device)))
        classifier.eval()
        dataset.set_split(split_to_test)
        batch_generator_val = generate_batches(dataset, workers=args_in.workers, batch_size=args_in.batch_size,
                                               device=args_in.device, shuffle=args_in.shuffle_data)
        running_loss_val = 0.
        running_acc_val = 0.
        total_predictions = []
        total_gold = []

        no_of_batches= int(len(dataset) / args_in.batch_size)
        for batch_index, batch_dict in enumerate(tqdm(batch_generator_val, desc="dev_batches", total=no_of_batches)):
            # compute the output
            y_pred_val = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])

            # step 3. compute the class_loss
            class_loss = class_loss_func(y_pred_val, batch_dict['y_target'])
            loss_t = class_loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            acc_t=0
            # fnc alone has a different kind of scoring. we are using the official scoring function. Note that the
            #command line argument 'database_to_test_with' is used only for deciding the scoring function. it has nothing
            # to do with which test file to load.
            if(args_in.database_to_test_with=="fnc"):
                        predictions_index_labels=self.get_argmax(y_pred_val.float())
                        predictions_str_labels=self.get_label_strings_given_vectorizer(vectorizer, predictions_index_labels)
                        gold_str=self.get_label_strings_given_list(batch_dict['y_target'])
                        for e in gold_str:
                            total_gold.append(e)
                        for e in predictions_str_labels:
                            total_predictions.append(e)
            else:
                # compute the accuracy
                y_pred_labels_val_sf = F.softmax(y_pred_val, dim=1)
                right_predictions, acc_t, predictions_by_label_class = self.compute_accuracy(y_pred_labels_val_sf,
                                                                                         batch_dict['y_target'])
            running_acc_val += (acc_t - running_acc_val) / (batch_index + 1)

        if (args_in.database_to_test_with == "fnc"):
            running_acc_val = report_score(total_gold, total_predictions)

        self._LOG.info(
            f" accuracy on test partition by student:{round(running_acc_val,2)} ")

        print(f" accuracy on test partition by student:{round(running_acc_val,2)} ")

        self._LOG.info(
            f"****************end of loading and testing a model*********************")
        return


    def eval_no_fnc(self,classifier,args_in,dataset,epoch_index):
        batch_generator_val = generate_batches(dataset, workers=args_in.workers, batch_size=args_in.batch_size,
                                               device=args_in.device, shuffle=args_in.shuffle_data)
        running_loss_val = 0.
        running_acc_val = 0.

        no_of_batches= int(len(dataset) / args_in.batch_size)
        for batch_index, batch_dict in enumerate(tqdm(batch_generator_val, desc="dev_batches", total=no_of_batches)):
            # compute the output
            y_pred_val = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])

            # step 3. compute the class_loss
            class_loss = class_loss_func(y_pred_val, batch_dict['y_target'])
            loss_t = class_loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            y_pred_labels_val_sf = F.softmax(y_pred_val, dim=1)
            right_predictions, acc_t, predictions_by_label_class = self.compute_accuracy(y_pred_labels_val_sf,
                                                                                         batch_dict['y_target'])
            running_acc_val += (acc_t - running_acc_val) / (batch_index + 1)


            self._LOG.debug(
                f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches} \t per_batch_accuracy_dev_set:{round(acc_t,4)} \t moving_avg_val_accuracy:{round(running_acc_val,4)} ")

        return running_acc_val,running_loss_val

    def eval(self, classifier, args_in, dataset, epoch_index, vectorizer, list_of_datapoint_dictionaries,desc_in):
        batch_generator_val = generate_batches_without_sampler(dataset, workers=args_in.workers, batch_size=args_in.batch_size,
                                               device=args_in.device, shuffle=False,drop_last=False)
        running_loss_val = 0.
        plain_accuracy = 0.

        total_predictions = []
        total_gold = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #this is for calculating microf1 score
        all_predictions = []
        all_gold_labels = []
        #
        # if torch.cuda.is_available():
        #     all_predictions = torch.cuda.FloatTensor(all_predictions)
        #     all_gold_labels_tensor = torch.cuda.LongTensor(all_gold_labels_tensor)
        # else:
        # all_predictions = torch.tensor(all_predictions)
        # all_gold_labels_tensor = torch.LongTensor(all_gold_labels_tensor)


        no_of_batches= int(len(dataset) / args_in.batch_size)
        for batch_index, batch_dict in enumerate(tqdm(batch_generator_val, desc=desc_in, total=no_of_batches)):
            # compute the output
            y_pred_val = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])

            # step 3. compute the class_loss
            class_loss = class_loss_func(y_pred_val, batch_dict['y_target'])
            loss_t = class_loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy
            all_gold_labels.extend(batch_dict['y_target'].tolist())
            #all_predictions= torch.cat((all_predictions, y_pred_val), 0)

            predictions_by_label_class_from_fnc=[]
            acc_t = 0
            # fnc alone has a different kind of scoring. we are using the official scoring function. Note that the
            # command line argument 'database_to_test_with' is used only for deciding the scoring function. it has nothing
            # to do with which test file to load.
            if (args_in.database_to_test_with == "fnc"):
                predictions_by_label_class_from_fnc = self.get_argmax(y_pred_val.float())
                predictions_str_labels = self.get_label_strings_given_vectorizer(vectorizer, predictions_by_label_class_from_fnc)
                gold_str = self.get_label_strings_given_list(batch_dict['y_target'])
                for e in gold_str:
                    total_gold.append(e)
                for e in predictions_str_labels:
                    total_predictions.append(e)
            #all_predictions.extend(predictions_by_label_class.tolist())
            #predictions_by_label_class=predictions_by_label_class[0]


            # compute the plain/classic accuracy
            y_pred_labels_val_sf = F.softmax(y_pred_val, dim=1)
            right_predictions, acc_t, predictions_by_label_class_from_accuracy = self.compute_accuracy(y_pred_labels_val_sf,
                                                                                         batch_dict['y_target'])
            plain_accuracy += (acc_t - plain_accuracy) / (batch_index + 1)

            #if fnc_prediction does exist- both should ideally be the same
            if (len(predictions_by_label_class_from_fnc)>0):
                for x,y in zip(predictions_by_label_class_from_fnc,predictions_by_label_class_from_accuracy):
                    assert x==y

            all_predictions.extend(predictions_by_label_class_from_accuracy.tolist())




            self._LOG.debug(
                f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches} \t per_batch_accuracy_dev_set:{round(acc_t,4)} \t moving_avg_val_accuracy:{round(plain_accuracy,4)} ")




            # get predictions as plain text
            indices_this_batch_of_delex = batch_dict["datapoint_index"]
            predictions_within_batch=[]
            self.get_plain_text_given_data_point_batch_in_indices(batch_dict, vectorizer,
                                                                  predictions_within_batch,
                                                                  y_pred_val,
                                                                  predictions_by_label_class_from_accuracy,
                                                                  indices_this_batch_of_delex)
            list_of_datapoint_dictionaries.extend(predictions_within_batch)
        fnc_score=0.00
        if (args_in.database_to_test_with == "fnc"):
            fnc_score = report_score(total_gold, total_predictions)

        microf1 = self.calculate_micro_f1(all_predictions, all_gold_labels)

        return plain_accuracy,running_loss_val,microf1,fnc_score


    def write_dict_as_json(self,out_path,list_of_dictionaries):

        for d in list_of_dictionaries:
            with open(out_path, 'a+') as outfile:
                json.dump(d, outfile)
                outfile.write("\n")

    def get_plain_text_given_data_point_batch_in_indices(self, batch, vectorizer, list_of_datapoint_dictionaries, batch_predictions_logits,batch_predictions_labels,indices_this_batch):
        '''
        input: batch of data points in indices format.
        take batch, run through each item, separate claim, evidence indices, feed it into another function which converts indices to tokens
        :return: data points (claim, evidence, prediciton logits, prediciton labels and gold labels) in plain text
        '''
        #convert from tensor to lists of lists
        list_of_all_claims_in_this_batch=batch['x_claim'].tolist()
        list_of_all_evidences_in_this_batch=batch['x_evidence'].tolist()
        list_of_all_gold_labels_in_this_batch=batch['y_target'].tolist()
        list_of_all_prediction_logits_in_this_batch=batch_predictions_logits.tolist()
        list_of_all_prediction_labels_in_this_batch=batch_predictions_labels.tolist()
        list_of_all_indices_in_this_batch = indices_this_batch.tolist()


        #for each data point in the batch
        for index,claim,ev,gold,prediction_logit,predictions_label in zip(list_of_all_indices_in_this_batch,
                                                                    list_of_all_claims_in_this_batch,
                                                                    list_of_all_evidences_in_this_batch,
                                                                    list_of_all_gold_labels_in_this_batch,
                                                                    list_of_all_prediction_logits_in_this_batch,
                                                                    list_of_all_prediction_labels_in_this_batch):
            datapoint = {}
            claim_plain_text=self.look_up_plain_text_datapoint_using_vectorizer(claim,vectorizer)
            evidence_plain_text = self.look_up_plain_text_datapoint_using_vectorizer(ev, vectorizer)
            gold_label_plain_text = self.get_label_string_given_vectorizer(vectorizer,gold )
            prediction_label_plain_text = self.get_label_string_given_vectorizer(vectorizer, predictions_label)

            datapoint["index"] = index
            datapoint["claim"]=claim_plain_text
            datapoint["evidence"] = evidence_plain_text
            datapoint["gold_label"] = gold_label_plain_text
            datapoint["prediction_logit"] = prediction_logit
            datapoint["prediction_label"] = prediction_label_plain_text


            list_of_datapoint_dictionaries.append(datapoint)
        return


    def convert_predicted_logits_into_batch_prediction_format(self,logits_vertical):
        all_rows_in_batch=[]
        for column in range(len(logits_vertical[0])):
            one_data_point_predictions=[]
            for row in range(4):
                one_data_point_predictions.append(logits_vertical[row][column])
            all_rows_in_batch.append(one_data_point_predictions)
        return torch.FloatTensor(all_rows_in_batch)






    def train(self, args_in, classifier_teacher_lex, classifier_student_delex, dataset, comet_value_updater, vectorizer):


        if args_in.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif args_in.consistency_type == 'kl':
            consistency_criterion = losses.softmax_kl_loss

        classifier_teacher_lex = classifier_teacher_lex.to(args_in.device)

        # if you want to train teacher alone, you should turn add_student=False
        if (args_in.add_student == True):
            classifier_student_delex = classifier_student_delex.to(args_in.device)
            input_optimizer, inter_atten_optimizer = initialize_optimizers([classifier_teacher_lex, classifier_student_delex], args_in)
        else:

            input_optimizer, inter_atten_optimizer = initialize_optimizers(
                [classifier_teacher_lex], args_in)

        train_state_in = self.make_train_state(args_in)



        try:
            # Iterate over training dataset
            for epoch_index in range(args_in.num_epochs):

                train_state_in['epoch_index'] = epoch_index


                # setup: batch generator, set class_loss_lex and acc to 0, set train mode on
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
                running_loss_delex = 0.0
                running_acc_delex = 0.0

                #when you are loading a trained teachremodel you don't need backpropagation
                if (args_in.use_trained_teacher_inside_student_teacher_arch):
                    classifier_teacher_lex.eval()
                else:
                    classifier_teacher_lex.train()

                classifier_student_delex.train()









                total_right_predictions_teacher_lex=0
                total_right_predictions_student_delex = 0
                total_gold_label_count=0


                combined_data_generators = zip(batch_generator_lex_data, batch_generator_delex_data)

                assert combined_data_generators is not None

                for batch_index, (batch_dict_lex,batch_dict_delex) in enumerate(tqdm(combined_data_generators,desc="training_batches",total=no_of_batches_delex)):

                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    input_optimizer.zero_grad()
                    inter_atten_optimizer.zero_grad()



                    #initializing initial state of the optimizer to start from 0. This should be learned/tuned hyper parameter.
                    # remove if not having any effect/improvement
                    if epoch_index == 0 and args_in.optimizer == 'adagrad':
                        update_optimizer_state(input_optimizer, inter_atten_optimizer, args_in)




                    # step 2. compute the output
                    y_pred_lex=None

                    #if you want to load the teacher which was already trained in a previous phase.
                    if(args_in.use_trained_teacher_inside_student_teacher_arch):
                        #directly use the logits of the prediction from phase1
                        y_pred_lex =  batch_dict_lex['predicted_logits']


                    else:

                        if (args_in.use_ema):
                            # when in ema mode, make the teacher also make its  prediction over delex data .
                            #  In ema mode, this wont be back propagated, so wouldn't really matter.
                            y_pred_lex = classifier_teacher_lex(batch_dict_delex['x_claim'], batch_dict_delex['x_evidence'])
                        else:
                            y_pred_lex = classifier_teacher_lex(batch_dict_lex['x_claim'], batch_dict_lex['x_evidence'])


                    assert y_pred_lex is not None
                    assert len(y_pred_lex) > 0


                    total_gold_label_count=total_gold_label_count+len(batch_dict_lex['y_target'])


                    # step 3.1 compute the class_loss_lex
                    class_loss_lex = class_loss_func(y_pred_lex, batch_dict_lex['y_target'])

                    loss_t_lex = class_loss_lex.item()
                    running_loss_lex += (loss_t_lex - running_loss_lex) / (batch_index + 1)
                    self._LOG.debug(f"loss_t_lex={loss_t_lex}\trunning_loss_lex={running_loss_lex}")

                    combined_class_loss = class_loss_lex
                    if (args_in.use_trained_teacher_inside_student_teacher_arch):
                        combined_class_loss = 0

                    consistency_loss=0
                    class_loss_delex=None

                    #all classifier2 related code (the one which feeds off delexicalized data). all steps before .backward()
                    if (args_in.add_student == True):
                        y_pred_delex = classifier_student_delex(batch_dict_delex['x_claim'], batch_dict_delex['x_evidence'])
                        class_loss_delex = class_loss_func(y_pred_delex, batch_dict_delex['y_target'])
                        loss_t_delex = class_loss_delex.item()
                        running_loss_delex += (loss_t_delex - running_loss_delex) / (batch_index + 1)
                        #LOG.debug(f"loss_t_delex={loss_t_delex}\trunning_loss_delex={running_loss_delex}")

                        consistency_loss = consistency_criterion(y_pred_lex, y_pred_delex)
                        consistency_loss_value = consistency_loss.item()
                        running_consistency_loss += (consistency_loss_value - running_consistency_loss) / (batch_index + 1)

                        # when in ema mode (teacher is the exponential moving average of student) or when loading a
                        # trained teacher there is no back propagation in teacher and hence adding its classification loss is useless

                        if (args_in.use_ema) or (args_in.use_trained_teacher_inside_student_teacher_arch):
                            combined_class_loss=class_loss_delex
                        else:
                            combined_class_loss = class_loss_delex + class_loss_lex






                    #combined loss is the sum of two classification losses and one consistency loss
                    combined_loss = (args_in.consistency_weight * consistency_loss) + (combined_class_loss)
                    combined_loss.backward()

                    #to run both student and teacher independently
                    #class_loss_lex.backward()
                    #class_loss_delex.backward()





                    # step 5. use optimizer to take gradient step
                    #optimizer.step()
                    input_optimizer.step()
                    inter_atten_optimizer.step()

                    global_variables.global_step += 1
                    # when in ema mode, teacher is the exponential moving average of the student. that calculation is done here
                    if (args_in.use_ema):
                        self.update_ema_variables(classifier_student_delex, classifier_teacher_lex, args_in.ema_decay, global_variables.global_step)



                    # -----------------------------------------



                    # compute the accuracy for lex data
                    y_pred_labels_lex_sf = F.softmax(y_pred_lex, dim=1)
                    count_of_right_predictions_teacher_lex_per_batch, acc_t_lex,teacher_predictions_by_label_class = \
                        self.compute_accuracy(y_pred_labels_lex_sf, batch_dict_lex['y_target'])


                    total_right_predictions_teacher_lex=total_right_predictions_teacher_lex+count_of_right_predictions_teacher_lex_per_batch
                    running_acc_lex += (acc_t_lex - running_acc_lex) / (batch_index + 1)



                    # all classifier2 related code to calculate accuracy
                    if (args_in.add_student == True):
                        y_pred_labels_delex_sf = F.softmax(y_pred_delex, dim=1)
                        count_of_right_predictions_student_delex_per_batch,acc_t_delex,student_predictions_by_label_class = self.compute_accuracy(y_pred_labels_delex_sf, batch_dict_lex['y_target'])

                        total_right_predictions_student_delex=total_right_predictions_student_delex+count_of_right_predictions_student_delex_per_batch
                        running_acc_delex += (acc_t_delex - running_acc_delex) / (batch_index + 1)
                        self._LOG.debug(
                            f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                            f"classification_loss_lex:{round(running_loss_lex,2)}\t classification_loss_delex:{round(running_loss_delex,2)} "
                            f"\t consistencyloss:{round(running_consistency_loss,6)}"
                            f" \t running_acc_lex:{round(running_acc_lex,4) }  \t running_acc_delex:{round(running_acc_delex,4)}   ")




                    else:

                        self._LOG.debug(
                            f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                            f"training_loss_lex_per_batch:{round(running_loss_lex,2)}\t"
                            f" \t training_accuracy_lex_per_batch:{round(running_acc_lex,2) }")
                    assert len(teacher_predictions_by_label_class)>0
                    assert len(batch_dict_lex['y_target']) > 0

                    if (args_in.add_student == True):
                        assert len(student_predictions_by_label_class) > 0



                        comet_value_updater.log_metric(
                            "training accuracy of lex teacher  across batches",
                            running_acc_lex,
                            step=batch_index)



                    if (args_in.add_student == True):
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
                    comet_value_updater.log_metric("training accuracy of teacher model per epoch", running_acc_lex,step=epoch_index)
                    comet_value_updater.log_metric("training accuracy of student model per epoch", running_acc_delex,
                                                   step=epoch_index)





                    if (args_in.add_student == True):
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

                if (args_in.add_student == True):
                    # Iterate over val dataset and check on dev using the intended trained model, which usually is the student delex model
                    dataset.set_split('val_delex')
                    classifier_student_delex.eval()
                    predictions_by_student_model_on_dev=[]

                    running_acc_val_student,running_loss_val_student,microf1_student_dev,fnc_score_student_dev= self.eval(classifier_student_delex, args_in, dataset,epoch_index,vectorizer,predictions_by_student_model_on_dev,"student_delex_on_dev")





                #when in ema mode, teacher is same as student pretty much. so test on delex partition of dev.
                # else teacher and student are separate entities. use teacher to test on dev parition of lexicalized data itself.
                if not (args_in.use_ema):
                    dataset.set_split('val_lex')

                #eval on the lex dev dataset
                classifier_teacher_lex.eval()
                predictions_by_teacher_model_on_dev = []
                running_acc_val_teacher,running_loss_val_teacher,microf1_teacher_dev,fnc_score_teacher_dev = self.eval(classifier_teacher_lex, args_in, dataset,epoch_index,vectorizer,predictions_by_teacher_model_on_dev,"teacher_lex_on_dev")



                assert comet_value_updater is not None
                if (args_in.add_student == True):
                    comet_value_updater.log_metric("acc_dev_per_epoch_using_student_model", running_acc_val_student, step=epoch_index)
                comet_value_updater.log_metric("acc_dev_per_epoch_using_teacher_model", running_acc_val_teacher, step=epoch_index)
                comet_value_updater.log_metric("microf1_dev_per_epoch_using_student_model", microf1_student_dev,
                                               step=epoch_index)
                comet_value_updater.log_metric("microf1_dev_per_epoch_using_teacher_model", microf1_teacher_dev,
                                               step=epoch_index)
 
                # Do early stopping based on when the dev accuracy drops from its best for patience=5
                train_state_in['val_loss'].append(running_loss_val_student)
                train_state_in['val_acc'].append(running_acc_val_student)
                train_state_in = self.update_train_state(args=args_in,
                                                         models=[classifier_student_delex, classifier_teacher_lex],
                                                         train_state=train_state_in)




                # Do early stopping based on when the dev accuracy drops from its best for patience=5
                # update: the code here does early stopping based on cross domain dev(which is being fed as a test partition)
                # . i.e not based on in-domain dev anymore.
                if (args_in.add_student == True):
                    train_state_in['val_loss'].append(running_loss_val_student)
                    train_state_in['val_acc'].append(running_acc_val_student)
                    train_state_in = self.update_train_state(args=args_in,
                                                             models=[classifier_student_delex, classifier_teacher_lex],
                                                             train_state=train_state_in)
                else:
                    train_state_in['val_loss'].append(running_loss_val_teacher)
                    train_state_in['val_acc'].append(running_acc_val_teacher)
                    train_state_in = self.update_train_state(args=args_in,
                                                         models=[classifier_student_delex, classifier_teacher_lex],
                                                         train_state=train_state_in)


                # also test it on a third dataset which is usually cross domain on fnc
                if(args_in.test_in_cross_domain_dataset):
                    args_in.database_to_test_with="fnc"

                    if (args_in.add_student == True):
                        dataset.set_split('test_delex')
                        predictions_by_student_model_on_test_partition = []
                        classifier_student_delex.eval()
                        running_acc_test_student, running_loss_test_student,microf1_student_test, \
                        fnc_score_student_test= self.eval(classifier_student_delex, args_in,
                                                                                    dataset, epoch_index,vectorizer,predictions_by_student_model_on_test_partition,"student_delex_on_test")

                    dataset.set_split('test_lex')
                    classifier_teacher_lex.eval()
                    predictions_by_teacher_model_on_test_partition=[]




                    running_acc_test_teacher, running_loss_test_teacher,microf1_teacher_test\
                        ,fnc_score_teacher_test= self.eval(classifier_teacher_lex, args_in,
                                                                                   dataset,
                                                                                   epoch_index,vectorizer,predictions_by_teacher_model_on_test_partition,"teacher_lex_on_test")

                    if (args_in.add_student == True):
                        comet_value_updater.log_metric("plain_acc_test_student", running_acc_test_student,
                                                   step=epoch_index)
                        comet_value_updater.log_metric("microf1_test_student", microf1_student_test,
                                                       step=epoch_index)
                        comet_value_updater.log_metric("fnc_score_student_on_test_partition", fnc_score_student_test,
                                                       step=epoch_index)


                comet_value_updater.log_metric("plain_acc_test_teacher", running_acc_test_teacher,
                                                   step=epoch_index)
                comet_value_updater.log_metric("microf1_test_teacher", microf1_teacher_test,
                                               step=epoch_index)
                comet_value_updater.log_metric("fnc_score_teacher_on_test_partition", fnc_score_teacher_test,
                                               step=epoch_index)


                #resetting args_in.database_to_test_with to make sure the values don't persist across epochs
                args_in.database_to_test_with = "dummy"
                dataset.set_split('val_lex')

                # empty out the predictions file and write into it at the end of every epoch
                #note: this is for debug purposes on april 12th. ideally the emptying out shoudl happen before all epochs and
                #writing out should happen only at early stopping
                with open(args_in.predictions_teacher_dev_file, 'w') as outfile:
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
                self.write_dict_as_json(args_in.predictions_teacher_dev_file, predictions_by_teacher_model_on_dev)
                self.write_dict_as_json(args_in.predictions_student_test_file,
                                        predictions_by_student_model_on_test_partition)
                self.write_dict_as_json(args_in.predictions_teacher_test_file,
                                        predictions_by_teacher_model_on_test_partition)

                if train_state_in['stop_early']:
                    ## whenever you hit early stopping just store all the data and predictions at that point to disk for debug purposes

                    assert len(predictions_by_student_model_on_dev) > 0
                    assert len(predictions_by_teacher_model_on_dev) > 0
                    assert len(predictions_by_student_model_on_test_partition) > 0
                    assert len(predictions_by_teacher_model_on_test_partition) > 0

                    self.write_dict_as_json(args_in.predictions_student_dev_file, predictions_by_student_model_on_dev)
                    self.write_dict_as_json(args_in.predictions_teacher_dev_file, predictions_by_teacher_model_on_dev)
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
                    f" microf1 on dev partition by student:{round(microf1_student_dev,2)} ")
                self._LOG.info(
                    f" microf1 on dev partition by teacher:{round(microf1_teacher_dev,2)} ")
                self._LOG.info(
                    f" microf1 on test partition by student:{round(microf1_student_test,2)} ")
                self._LOG.info(
                    f" microf1 on test partition by teacher:{round(microf1_teacher_test,2)} ")
                self._LOG.info(
                    f"****************end of epoch {epoch_index}*********************")
            print("****************end of all epochs*********************")
            self._LOG.info(
            f"****************end of all epochs*********************")

        except KeyboardInterrupt:
            print("Exiting loop")



    def create_vocabulary_for_cross_domain_dataset(self,lex_input_file,delex_input_file,args):
        dataset_cross_domain = RTEDataset.create_vocab_given_lex_delex_file_paths(lex_input_file,delex_input_file,args)
        vectorizer_cross_domain = dataset_cross_domain.get_vectorizer()
        # taking embedding size from user initially, but will get replaced by original embedding size if its loaded
        embedding_size = args.embedding_size
        # Use GloVe or randomly initialized embeddings
        self._LOG.info(f"{current_time} going to load glove from path:{glove_filepath_in}")
        if args.use_glove:
            words = vectorizer_cross_domain.claim_ev_vocab._token_to_idx.keys()
            embeddings, embedding_size = make_embedding_matrix(main.glove_filepath_in, words)
        return dataset_cross_domain, vectorizer_cross_domain