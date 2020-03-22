from mean_teacher.utils.utils_rao import generate_batches,initialize_optimizers,update_optimizer_state,generate_batches_for_semi_supervised
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
                torch.save(model.state_dict(), train_state['model_filename'] + model_type + "_e" + str(train_state['epoch_index']) + ".pth")
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
                # Save the best model
                for index, model in enumerate(models):
                    model_type = "student"
                    if index > 0:
                        model_type = "teacher"
                    torch.save(model.state_dict(), train_state['model_filename']+"_best_"+model_type + ".pth")
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

        # gold labels and predictions are in transposes (eg:1x15 vs 15x1). so take a transpose to correct it.
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
        _, y_pred_classes = y_pred.max(dim=1)
        labels_to_include =[]
        for index,l in enumerate(y_target):
            if not (l==3):
                labels_to_include.append(index)
        mf1=metrics.f1_score(y_target.tolist(),y_pred_classes.tolist(), average='micro', labels=labels_to_include)
        return mf1

    def compute_accuracy(self,y_pred, y_target):
        assert len(y_pred)==len(y_target)
        _, y_pred_classes = y_pred.max(dim=1)
        n_correct = torch.eq(y_pred_classes, y_target).sum().item()

        m = nn.Softmax()
        output_sftmax = m(y_pred.tolist())
        _, pred = output_sftmax.topk(1, 1, True, True)


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
        return pred.t()

    def get_label_strings_given_vectorizer(self, vectorizer, predictions_index_labels):
        labels_str=[]
        for e in predictions_index_labels[0]:
            labels_str.append(vectorizer.label_vocab.lookup_index(e.item()).lower())
        return labels_str

    def get_label_strings_given_list(self, labels_tensor):
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
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
                                               device=args_in.device, shuffle=False)
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
                                               device=args_in.device, shuffle=False)
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

    def eval(self,classifier,args_in,dataset,epoch_index,vectorizer):
        batch_generator_val = generate_batches(dataset, workers=args_in.workers, batch_size=args_in.batch_size,
                                               device=args_in.device, shuffle=False)
        running_loss_val = 0.
        running_acc_val = 0.

        total_predictions = []
        total_gold = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #this is for calculating microf1 score
        all_predictions_tensor = []
        all_gold_labels_tensor = []
        #
        # if torch.cuda.is_available():
        #     all_predictions_tensor = torch.cuda.FloatTensor(all_predictions_tensor)
        #     all_gold_labels_tensor = torch.cuda.LongTensor(all_gold_labels_tensor)
        # else:
        all_predictions_tensor = torch.tensor(all_predictions_tensor)
        all_gold_labels_tensor = torch.LongTensor(all_gold_labels_tensor)


        no_of_batches= int(len(dataset) / args_in.batch_size)
        for batch_index, batch_dict in enumerate(tqdm(batch_generator_val, desc="dev_batches", total=no_of_batches)):
            # compute the output
            y_pred_val = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])

            # step 3. compute the class_loss
            class_loss = class_loss_func(y_pred_val, batch_dict['y_target'])
            loss_t = class_loss.item()
            running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # compute the accuracy

            all_gold_labels_tensor = torch.cat((all_gold_labels_tensor, batch_dict['y_target']), 0)
            all_predictions_tensor= torch.cat((all_predictions_tensor, y_pred_val), 0)


            acc_t = 0
            # fnc alone has a different kind of scoring. we are using the official scoring function. Note that the
            # command line argument 'database_to_test_with' is used only for deciding the scoring function. it has nothing
            # to do with which test file to load.
            if (args_in.database_to_test_with == "fnc"):
                predictions_index_labels = self.get_argmax(y_pred_val.float())
                predictions_str_labels = self.get_label_strings_given_vectorizer(vectorizer, predictions_index_labels)
                gold_str = self.get_label_strings_given_list(batch_dict['y_target'])
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

            self._LOG.debug(
                f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches} \t per_batch_accuracy_dev_set:{round(acc_t,4)} \t moving_avg_val_accuracy:{round(running_acc_val,4)} ")

        microf1 = self.calculate_micro_f1(all_predictions_tensor, all_gold_labels_tensor)
        return running_acc_val,running_loss_val,microf1



    def train(self, args_in, classifier_teacher_lex, classifier_student_delex, dataset, comet_value_updater, vectorizer):


        if args_in.consistency_type == 'mse':
            consistency_criterion = losses.softmax_mse_loss
        elif args_in.consistency_type == 'kl':
            consistency_criterion = losses.softmax_kl_loss

        classifier_teacher_lex = classifier_teacher_lex.to(args_in.device)

        #historically we had a case where we had to run just the teacher alone. This is vestigial from there. Right now, Feb 2020, we
        #almost always run in student-teacher mode
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
                #WHEN use_semi_supervised is turned on, only part of the gold labels will be given to the classifier. Rest all will be masked.
                if(args_in.use_semi_supervised==True):
                    assert args_in.percentage_labels_for_semi_supervised > 0
                    batch_generator_lex_data = generate_batches_for_semi_supervised(dataset_lex, args_in.percentage_labels_for_semi_supervised, workers=args_in.workers, batch_size=args_in.batch_size,
                                                        device=args_in.device,mask_value=args_in.NO_LABEL )
                else:
                    batch_generator_lex_data = generate_batches(dataset_lex, workers=args_in.workers, batch_size=args_in.batch_size,device=args_in.device)

                no_of_batches_lex = int(len(dataset)/args_in.batch_size)

                assert batch_generator_lex_data is not None
                batch_generator_delex_data = None

                if (args_in.add_student == True):
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
                        batch_generator_delex_data = generate_batches(dataset_delex, workers=args_in.workers, batch_size=args_in.batch_size,
                                                            device=args_in.device)

                    assert batch_generator_delex_data is not None

                no_of_batches_delex = int(len(dataset) / args_in.batch_size)

                running_consistency_loss = 0.0


                running_loss_lex = 0.0
                running_acc_lex = 0.0
                running_loss_delex = 0.0
                running_acc_delex = 0.0
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
                    #when in ema mode, make the teacher also make its  prediction over delex data .
                    #  In ema mode, this wont be back propagated, so wouldn't really matter.
                    if (args_in.use_ema):
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

                        # when in ema mode, teacher is the exponential moving average of student. therefore there is no
                        #back propagation in teacher and hence adding its classification loss is useless

                        if (args_in.use_ema):
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
                    assert len(student_predictions_by_label_class) > 0
                    assert len(batch_dict_lex['y_target']) > 0

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



                self._LOG.info(
                    f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                    f"classification_loss_lex:{round(running_loss_lex,2)}\t classification_loss_delex:{round(running_loss_delex,2)} "
                    f"\t consistencyloss:{round(running_consistency_loss,6)}"
                    f" \t running_acc_lex:{round(running_acc_lex,4) }  \t running_acc_delex:{round(running_acc_delex,4)}  ")


                train_state_in['train_acc'].append(running_acc_lex)
                train_state_in['train_loss'].append(running_loss_lex)

                #for debugging: make the model predict on training data at the end of every epoch

                self.number_of_datapoints = total_gold_label_count
                accuracy_teacher_model_by_per_batch_prediction = self.calculate_percentage(total_right_predictions_teacher_lex,self.number_of_datapoints)
                accuracy_student_model_by_per_batch_prediction = self.calculate_percentage(
                    total_right_predictions_student_delex, self.number_of_datapoints)



                self._LOG.info(
                    f"running_acc_lex training by old method at the end of {epoch_index}:{running_acc_lex}")
                self._LOG.info(
                    f"accuracy_teacher_model_by_per_batch_prediction at the end of epoch{epoch_index}:{accuracy_teacher_model_by_per_batch_prediction}")

                self._LOG.info(
                    f"acc_t_delex by old method {epoch_index}:{running_acc_delex}")

                self._LOG.info(
                    f"accuracy_student_model_by_per_batch_prediction method at the end of epoch{epoch_index}:{ accuracy_student_model_by_per_batch_prediction}")




                self._LOG.info(
                    f"epoch:{epoch_index}")



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
                    if (comet_value_updater is not None):
                        comet_value_updater.log_metric("consistency_loss per epoch",
                                                       running_consistency_loss,
                                                       step=epoch_index)



                # Iterate over val dataset and check on dev using the intended trained model, which usually is the student delex model
                dataset.set_split('val_delex')
                classifier_student_delex.eval()
                running_acc_val_student,running_loss_val_student,microf1_student_dev= self.eval(classifier_student_delex, args_in, dataset,epoch_index,vectorizer)

                #when in ema mode, teacher is same as student pretty much. so test on delex partition of dev.
                # else teacher and student are separate entities. use teacher to test on dev parition of lexicalized data itself.
                if not (args_in.use_ema):
                    dataset.set_split('val_lex')
                classifier_teacher_lex.eval()
                running_acc_val_teacher,running_loss_val_teacher,microf1_teacher_dev = self.eval(classifier_teacher_lex, args_in, dataset,epoch_index,vectorizer)




                assert comet_value_updater is not None
                comet_value_updater.log_metric("acc_dev_per_epoch_using_student_model", running_acc_val_student, step=epoch_index)
                comet_value_updater.log_metric("acc_dev_per_epoch_using_teacher_model", running_acc_val_teacher, step=epoch_index)
                comet_value_updater.log_metric("microf1_dev_per_epoch_using_student_model", microf1_student_dev,
                                               step=epoch_index)
                comet_value_updater.log_metric("microf1_dev_per_epoch_using_teacher_model", microf1_teacher_dev,
                                               step=epoch_index)

                # also test it on a third dataset which is usually cross domain on fnc
                args_in.database_to_test_with="fnc"
                dataset.set_split('test_delex')
                classifier_student_delex.eval()
                running_acc_test_student, running_loss_test_student,microf1_student_test = self.eval(classifier_student_delex, args_in,
                                                                                dataset, epoch_index,vectorizer)

                dataset.set_split('test_lex')
                classifier_teacher_lex.eval()
                running_acc_test_teacher, running_loss_test_teacher,microf1_teacher_test = self.eval(classifier_teacher_lex, args_in,
                                                                                dataset,
                                                                                epoch_index,vectorizer)

                comet_value_updater.log_metric("running_acc_test_student", running_acc_test_student,
                                               step=epoch_index)
                comet_value_updater.log_metric("running_acc_test_teacher", running_acc_test_teacher,
                                               step=epoch_index)

                comet_value_updater.log_metric("microf1_test_student", microf1_student_test,
                                               step=epoch_index)
                comet_value_updater.log_metric("microf1_test_teacher", microf1_teacher_test,
                                               step=epoch_index)

                # Do early stopping based on when the dev accuracy drops from its best for patience=5
                # update: the code here does early stopping based on cross domain dev. i.e not based on in-domain dev anymore.
                train_state_in['val_loss'].append(running_loss_test_student)
                train_state_in['val_acc'].append(running_acc_test_student)
                train_state_in = self.update_train_state(args=args_in, models=[classifier_student_delex,classifier_teacher_lex],train_state=train_state_in)

                #resetting args_in.database_to_test_with to make sure the values don't persist across epochs
                args_in.database_to_test_with = "dummy"
                dataset.set_split('val_lex')


                if train_state_in['stop_early']:
                    break



                self._LOG.info(
                    f" accuracy on dev partition by student:{round(running_acc_val_student,2)} ")
                self._LOG.info(
                    f" accuracy on dev partition by teacher:{round(running_acc_val_teacher,2)} ")
                self._LOG.info(
                    f" accuracy on test partition by student:{round(running_acc_test_student,2)} ")
                self._LOG.info(
                    f" accuracy on test partition by teacher:{round(running_acc_test_teacher,2)} ")
                self._LOG.info(
                    f"****************end of epoch {epoch_index}*********************")


        except KeyboardInterrupt:
            print("Exiting loop")



