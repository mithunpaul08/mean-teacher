from mean_teacher.utils.utils_rao import generate_batches,initialize_optimizers,update_optimizer_state,generate_batches_for_semi_supervised
from mean_teacher.utils import losses
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm,tqdm_notebook
from torch.nn import functional as F
from mean_teacher.utils.logger import LOG
NO_LABEL=-1
class Trainer():
    def __init__(self,LOG):
        self._LOG=LOG
        self._current_time={time.strftime("%c")}

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

    def update_train_state(self, args, model, train_state):
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
            torch.save(model.state_dict(), "model" + "_e" + str(train_state['epoch_index']) + ".pth")
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
                LOG.info(f"found that acc_current_epoch  {acc_current_epoch} is less than or equal to the best dev "
                         f"accuracy value so far which is"
                         f" {train_state['early_stopping_best_val']}. "
                         f"Increasing patience total value. "
                         f"of patience now is {train_state['early_stopping_step']}")
            # accuracy increased
            else:
                # Save the best model
                torch.save(model.state_dict(), train_state['model_filename'] + ".pth")
                LOG.info(
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

    def compute_accuracy(self,y_pred, y_target):
        y_target = y_target.cpu()
        n_correct = torch.eq(y_pred, y_target.float()).sum().item()
        return n_correct / len(y_pred) * 100

    def get_learning_rate(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def calculate_argmax_list(self, logit):
        list_labels_pred = []
        for tensor in logit:
            values, indices = torch.max(tensor, 0)
            list_labels_pred.append(indices.data.item())
        return list_labels_pred

    def train(self, args_in, classifier_student1,classifier_student2, dataset,comet_value_updater):




        # if args_in.consistency_type == 'mse':
        #     #consistency_criterion = losses.softmax_mse_loss
        #     consistency_criterion=nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()
        # elif args_in.consistency_type == 'kl':
        #     consistency_criterion = losses.softmax_kl_loss

        if torch.cuda.is_available():
            consistency_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
        else:
            consistency_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()


        if torch.cuda.is_available():
            class_loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
        else:
            class_loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cpu()




        if (args_in.add_second_student == True):
            classifier_student2 = classifier_student2.to(args_in.device)
            input_optimizer, inter_atten_optimizer = initialize_optimizers([classifier_student1,classifier_student2], args_in)
        else:
            classifier_student1 = classifier_student1.to(args_in.device)
            input_optimizer, inter_atten_optimizer = initialize_optimizers(
                [classifier_student1], args_in)

        train_state_in = self.make_train_state(args_in)
        epoch_bar = tqdm_notebook(desc='training routine',
                                  total=args_in.num_epochs,
                                  position=0)
        dataset.set_split('train_lex')
        train_bar = tqdm_notebook(desc='split=train',
                                  total=dataset.get_num_batches(args_in.batch_size),
                                  position=1,
                                  leave=True)
        dataset.set_split('val_lex')
        val_bar = tqdm_notebook(desc='split=val',
                                total=dataset.get_num_batches(args_in.batch_size),
                                position=1,
                                leave=True)


        try:
            for epoch_index in range(args_in.num_epochs):
                train_state_in['epoch_index'] = epoch_index

                # Iterate over training dataset

                # setup: batch generator, set class_loss_lex and acc to 0, set train mode on
                dataset.set_split('train_lex')

                batch_generator1=None
                if(args_in.use_semi_supervised==True):
                    assert args_in.percentage_labels_for_semi_supervised > 0
                    batch_generator1 = generate_batches_for_semi_supervised(dataset, args_in.percentage_labels_for_semi_supervised, workers=args_in.workers, batch_size=args_in.batch_size,
                                                        device=args_in.device,mask_value=args_in.NO_LABEL )
                else:
                    batch_generator1 = generate_batches(dataset, workers=args_in.workers, batch_size=args_in.batch_size,device=args_in.device)

                no_of_batches_lex = int(len(dataset)/args_in.batch_size)

                assert batch_generator1 is not None

                if (args_in.add_second_student == True):
                    dataset.set_split('train_delex')
                    batch_generator2=None
                    if (args_in.use_semi_supervised == True):
                        assert args_in.percentage_labels_for_semi_supervised > 0
                        batch_generator2 = generate_batches_for_semi_supervised(dataset,
                                                                                args_in.percentage_labels_for_semi_supervised,
                                                                                workers=args_in.workers,
                                                                                batch_size=args_in.batch_size,
                                                                                device=args_in.device,mask_value=args_in.NO_LABEL  )

                    else:
                        batch_generator2 = generate_batches(dataset, workers=args_in.workers, batch_size=args_in.batch_size,
                                                            device=args_in.device)

                    assert batch_generator2 is not None

                no_of_batches_delex = int(len(dataset) / args_in.batch_size)

                running_consistency_loss = 0.0

                running_loss_lex = 0.0
                running_acc_lex = 0.0
                running_loss_delex = 0.0
                running_acc_delex = 0.0
                classifier_student1.train()
                classifier_student2.train()





                for batch_index, (batch_dict_lex,batch_dict_delex) in enumerate(zip(batch_generator1,batch_generator2)):

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
                    y_pred_lex = classifier_student1(batch_dict_lex['x_claim'], batch_dict_lex['x_evidence'])


                    # step 3.1 compute the class_loss_lex
                    class_loss_lex = class_loss_func(y_pred_lex, batch_dict_lex['y_target'])
                    loss_t_lex = class_loss_lex.item()
                    running_loss_lex += (loss_t_lex - running_loss_lex) / (batch_index + 1)
                    LOG.debug(f"loss_t_lex={loss_t_lex}\trunning_loss_lex={running_loss_lex}")

                    combined_class_loss = class_loss_lex
                    consistency_loss=0

                    #all classifier2 related code (the one which feeds off delexicalized data). all steps before .backward()
                    if (args_in.add_second_student == True):
                        y_pred_delex = classifier_student2(batch_dict_delex['x_claim'], batch_dict_delex['x_evidence'])
                        class_loss_delex = class_loss_func(y_pred_delex, batch_dict_delex['y_target'])
                        loss_t_delex = class_loss_delex.item()
                        running_loss_delex += (loss_t_delex - running_loss_delex) / (batch_index + 1)
                        #LOG.debug(f"loss_t_delex={loss_t_delex}\trunning_loss_delex={running_loss_delex}")

                        # step 4. use combined classification loss to produce gradients
                        y_pred_delex_softmax_output = self.calculate_argmax_list((y_pred_delex))

                        if torch.cuda.is_available():
                            y_pred_delex_softmax_output=torch.cuda.LongTensor(y_pred_delex_softmax_output)
                        else:
                            y_pred_delex_softmax_output =torch.LongTensor(y_pred_delex_softmax_output)
                        assert y_pred_delex_softmax_output is not None
                        consistency_loss = consistency_criterion(y_pred_lex, y_pred_delex_softmax_output)
                        consistency_loss_value = consistency_loss.item()
                        running_consistency_loss += (consistency_loss_value - running_consistency_loss) / (batch_index + 1)
                        combined_class_loss=class_loss_lex+class_loss_delex
                        #LOG.debug(f"consistency_loss_value={consistency_loss_value}\trunning_consistency_loss={running_consistency_loss}")


                    combined_loss=(args_in.consistency_weight*consistency_loss)+(combined_class_loss)
                    combined_loss.backward()



                    # step 5. use optimizer to take gradient step
                    #optimizer.step()
                    input_optimizer.step()
                    inter_atten_optimizer.step()



                    # -----------------------------------------



                    # compute the accuracy for lex data

                    y_pred_labels = self.calculate_argmax_list(y_pred_lex)
                    y_pred_labels = torch.FloatTensor(y_pred_labels)
                    acc_t_lex = self.compute_accuracy(y_pred_labels, batch_dict_lex['y_target'])
                    running_acc_lex += (acc_t_lex - running_acc_lex) / (batch_index + 1)

                    if (args_in.add_second_student == True):
                        #all classifier2 related code. second set. i.e all steps per batch after .backward()
                        y_pred_labels_delex = self.calculate_argmax_list(y_pred_delex)
                        y_pred_labels_delex = torch.FloatTensor(y_pred_labels_delex)
                        acc_t_delex = self.compute_accuracy(y_pred_labels_delex, batch_dict_lex['y_target'])
                        running_acc_delex += (acc_t_delex - running_acc_delex) / (batch_index + 1)
                        LOG.info(
                            f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                            f"classification_loss_lex:{round(running_loss_lex,2)}\t classification_loss_delex:{round(running_loss_delex,2)} "
                            f"\t consistency_loss:{round(running_consistency_loss,6)}"
                            f" \t running_acc_lex:{round(running_acc_lex,2) }  \t running_acc_delex:{round(running_acc_delex,2)}  ")
                    else:

                        LOG.info(
                            f"{epoch_index} \t :{batch_index}/{no_of_batches_lex} \t "
                            f"training_loss_lex_per_batch:{round(running_loss_lex,2)}\t"
                            f" \t training_accuracy_lex_per_batch:{round(running_acc_lex,2) }")


                    # update bar
                    train_bar.set_postfix(loss=running_loss_lex,
                                          acc=running_acc_lex,
                                          epoch=epoch_index)
                    train_bar.update()


                train_state_in['train_acc'].append(running_acc_lex)
                train_state_in['train_loss'].append(running_loss_lex)

                if (comet_value_updater is not None):
                    comet_value_updater.log_metric("training_classification_loss_lex_per_epoch", running_loss_lex,
                                                   step=epoch_index)

                if (comet_value_updater is not None):
                    comet_value_updater.log_metric("training_accuracy_lexicalized per epoch", running_acc_lex,
                                                   step=epoch_index)


                if (args_in.add_second_student == True):
                    if (comet_value_updater is not None):
                        comet_value_updater.log_metric("delex_training_loss per epoch", running_loss_delex,
                                                       step=epoch_index)
                    if (comet_value_updater is not None):
                        comet_value_updater.log_metric("training_accuracy_delexicalized per epoch", running_acc_delex,
                                                       step=epoch_index)

                        if (comet_value_updater is not None):
                            comet_value_updater.log_metric("running_consistency_loss per epoch",
                                                           running_consistency_loss,
                                                           step=epoch_index)



                # Iterate over val dataset
                # we will always test/validate on delexicalized data
                dataset.set_split('val_delex')
                batch_generator_val = generate_batches(dataset, workers=args_in.workers, batch_size=args_in.batch_size,
                                                    device=args_in.device, shuffle=False)
                running_loss_val = 0.
                running_acc_val = 0.


                if (args_in.add_second_student == True):
                    classifier_student2.eval()
                else:
                    classifier_student1.eval()

                no_of_batches_lex = int(len(dataset) / args_in.batch_size)

                for batch_index, batch_dict in enumerate(batch_generator_val):
                    # compute the output

                    if (args_in.add_second_student == True):
                        y_pred_val = classifier_student2(batch_dict['x_claim'], batch_dict['x_evidence'])
                    else:
                        y_pred_val = classifier_student1(batch_dict['x_claim'], batch_dict['x_evidence'])

                    # step 3. compute the class_loss
                    class_loss = class_loss_func(y_pred_val, batch_dict['y_target'])
                    loss_t = class_loss.item()
                    running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

                    # compute the accuracy
                    y_pred_labels_val = self.calculate_argmax_list(y_pred_val)
                    y_pred_labels_val = torch.FloatTensor(y_pred_labels_val)
                    acc_t = self.compute_accuracy(y_pred_labels_val, batch_dict['y_target'])
                    running_acc_val += (acc_t - running_acc_val) / (batch_index + 1)

                    if (comet_value_updater is not None):
                        comet_value_updater.log_metric("running_acc_dev_per_batch", running_acc_val, step=batch_index)

                    LOG.info(
                        f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches_lex} \t per_batch_accuracy_dev_set:{round(acc_t,4)} \t moving_avg_val_accuracy:{round(running_acc_val,4)} ")

                train_state_in['val_loss'].append(running_loss_val)
                train_state_in['val_acc'].append(running_acc_val)

                if (comet_value_updater is not None):
                    comet_value_updater.log_metric("running_acc_dev_per_epoch", running_acc_val, step=epoch_index)

                train_state_in = self.update_train_state(args=args_in, model=classifier_student2,
                                                         train_state=train_state_in)



                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state_in['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                LOG.info(
                    f" val_accuracy_end_of_epoch:{round(running_acc_val,2)} ")
                time.sleep(10)
                

        except KeyboardInterrupt:
            print("Exiting loop")



