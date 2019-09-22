from mean_teacher.utils.utils_rao import generate_batches,initialize_double_optimizers,update_optimizer_state

import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm,tqdm_notebook
from torch.nn import functional as F
from mean_teacher.utils.logger import Logger

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

    def update_train_state(self,args, model, train_state):
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
            torch.save(model.state_dict(), train_state['model_filename'])
            train_state['stop_early'] = False

        # Save model if performance improved
        elif train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= train_state['early_stopping_best_val']:
                # Update step
                train_state['early_stopping_step'] += 1
            # Loss decreased
            else:
                # Save the best model
                if loss_t < train_state['early_stopping_best_val']:
                    torch.save(model.state_dict(), train_state['model_filename'])

                # Reset early stopping step
                train_state['early_stopping_step'] = 0

            # Stop early ?
            train_state['stop_early'] = \
                train_state['early_stopping_step'] >= args.early_stopping_criteria

        return train_state

    def compute_accuracy(self,y_pred, y_target):
        y_target = y_target.cpu()
        y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()  # .max(dim=1)[1]
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    def get_learning_rate(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def calculate_argmax_list(self, logit):
        list_labels_pred = []
        for tensor in logit:
            values, indices = torch.max(tensor, 0)
            list_labels_pred.append(indices.data.item())
        return list_labels_pred

    def train(self, args_in,classifier,dataset):
        classifier = classifier.to(args_in.device)

        if torch.cuda.is_available():
            class_loss_func = nn.CrossEntropyLoss(size_average=False).cuda()
            #todo: use this code below instead when doing semi supervised :
            # class_loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
        else:
            class_loss_func = nn.CrossEntropyLoss(size_average=False).cpu()

        #optimizer = optim.Adam(classifier.parameters(), lr=args_in.learning_rate)
        input_optimizer, inter_atten_optimizer = initialize_double_optimizers(classifier, args_in)

        self._LOG.debug(f"going to get into ReduceLROnPlateau ")
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer=input_optimizer,mode='min', factor=0.5,patience=1)
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer=inter_atten_optimizer, mode='min', factor=0.5, patience=1)




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
                batch_generator1 = generate_batches(dataset,
                                                   batch_size=args_in.batch_size,
                                                   device=args_in.device)
                no_of_batches_lex = int(len(dataset) / args_in.batch_size)

                dataset.set_split('train_delex')

                batch_generator2 = generate_batches(dataset,
                                                    batch_size=args_in.batch_size,
                                                    device=args_in.device)
                no_of_batches_delex = int(len(dataset) / args_in.batch_size)

                running_loss_lex = 0.0
                running_acc_lex = 0.0
                running_loss_delex = 0.0
                running_acc_delex = 0.0
                classifier.train()





                for batch_index, (batch_dict_lex,batch_dict_delex) in enumerate(zip(batch_generator1,batch_generator2)):

                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    input_optimizer.zero_grad()
                    inter_atten_optimizer.zero_grad()

                    #this code is from the libowen code base we are using for decomposable attention
                    if epoch_index == 0 and args_in.optimizer == 'adagrad':
                        update_optimizer_state(input_optimizer, inter_atten_optimizer, args_in)



                    # step 2. compute the output
                    y_pred_lex = classifier(batch_dict_lex['x_claim'], batch_dict_lex['x_evidence'])
                    y_pred_delex = classifier(batch_dict_delex['x_claim'], batch_dict_delex['x_evidence'])

                    # step 3.1 compute the class_loss_lex
                    class_loss_lex = class_loss_func(y_pred_lex, batch_dict_lex['y_target'])
                    loss_t_lex = class_loss_lex.item()
                    running_loss_lex += (loss_t_lex - running_loss_lex) / (batch_index + 1)

                    # step 3.1 compute the class_loss_delex
                    class_loss_delex = class_loss_func(y_pred_delex, batch_dict_delex['y_target'])
                    loss_t_delex = class_loss_delex.item()
                    running_loss_delex += (loss_t_delex - running_loss_delex) / (batch_index + 1)

                    # step 4. use combined classification loss to produce gradients
                    combined_class_loss=class_loss_lex+class_loss_delex
                    combined_class_loss.backward()

                    #step 4.5 this is specific to decomposable attention
                    # grad_norm = 0.
                    # para_norm = 0.
                    # for m in classifier.input_encoder.modules():
                    #     if isinstance(m, nn.Linear):
                    #         grad_norm += m.weight.grad.data.norm() ** 2
                    #         para_norm += m.weight.data.norm() ** 2
                    #         if m.bias:
                    #             grad_norm += m.bias.grad.data.norm() ** 2
                    #             para_norm += m.bias.data.norm() ** 2
                    #
                    # for m in classifier.inter_atten.modules():
                    #     if isinstance(m, nn.Linear):
                    #         grad_norm += m.weight.grad.data.norm() ** 2
                    #         para_norm += m.weight.data.norm() ** 2
                    #         if m.bias is not None:
                    #             grad_norm += m.bias.grad.data.norm() ** 2
                    #             para_norm += m.bias.data.norm() ** 2
                    #
                    # shrinkage = args_in.max_grad_norm / grad_norm
                    # if shrinkage < 1:
                    #     for m in classifier.input_encoder.modules():
                    #         if isinstance(m, nn.Linear):
                    #             m.weight.grad.data = m.weight.grad.data * shrinkage
                    #     for m in classifier.inter_atten.modules():
                    #         if isinstance(m, nn.Linear):
                    #             m.weight.grad.data = m.weight.grad.data * shrinkage
                    #             m.bias.grad.data = m.bias.grad.data * shrinkage


                    # step 5. use optimizer to take gradient step
                    #optimizer.step()
                    input_optimizer.step()
                    inter_atten_optimizer.step()

                    # -----------------------------------------



                    # compute the accuracy for lex data
                    y_pred_labels_lex=self.calculate_argmax_list(y_pred_lex)
                    y_pred_labels_lex = torch.FloatTensor(y_pred_labels_lex)
                    acc_t_lex = self.compute_accuracy(y_pred_labels_lex, batch_dict_lex['y_target'])
                    running_acc_lex += (acc_t_lex - running_acc_lex) / (batch_index + 1)

                    # compute the accuracy for delex data
                    y_pred_labels_delex = self.calculate_argmax_list(y_pred_delex)
                    y_pred_labels_delex = torch.FloatTensor(y_pred_labels_delex)
                    acc_t_delex = self.compute_accuracy(y_pred_labels_delex, batch_dict_delex['y_target'])
                    running_acc_delex += (acc_t_delex - running_acc_delex) / (batch_index + 1)

                    # update bar
                    train_bar.set_postfix(loss=running_loss_lex,
                                          acc=running_acc_lex,
                                          epoch=epoch_index)
                    train_bar.update()
                    self._LOG.info(f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches_lex} \t moving_avg_train_loss:{round(running_loss_lex,2)} \t moving_avg_train_accuracy:{round(running_acc_lex,2)} ")

                lr = self.get_learning_rate(input_optimizer)
                self._LOG.debug(f"value of learning rate now  for input_optimizer is:{lr}")
                lr = self.get_learning_rate(inter_atten_optimizer)
                self._LOG.debug(f"value of learning rate now  for inter_atten_optimizer is:{lr}")

                train_state_in['train_loss'].append(running_loss_lex)
                train_state_in['train_acc'].append(running_acc_lex)

                # Iterate over val dataset

                # setup: batch generator, set class_loss_lex and acc to 0; set eval mode on
                dataset.set_split('val')
                batch_generator1 = generate_batches(dataset,
                                                   batch_size=args_in.batch_size,
                                                   device=args_in.device)
                running_loss_lex = 0.
                running_acc_lex = 0.
                classifier.eval()
                no_of_batches_lex = int(len(dataset) / args_in.batch_size)

                for batch_index, batch_dict_lex in enumerate(batch_generator1):
                    # compute the output
                    y_pred_lex = classifier(batch_dict_lex['x_claim'], batch_dict_lex['x_evidence'])

                    # step 3. compute the class_loss_lex
                    class_loss_lex = class_loss_func(y_pred_lex, batch_dict_lex['y_target'])
                    loss_t_lex = class_loss_lex.item()
                    running_loss_lex += (loss_t_lex - running_loss_lex) / (batch_index + 1)

                    # compute the accuracy
                    y_pred_labels_lex = self.calculate_argmax_list(y_pred_lex)
                    y_pred_labels_lex = torch.FloatTensor(y_pred_labels_lex)
                    acc_t_lex = self.compute_accuracy(y_pred_labels_lex, batch_dict_lex['y_target'])
                    running_acc_lex += (acc_t_lex - running_acc_lex) / (batch_index + 1)

                    val_bar.set_postfix(loss=running_loss_lex,
                                        acc=running_acc_lex,
                                        epoch=epoch_index)
                    val_bar.update()
                    self._LOG.info(
                        f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches_lex} \t moving_avg_val_loss:{round(running_loss_lex,2)} \t moving_avg_val_accuracy:{round(running_acc_lex,2)} ")

                train_state_in['val_loss'].append(running_loss_lex)
                train_state_in['val_acc'].append(running_acc_lex)

                train_state_in = self.update_train_state( args=args_in, model=classifier,
                                                      train_state=train_state_in)

                scheduler1.step(train_state_in['val_loss'][-1])
                scheduler2.step(train_state_in['val_loss'][-1])

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state_in['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                self._LOG.info(f"epoch:{epoch_index}\tval_loss_end_of_epoch:{round(running_loss_lex,4)}\tval_accuracy_end_of_epoch:{round(running_acc_lex,4)} ")
                time.sleep(10)
                

        except KeyboardInterrupt:
            print("Exiting loop")



        # uncomment to compute the class_loss_lex & accuracy on the test set using the best available model
        #
        # classifier.load_state_dict(torch.load(train_state_in['model_filename']))
        # classifier = classifier.to(args_in.device)
        #
        # dataset.set_split('test')
        # batch_generator1 = generate_batches(dataset,
        #                                    batch_size=args_in.batch_size,
        #                                    device=args_in.device)
        # running_loss_lex = 0.
        # running_acc_lex = 0.
        # classifier.eval()
        #
        # for batch_index, batch_dict in enumerate(batch_generator1):
        #     # compute the output
        #     y_pred_lex = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])
        #
        #
        #     # compute the class_loss_lex
        #     class_loss_lex = class_loss_func(y_pred_lex, batch_dict['y_target'].float())
        #     loss_t_lex = class_loss_lex.item()
        #     running_loss_lex += (loss_t_lex - running_loss_lex) / (batch_index + 1)
        #
        #     # compute the accuracy
        #     acc_t_lex = self.compute_accuracy(y_pred_lex, batch_dict['y_target'])
        #     running_acc_lex += (acc_t_lex - running_acc_lex) / (batch_index + 1)
        #train_state_in['test_loss'] = running_loss_lex
        #train_state_in['test_acc'] = running_acc_lex
        self._LOG.info(f"{self._current_time:}Val class_loss_lex at end of all epochs: {(train_state_in['val_loss'])}")
        self._LOG.info(f"{self._current_time:}Val accuracy at end of all epochs: {(train_state_in['val_acc'])}")

