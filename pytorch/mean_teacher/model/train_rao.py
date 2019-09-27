from mean_teacher.utils.utils_rao import generate_batches,initialize_double_optimizers,update_optimizer_state
from mean_teacher.modules.rao_datasets import RTEDataset
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm,tqdm_notebook
from torch.nn import functional as F
from mean_teacher.utils.logger import LOG

class Trainer():
    def __init__(self):
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

    def accuracy_fever(self,predicted_labels, gold_labels):
        m = nn.Softmax()
        output_sftmax = m(predicted_labels)
        NO_LABEL = -1
        labeled_minibatch_size = max(gold_labels.ne(NO_LABEL).sum(), 1e-8)
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

        LOG.debug(f"going to get into ReduceLROnPlateau ")
        #scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer=input_optimizer,mode='min', factor=0.5,patience=1)
        #scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer=inter_atten_optimizer, mode='min', factor=0.5, patience=1)




        train_state_in = self.make_train_state(args_in)

        epoch_bar = tqdm_notebook(desc='training routine',
                                  total=args_in.num_epochs,
                                  position=0)

        dataset.set_split('train')
        train_bar = tqdm_notebook(desc='split=train',
                                  total=dataset.get_num_batches(args_in.batch_size),
                                  position=1,
                                  leave=True)
        dataset.set_split('val')
        val_bar = tqdm_notebook(desc='split=val',
                                total=dataset.get_num_batches(args_in.batch_size),
                                position=1,
                                leave=True)


        try:
            for epoch_index in range(args_in.num_epochs):
                train_state_in['epoch_index'] = epoch_index

                # Iterate over training dataset

                # setup: batch generator, set loss and acc to 0, set train mode on
                dataset.set_split('train')
                #(dataset, batch_size, shuffle=True,
                     #drop_last=True, device="cpu",workers=1):
                #
                batch_generator1 = generate_batches(dataset,workers=args_in.workers,batch_size=args_in.batch_size,device=args_in.device)


                running_loss = 0.0
                running_acc = 0.0
                classifier.train()
                no_of_batches= int(len(dataset)/args_in.batch_size)




                for batch_index, batch_dict1 in enumerate(batch_generator1):

                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    input_optimizer.zero_grad()
                    inter_atten_optimizer.zero_grad()

                    #this code is from the libowen code base we are using for decomposable attention
                    if epoch_index == 0 and args_in.optimizer == 'adagrad':
                        update_optimizer_state(input_optimizer, inter_atten_optimizer, args_in)



                    # step 2. compute the output
                    y_pred = classifier(batch_dict1['x_claim'], batch_dict1['x_evidence'])

                    # step 3. compute the loss
                    loss = class_loss_func(y_pred, batch_dict1['y_target'])
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # step 4. use loss to produce gradients
                    loss.backward()

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
                    # compute the accuracy
                    y_pred_labels=self.calculate_argmax_list(y_pred)
                    y_pred_labels = torch.FloatTensor(y_pred_labels)
                    acc_t = self.compute_accuracy(y_pred_labels, batch_dict1['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    # update bar
                    train_bar.set_postfix(loss=running_loss,
                                          acc=running_acc,
                                          epoch=epoch_index)
                    train_bar.update()
                    LOG.info(f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches} \t moving_avg_train_loss:{round(running_loss,2)} \t moving_avg_train_accuracy:{round(running_acc,2)} ")

                lr = self.get_learning_rate(input_optimizer)
                LOG.debug(f"value of learning rate now  for input_optimizer is:{lr}")
                lr = self.get_learning_rate(inter_atten_optimizer)
                LOG.debug(f"value of learning rate now  for inter_atten_optimizer is:{lr}")

                train_state_in['train_loss'].append(running_loss)
                train_state_in['train_acc'].append(running_acc)

                # Iterate over val dataset

                # setup: batch generator, set loss and acc to 0; set eval mode on
                dataset.set_split('val')
                # batch_generator1 = generate_batches(dataset,
                #                                    batch_size=args_in.batch_size,
                #                                    device=args_in.device)

                batch_generator1 = generate_batches(dataset,workers=args_in.workers,batch_size=args_in.batch_size,device=args_in.device,shuffle=False)

                running_loss = 0.
                running_acc = 0.
                classifier.eval()
                no_of_batches = int(len(dataset) / args_in.batch_size)

                for batch_index, batch_dict1 in enumerate(batch_generator1):
                    # compute the output
                    y_pred = classifier(batch_dict1['x_claim'], batch_dict1['x_evidence'])

                    # step 3. compute the loss
                    loss = class_loss_func(y_pred, batch_dict1['y_target'])
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # compute the accuracy
                    y_pred_labels = self.calculate_argmax_list(y_pred)
                    y_pred_labels = torch.FloatTensor(y_pred_labels)
                    acc_t = self.compute_accuracy(y_pred_labels, batch_dict1['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    val_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        epoch=epoch_index)
                    val_bar.update()
                    LOG.info(
                        f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches} \t moving_avg_val_loss:{round(running_loss,2)} \t moving_avg_val_accuracy:{round(running_acc,2)} ")

                train_state_in['val_loss'].append(running_loss)
                train_state_in['val_acc'].append(running_acc)

                train_state_in = self.update_train_state( args=args_in, model=classifier,
                                                      train_state=train_state_in)

                #scheduler1.step(train_state_in['val_loss'][-1])
                #scheduler2.step(train_state_in['val_loss'][-1])

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state_in['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                LOG.info(f"epoch:{epoch_index}\tval_loss_end_of_epoch:{round(running_loss,4)}\tval_accuracy_end_of_epoch:{round(running_acc,4)} ")
                time.sleep(10)
                

        except KeyboardInterrupt:
            print("Exiting loop")



        # uncomment to compute the loss & accuracy on the test set using the best available model
        #
        # classifier.load_state_dict(torch.load(train_state_in['model_filename']))
        # classifier = classifier.to(args_in.device)
        #
        # dataset.set_split('test')
        # batch_generator1 = generate_batches(dataset,
        #                                    batch_size=args_in.batch_size,
        #                                    device=args_in.device)
        # running_loss = 0.
        # running_acc = 0.
        # classifier.eval()
        #
        # for batch_index, batch_dict in enumerate(batch_generator1):
        #     # compute the output
        #     y_pred = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])
        #
        #
        #     # compute the loss
        #     loss = class_loss_func(y_pred, batch_dict['y_target'].float())
        #     loss_t = loss.item()
        #     running_loss += (loss_t - running_loss) / (batch_index + 1)
        #
        #     # compute the accuracy
        #     acc_t = self.compute_accuracy(y_pred, batch_dict['y_target'])
        #     running_acc += (acc_t - running_acc) / (batch_index + 1)
        #train_state_in['test_loss'] = running_loss
        #train_state_in['test_acc'] = running_acc
        LOG.info(f"{self._current_time:}Val loss at end of all epochs: {(train_state_in['val_loss'])}")
        LOG.info(f"{self._current_time:}Val accuracy at end of all epochs: {(train_state_in['val_acc'])}")

