from mean_teacher.utils.utils_rao import generate_batches
from mean_teacher.modules.rao_datasets import RTEDataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm_notebook

class Trainer:
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


    def train(self, args_in,classifier,dataset):
        classifier = classifier.to(args_in.device)

        loss_func = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=args_in.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)

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
                batch_generator = generate_batches(dataset,
                                                   batch_size=args_in.batch_size,
                                                   device=args_in.device)
                running_loss = 0.0
                running_acc = 0.0
                classifier.train()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    optimizer.zero_grad()

                    # step 2. compute the output
                    y_pred = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])

                    # step 3. compute the loss
                    loss = loss_func(y_pred, batch_dict['y_target'].float())
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # step 4. use loss to produce gradients
                    loss.backward()

                    # step 5. use optimizer to take gradient step
                    optimizer.step()
                    # -----------------------------------------
                    # compute the accuracy
                    acc_t = self.compute_accuracy(y_pred, batch_dict['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    # update bar
                    train_bar.set_postfix(loss=running_loss,
                                          acc=running_acc,
                                          epoch=epoch_index)
                    train_bar.update()

                train_state_in['train_loss'].append(running_loss)
                train_state_in['train_acc'].append(running_acc)

                # Iterate over val dataset

                # setup: batch generator, set loss and acc to 0; set eval mode on
                dataset.set_split('val')
                batch_generator = generate_batches(dataset,
                                                   batch_size=args_in.batch_size,
                                                   device=args_in.device)
                running_loss = 0.
                running_acc = 0.
                classifier.eval()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # compute the output
                    y_pred = classifier(x_in=batch_dict['x_data'].float())

                    # step 3. compute the loss
                    loss = loss_func(y_pred, batch_dict['y_target'].float())
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # compute the accuracy
                    acc_t = self.compute_accuracy(y_pred, batch_dict['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    val_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        epoch=epoch_index)
                    val_bar.update()

                train_state_in['val_loss'].append(running_loss)
                train_state_in['val_acc'].append(running_acc)

                train_state_in = self.update_train_state( args=args_in, model=classifier,
                                                      train_state=train_state_in)

                scheduler.step(train_state_in['val_loss'][-1])

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state_in['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                print(f"epoch:{epoch_index} val loss: {running_loss} val Accuracy: {running_acc} ")
                

        except KeyboardInterrupt:
            print("Exiting loop")



        # uncomment to compute the loss & accuracy on the test set using the best available model
        #
        classifier.load_state_dict(torch.load(train_state_in['model_filename']))
        classifier = classifier.to(args_in.device)

        dataset.set_split('val')
        batch_generator = generate_batches(dataset,
                                           batch_size=args_in.batch_size,
                                           device=args_in.device)
        running_loss = 0.
        running_acc = 0.
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred = classifier(x_in=batch_dict['x_data'].float())

            # compute the loss
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = self.compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        train_state_in['test_loss'] = running_loss
        train_state_in['test_acc'] = running_acc
        print("Val loss at end of all epochs: {:.3f}".format(train_state_in['test_loss']))
        print("Val accuracy at end of all epochs: {:.2f}".format(train_state_in['test_acc']))

