import gin
import tensorflow as tf
import logging
import sys
import os
import datetime
import wandb
from utils import utils_params


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, resume):
        # Summary Writer
        #run_paths = utils_params.gen_run_folder()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = run_paths['path_model_id'] + '/logs/gradient_tape/' + current_time + '/train'
        val_log_dir = run_paths['path_model_id'] + '/logs/gradient_tape/' + current_time + '/val'
        gradient_log_dir = run_paths['path_model_id'] + '/logs/gradient_tape/' + current_time + '/gradient'
        lr_log_dir = run_paths['path_model_id'] + '/logs/gradient_tape/' + current_time + '/lr'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        self.gradient_summary_writer = tf.summary.create_file_writer(gradient_log_dir)
        self.lr_summary_writer = tf.summary.create_file_writer(lr_log_dir)

        # resume train flag
        # the path should be changed , this is only test path
        self.resume = resume
        self.resume_dir = run_paths['path_model_id'] + '/ckpts'

        # Checkpoint Manager Checkpoint Path
        # save the checkpoint max checkpoint save nummer is 20
        self.checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model)
        checkpoint_dir = run_paths['path_model_id'] + '/ckpts'
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=checkpoint_dir, max_to_keep=20)
        self.best_ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=checkpoint_dir, max_to_keep=20,
                                                            checkpoint_name='best')

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.val_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # dynamic learning rate
        # for vgg 0.001 seems to be too low
        #
        initial_lr = 0.001
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=100,
            decay_rate=0.95,
            staircase=True
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # inital lr to see the adam optimizer
        # self.optimizer = tf.keras.optimizers.Adam()

        # early stopping
        # monitor : validation_loss value
        # patience: when X epochs the monitor not better, stop training
        # restore the best_weights
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Metric the AUC Curve
        self.AUC = tf.keras.metrics.AUC(name='AUC')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval


    #@tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

        # print the loss and accurancy
        sys.stdout.write('\r' + 'Train Loss:' + str(loss) + ' Acccurancy ' + str(self.train_accuracy(labels, predictions)))
        sys.stdout.flush()


        # Write gradient summary to Tensroboard
        with self.gradient_summary_writer.as_default():
            for i, grad in enumerate(gradients):
                tf.summary.histogram(f"Layer_{i}_gradient", grad, step=0)

        """
        # print the gradient for every layer
        for var, grad in zip(self.model.trainable_variables, gradients):
            if grad is not None:  # check if the gradient exist
                grad_mean = tf.reduce_mean(grad)
                grad_stddev = tf.math.reduce_std(grad)
                print(f"Layer: {var.name}, Gradient Mean: {grad_mean:.6f}, Gradient StdDev: {grad_stddev:.6f}")
            else:
                print(f"Layer: {var.name}, Gradient is None")
        """



    # @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # why here the prediction is negative values? with this code # predictions = self.model(images, training=False)
        predictions = self.model(images, training=False)
        t_loss = self.val_loss_object(labels, predictions)
        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)




    def train(self):
        best_val_loss = 9999
        best_accuracy = 0
        patience_count = 0

        # resume the training from checkpoints
        if self.resume:
            self.checkpoint.restore(self.resume_dir)
            print("Resume the from the checkpoint")

        # training step
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            # visualization the train steps
            print("\r", end="")
            print("Training progress: {}%: ".format(step//50), "▋" * (step // 50), end="")
            sys.stdout.flush()

            if step % self.log_interval == 0:

                # Reset test metrics
                #self.val_loss.reset_states()
                #self.val_accuracy.reset_states()
                self.val_loss.reset_state()
                self.val_accuracy.reset_state()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)
                    #predictions_output = self.model(val_images)
                    # predictions_output = tf.argmax(predictions_output, axis=1)
                    # print(f"Predicted: {predictions_output}")

                print('\n')
                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # Write train summary to tensorboard
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss',  self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)

                # Write validation summary to tensorboard
                with self.val_summary_writer.as_default():
                    tf.summary.scalar('loss',  self.val_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.val_accuracy.result(), step=step)

                # Write lr summary to tensorboard
                current_lr = self.lr_schedule(step) if callable(
                    self.optimizer.learning_rate) else self.optimizer.learning_rate.numpy()

                # Log to TensorBoard
                with self.train_summary_writer.as_default():
                    tf.summary.scalar("loss", self.train_loss.result(), step=step)
                    tf.summary.scalar("learning_rate", current_lr, step=step)

                # wandb logging
                #wandb.log({'train_acc': self.train_accuracy.result() * 100, 'train_loss': self.train_loss.result(),
                #           'val_acc': self.val_accuracy.result() * 100, 'val_loss': self.val_loss.result(),
                #           'step': step})

                # early Stopping , val_loss
                """
                if self.val_loss.result() < best_val_loss:
                    best_val_loss = self.val_loss.result()
                    # Save the best checkpoint
                    logging.info('save the best checkpoint')
                    self.best_ckpt_manager.save()
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.early_stopping.patience:
                        print(f"Stopping early at steps {step}")
                        break
                """
                # save the best weight!
                if self.val_loss.result() < best_val_loss:
                    best_val_loss = self.val_loss.result()
                    #self.model.save_weights('/home/kusabi/DLLAB/dl-lab-24w-team06/'+str(step)+ 'best.weights.h5')
                    self.model.save_weights(os.getcwd() + '/weight/' + str(step) + 'best.weights.h5')

                if self.val_accuracy.result() > best_accuracy:
                    best_accuracy = self.val_accuracy.result()
                    self.model.save_weights(os.getcwd() + '/weight/' + str(step) + 'best_accu.weights.h5')

                # Reset train metrics
                # self.train_loss.reset_states()
                # self.train_accuracy.reset_states()
                self.train_loss.reset_state()
                self.train_accuracy.reset_state()
                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                #close the summary writer
                self.train_summary_writer.close()
                # Save final checkpoint
                # ...
                return self.val_accuracy.result().numpy()
