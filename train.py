import tensorflow as tf
import numpy as np
import time
import datetime
import model
import get_data
import config
from sklearn.metrics import average_precision_score, roc_auc_score
from absl import flags, app
# New import
import tf_slim as slim
import os
FLAGS = flags.FLAGS

def make_summary(name, value, writer, step):
    """Logs a scalar summary using TensorFlow 2.x summary API."""
    """Creates a tf.Summary proto with the given name and value."""
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)

def train_step(hg, input_label, input_nlcd, optimizer, step, writer):
    
    # Perform training step
    with tf.GradientTape() as tape:
        # Ensure the model's trainable variables are watched by the tape
        tape.watch(hg.trainable_variables)

        # Get predictions and losses from the model
        nll_loss, l2_loss, total_loss = hg(is_training=True)

        # Ensure that total_loss is a scalar value for gradient computation
        total_loss = tf.reduce_mean(total_loss)  # This ensures it's a scalar

        # Print intermediate losses for debugging
        print(f"NLL Loss: {nll_loss}, L2 Loss: {l2_loss}, Total Loss: {total_loss}")
        
        # Compute gradients based on total loss
        gradients = tape.gradient(total_loss, hg.trainable_variables)

        # Apply gradients to the model using the optimizer
        optimizer.apply_gradients(zip(gradients, hg.trainable_variables))

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar("train/nll_loss", nll_loss, step=step)
            tf.summary.scalar("train/l2_loss", l2_loss, step=step)
            tf.summary.scalar("train/total_loss", total_loss, step=step)

        # Calculate saliency map for explainability
        saliency_map = tape.gradient(total_loss, input_nlcd)

        # Return the individual predictions and the losses
        return  nll_loss, l2_loss, total_loss, saliency_map


def validation_step(hg, data, valid_idx, writer, step):
    print("Validating...")

    all_nll_loss, all_l2_loss, all_total_loss, all_marginal_loss = 0, 0, 0, 0
    all_indiv_prob, all_label = [], []

    for i in range(0, len(valid_idx), FLAGS.batch_size):
        batch_indices = valid_idx[i:i + FLAGS.batch_size]

        input_nlcd = get_data.get_nlcd(data, batch_indices)
        input_label = get_data.get_label(data, batch_indices)

        # Forward pass (no gradient calculation during validation)
        predictions = hg(input_label, is_training=False)
        nll_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(input_label, predictions))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in hg.trainable_variables])

        # Assuming marginal loss is computed separately (replace with actual computation)
        marginal_loss = tf.reduce_mean(tf.abs(input_label - predictions))

        total_loss = nll_loss + l2_loss + marginal_loss

        # Aggregate results
        all_nll_loss += nll_loss.numpy() * len(batch_indices)
        all_l2_loss += l2_loss.numpy() * len(batch_indices)
        all_total_loss += total_loss.numpy() * len(batch_indices)
        all_marginal_loss += marginal_loss.numpy() * len(batch_indices)

        all_indiv_prob.append(predictions.numpy())
        all_label.append(input_label)

    all_indiv_prob = np.concatenate(all_indiv_prob).flatten()
    all_label = np.concatenate(all_label).flatten()
    # Convert accumulated results to numpy arrays
    # Compute AP and handle potential AUC computation issues
    ap = average_precision_score(all_label, all_indiv_prob)
    try:
        auc = roc_auc_score(all_label, all_indiv_prob)
    except ValueError:
        auc = 0.0

    # Compute mean loss values
    mean_nll_loss = all_nll_loss / len(valid_idx)
    mean_l2_loss = all_l2_loss / len(valid_idx)
    mean_total_loss = all_total_loss / len(valid_idx)
    mean_marginal_loss = all_marginal_loss / len(valid_idx)

    print(f"Validation Results - AUC: {auc:.6f}, AP: {ap:.6f}, "
          f"NLL Loss: {mean_nll_loss:.6f}, Marginal Loss: {mean_marginal_loss:.6f}, "
          f"L2 Loss: {mean_l2_loss:.6f}, Total Loss: {mean_total_loss:.6f}")

    # Log results to TensorBoard
    with writer.as_default():
        tf.summary.scalar("validation/auc", auc, step=step)
        tf.summary.scalar("validation/ap", ap, step=step)
        tf.summary.scalar("validation/nll_loss", mean_nll_loss, step=step)
        tf.summary.scalar("validation/l2_loss", mean_l2_loss, step=step)
        tf.summary.scalar("validation/total_loss", mean_total_loss, step=step)

    return mean_nll_loss



def main(_):
    st_time = time.time()
    print('Reading npy...')
    np.random.seed(19950420)
    
    data = np.load(FLAGS.data_dir, allow_pickle=True)
    train_idx = np.load(FLAGS.train_idx)

    if not os.path.exists(FLAGS.train_idx):
        print(f"File not found: {FLAGS.train_idx}")

    valid_idx = np.load(FLAGS.valid_idx)
    labels = get_data.get_label(data, train_idx)
    print("label:", labels)
    print("Positive label rate:", np.mean(labels))

    one_epoch_iter = len(train_idx) // FLAGS.batch_size
    print('Reading completed')
    #config the tesorflow
    # GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #building the model
    print('Building model...')
    hg = model.MODEL(is_training=True)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Learning rate schedule
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )
    #use the Adam optimizer 
    optimizer = tf.keras.optimizers.Adam(learning_rate_schedule)
    # Prepare checkpointing
    checkpoint_dir = FLAGS.model_dir
    checkpoint_prefix = checkpoint_dir + "/ckpt"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=hg)
    # Summary writer for TensorBoard
    #log the learning rate 
    summary_writer = tf.summary.create_file_writer(FLAGS.summary_dir)

    best_loss = float('inf')
    current_step = 0  

    for one_epoch in range(FLAGS.max_epoch):
        print(f'Epoch {one_epoch + 1} starts!')

        np.random.shuffle(train_idx)

        smooth_nll_loss = 0.0
        smooth_l2_loss = 0.0
        smooth_total_loss = 0.0

        temp_label = []
        temp_indiv_prob = []

        for i in range(one_epoch_iter):
            start = i * FLAGS.batch_size
            end = start + FLAGS.batch_size

            input_nlcd = get_data.get_nlcd(data, train_idx[start:end])
            input_label = get_data.get_label(data, train_idx[start:end])

            # Perform training step
            with tf.GradientTape() as tape:
                # Pass inputs to the model to get predictions and losses
                nll_loss, l2_loss, total_loss = hg(is_training=False)
                print("Total loss:", total_loss)  # Print total_loss to inspect

            gradients = tape.gradient(total_loss, hg.trainable_variables)
            optimizer.apply_gradients(zip(gradients, hg.trainable_variables))

            smooth_nll_loss += nll_loss.numpy()
            smooth_l2_loss += l2_loss.numpy()
            smooth_total_loss += total_loss.numpy()

            temp_label.append(input_label)
            temp_indiv_prob.append(indiv_prob.numpy())

            # Log progress
            if (i + 1) % FLAGS.check_freq == 0:
                mean_nll_loss = smooth_nll_loss / FLAGS.check_freq
                mean_l2_loss = smooth_l2_loss / FLAGS.check_freq
                mean_total_loss = smooth_total_loss / FLAGS.check_freq

                temp_indiv_prob = np.reshape(np.array(temp_indiv_prob), (-1))
                temp_label = np.reshape(np.array(temp_label), (-1))
                ap = average_precision_score(temp_label, temp_indiv_prob)

                time_str = datetime.datetime.now().isoformat()
                print(f"{time_str}\tEpoch={one_epoch+1}\tAP={ap:.6f}\tNLL Loss={mean_nll_loss:.6f}\tL2 Loss={mean_l2_loss:.6f}\tTotal Loss={mean_total_loss:.6f}")

                with summary_writer.as_default():
                    tf.summary.scalar('train/ap', ap, step=one_epoch)
                    tf.summary.scalar('train/nll_loss', mean_nll_loss, step=one_epoch)
                    tf.summary.scalar('train/l2_loss', mean_l2_loss, step=one_epoch)

                temp_indiv_prob = []
                temp_label = []

                smooth_nll_loss = 0.0
                smooth_l2_loss = 0.0
                smooth_total_loss = 0.0

        # Validation step
        mean_nll_loss = validation_step(hg, data, valid_idx)

        if mean_nll_loss < best_loss:
            print(f"New best loss: {mean_nll_loss:.6f}, saving model...")
            best_loss = mean_nll_loss
            best_iter = one_epoch + 1
            checkpoint.save(file_prefix=checkpoint_prefix)

    print('Training completed!')
    print(f'Best loss on validation: {best_loss}')
    print(f'Best epoch: {best_iter}')
    ed_time = time.time()
    print("Total running time:", ed_time - st_time)

if __name__ == '__main__':
    app.run(main)