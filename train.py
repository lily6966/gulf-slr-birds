import tensorflow as tf
import numpy as np
import time
import datetime
import model
import get_data
import config
from sklearn.metrics import average_precision_score, roc_auc_score
from absl import flags, app
import os
FLAGS = flags.FLAGS



def make_summary(name, value, writer, step):
    """Logs a scalar summary using TensorFlow 2.x summary API."""
    step = tf.convert_to_tensor(step, dtype=tf.int64)  # Ensure step is an int64 tensor
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)

def train_step(hg, input_nlcd, input_label, optimizer, step, writer):
    """Performs a training step, computes gradients, and logs metrics."""

    with tf.GradientTape() as tape:
        # Forward pass with inputs
        indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = hg(
            [input_nlcd, input_label], is_training=True
        )

    # Compute gradients
    gradients = tape.gradient(total_loss, hg.trainable_variables)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, hg.trainable_variables))

    # Convert losses to scalars for easy debugging
    indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = indiv_prob.numpy(), nll_loss.numpy(), marginal_loss.numpy(), l2_loss.numpy(), total_loss.numpy()

    # Log metrics
    with writer.as_default():
        tf.summary.scalar("train/nll_loss", nll_loss, step=step)
        tf.summary.scalar("train/marginal_loss", nll_loss, step=step)
        tf.summary.scalar("train/l2_loss", l2_loss, step=step)
        tf.summary.scalar("train/total_loss", total_loss, step=step)

    return indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss



def validation_step(hg, data, valid_idx, writer, step):
    """Performs one validation step and logs the results."""
    print("Validating...")

    all_nll_loss, all_marginal_loss, all_l2_loss, all_total_loss = 0, 0, 0, 0
    all_indiv_prob, all_label = [], []

    for i in range(0, len(valid_idx)-(len(valid_idx) % FLAGS.batch_size), FLAGS.batch_size):
        batch_indices = valid_idx[i:i + FLAGS.batch_size]

        input_nlcd = get_data.get_nlcd(data, batch_indices)
        input_label = get_data.get_label(data, batch_indices)

        # Forward pass (no gradient calculation during validation)
        indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = hg([input_nlcd, input_label], is_training=False)

        # Aggregate results
        all_nll_loss += nll_loss * len(batch_indices)
        all_l2_loss += l2_loss * len(batch_indices)
        all_total_loss += total_loss * len(batch_indices)
        all_marginal_loss += marginal_loss * len(batch_indices)   
        for ii in indiv_prob:
            print(ii.shape)
            all_indiv_prob.append(ii)
        for ii in input_label:
            all_label.append(ii)
   
    all_indiv_prob = np.array(all_indiv_prob)
    all_label = np.array(all_label)         
    
    # Compute average metrics
    mean_nll_loss = all_nll_loss / len(valid_idx)
    mean_l2_loss = all_l2_loss / len(valid_idx)
    mean_total_loss = all_total_loss / len(valid_idx)
    mean_marginal_loss = all_marginal_loss / len(valid_idx)

    # Compute average precision and AUC

    all_indiv_prob = np.concatenate(all_indiv_prob).flatten()
    all_label = np.concatenate(all_label).flatten()
    ap = average_precision_score(all_label, all_indiv_prob)
    
                
    try:
        auc = roc_auc_score(all_label, all_indiv_prob)
    except ValueError:
        auc = 0.0

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
    valid_idx = np.load(FLAGS.valid_idx)

    labels = get_data.get_label(data, train_idx)
    print("Label distribution: ", np.mean(labels))

    one_epoch_iter = len(train_idx) // FLAGS.batch_size
    print('Reading completed')

    # GPU memory configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Building the model
    print('Building model...')
    hg = model.MODEL(is_training=True)
    
    # Learning rate schedule
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )

    # Optimizer setup
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    global_step = optimizer.iterations

    # Ensure checkpoint directory exists
    checkpoint_dir = FLAGS.model_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=hg)
    
    print ('building finished')
    # TensorBoard summary writer
    summary_writer = tf.summary.create_file_writer(FLAGS.summary_dir)

    best_loss = float('inf')
    current_step = global_step.numpy()  # Convert to integer

    for epoch in range(FLAGS.max_epoch):
        print(f'Epoch {epoch + 1} starts!')

        np.random.shuffle(train_idx)

        smooth_nll_loss = 0.0
        smooth_marginal_loss = 0.0
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
            indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss = train_step(hg, input_nlcd, input_label, optimizer, current_step, summary_writer)

            # Update smooth losses
            smooth_nll_loss += nll_loss
            smooth_marginal_loss += marginal_loss
            smooth_l2_loss += l2_loss
            smooth_total_loss += total_loss

            # Store labels and predictions for AP & AUC
            temp_label.append(input_label)
            temp_indiv_prob.append(indiv_prob)
            
            
            # Log progress every `check_freq` iterations
            if (i + 1) % FLAGS.check_freq == 0:
                mean_nll_loss = smooth_nll_loss / FLAGS.check_freq
                mean_marginal_loss = smooth_marginal_loss/FLAGS.check_freq
                mean_l2_loss = smooth_l2_loss / FLAGS.check_freq
                mean_total_loss = smooth_total_loss / FLAGS.check_freq
                
                # Flatten and reshape predictions and labels
                temp_indiv_prob = np.reshape(np.array(temp_indiv_prob), (-1))
                temp_label = np.reshape(np.array(temp_label), (-1))

                # Compute AP & AUC
                ap = average_precision_score(temp_label, temp_indiv_prob.reshape(-1, 1))

                try:
                    auc = roc_auc_score(temp_label, temp_indiv_prob)
                except ValueError:
                    print('Warning: AUC computation failed due to label mismatch.')
                    auc = None 

                
                # Write to TensorBoard
                with summary_writer.as_default():
                    tf.summary.scalar('train/ap', ap, step=current_step)
                    if auc is not None:
                        tf.summary.scalar('train/auc', auc, step=current_step)
                    tf.summary.scalar('train/nll_loss', mean_nll_loss, step=current_step)
                    tf.summary.scalar('train/marginal_loss', mean_marginal_loss, step=current_step)
                    tf.summary.scalar('train/l2_loss', mean_l2_loss, step=current_step)
                    tf.summary.scalar('train/total_loss', mean_total_loss, step=current_step)
                
                time_str = datetime.datetime.now().isoformat()

                print ("train step: %s\tap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (time_str, ap, nll_loss, marginal_loss, l2_loss, total_loss))
                #print ("validation results: ap=%.6f\tnll_loss=%.6f\tmarginal_loss=%.6f\tl2_loss=%.6f\ttotal_loss=%.6f" % (ap, nll_loss, mean_marginal_loss, l2_loss, total_loss))

                # Reset accumulators
                temp_indiv_prob = []
                temp_label = []
                smooth_nll_loss = 0.0
                smooth_marginal_loss = 0.0
                smooth_l2_loss = 0.0
                smooth_total_loss = 0.0

        # Validation step
        mean_nll_loss = validation_step(hg, data, valid_idx, summary_writer, epoch)

        # Save the best model based on validation loss
        if mean_nll_loss < best_loss:
            print(f"New best loss: {mean_nll_loss:.6f}, saving model...")
            best_loss = mean_nll_loss
            checkpoint.save(file_prefix=checkpoint_prefix)

    print('Training completed!')
    print(f'Best validation loss: {best_loss}')
    ed_time = time.time()
    print("Total running time:", ed_time - st_time)

if __name__ == '__main__':
    app.run(main)
