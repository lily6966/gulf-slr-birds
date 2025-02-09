import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import datetime
import model2 as model
import get_data 
import config 
import os
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import math
import urllib
from pyheatmap.heatmap import HeatMap
import seaborn as sns
from absl import flags, app
import os
import model
FLAGS = flags.FLAGS



def analysis(species, indiv_prob, input_label, printNow=False):
    # Vectorized computation of TP, FP, TN, FN
    pred_label = (indiv_prob > FLAGS.threshold).astype(int)

    TP = np.sum((pred_label == 1) & (input_label == 1))
    FP = np.sum((pred_label == 1) & (input_label == 0))
    TN = np.sum((pred_label == 0) & (input_label == 0))
    FN = np.sum((pred_label == 0) & (input_label == 1))

    # Metrics calculation
    total = TP + TN + FP + FN
    eps = 1e-6
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    accuracy = (TP + TN) / total
    F1 = 2 * precision * recall / (precision + recall + eps)
    auc = roc_auc_score(input_label, indiv_prob)

    # Reshape for AUC and average precision score calculation
    indiv_prob = np.reshape(indiv_prob, (-1))
    input_label = np.reshape(input_label, (-1))

    new_auc = roc_auc_score(input_label, indiv_prob)
    ap = average_precision_score(input_label, indiv_prob)

    if printNow:
        print(f"\nAnalysis of species #{species}:")
        print(f"Occurrence rate: {np.mean(input_label)}")
        print(f"Overall AUC: {auc:.6f}, New AUC: {new_auc:.6f}, AP: {ap:.6f}")
        print(f"F1: {F1}, Accuracy: {accuracy}")
        print(f"Precision: {precision}, Recall: {recall}")
        print(f"TP={TP/total:.6f}, TN={TN/total:.6f}, FN={FN/total:.6f}, FP={FP/total:.6f}")
    
    return accuracy, F1, precision, recall, auc


def test_step(classifier, data, test_idx):
    print('Testing...')
    all_nll_loss, all_l2_loss, all_total_loss = 0, 0, 0
    all_indiv_prob, all_label = [], []
    prob_res, prob_res_sample, loc_res = [], [], []

    real_batch_size = min(FLAGS.testing_size, len(test_idx))
    N_test_batch = (len(test_idx) - 1) // real_batch_size + 1

    for i in range(N_test_batch):
        print(f"{(i * 100.0 / N_test_batch):.1f}% completed")

        start, end = real_batch_size * i, min(real_batch_size * (i + 1), len(test_idx))

        # Get data for the current batch
        input_loc = get_data.get_loc(data, test_idx[start:end])
        input_nlcd = get_data.get_nlcd(data, test_idx[start:end])
        input_label = get_data.get_label(data, test_idx[start:end])

        # Run model inference
        indiv_prob = classifier([input_nlcd, input_label], training=False)

        # Collect results
        prob_res.append(indiv_prob)
        loc_res.append(input_loc)

        # Compute loss
        nll_loss, l2_loss, total_loss = classifier.compute_losses(input_nlcd, input_label)
        all_nll_loss += nll_loss * (end - start)
        all_l2_loss += l2_loss * (end - start)
        all_total_loss += total_loss * (end - start)

        # Accumulate probabilities and labels
        all_indiv_prob.append(indiv_prob)
        all_label.append(input_label)

    # Average losses
    nll_loss = all_nll_loss / len(test_idx)
    l2_loss = all_l2_loss / len(test_idx)
    total_loss = all_total_loss / len(test_idx)

    time_str = datetime.datetime.now().isoformat()
    print(f"Performance on test set: nll_loss={nll_loss:.6f}, l2_loss={l2_loss:.6f}, total_loss={total_loss:.6f}\n{time_str}")

    return np.concatenate(all_indiv_prob), np.concatenate(all_label), prob_res, loc_res


def main(_):
    print('Reading npy...')
    data = np.load(FLAGS.data_dir)
    test_idx = np.load(FLAGS.test_idx) if "esrd" not in FLAGS.data_dir else list(range(data.shape[0]))
    print('Reading completed')

    # Build model using Keras
    classifier = model.MODEL(is_training=False)

    # Restore model weights
    classifier.load_weights(FLAGS.checkpoint_path)
    print(f'Restoring from {FLAGS.checkpoint_path}')

    # Save the feature embedding
    feature_embedding = classifier.get_layer("r_mu").get_weights()[0]
    feature_embedding = np.transpose(feature_embedding)
    os.makedirs(FLAGS.visual_dir, exist_ok=True)
    np.save(os.path.join(FLAGS.visual_dir, f"feature_emb_{FLAGS.mon}"), feature_embedding)

    # Read species names
    file_path = "./data/ebird_occurance_habitat.csv"
    try:
        with open(file_path, "r") as f:
            spe_name = f.readline().strip().split(",")[76:]
        assert len(spe_name) == 500
        print(spe_name[:10])  # Check first 10 species for sanity
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
    except AssertionError:
        print(f"Error: The species list doesn't have 500 species.")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Perform testing and evaluation
    all_indiv_prob, all_label, prob_res, loc_res = test_step(classifier, data, test_idx)

    # Save the prediction results to CSV
    with open("./predictions.csv", "w") as f:
        f.write("LON,LAT,\n")
        for iii, item in enumerate(spe_name):
            f.write(f"{item},")
        f.write("\n")
        for loc, prob in zip(loc_res, prob_res):
            f.write(f"{loc[0]},{loc[1]},")
            f.write(",".join([str(p) for p in prob]) + "\n")

    # Perform analysis for each species and collect summary
    summary = []
    for i in range(FLAGS.r_dim):
        sp_indiv_prob = all_indiv_prob[:, i].reshape(all_indiv_prob.shape[0], 1)
        sp_input_label = all_label[:, i].reshape(all_label.shape[0], 1)

        res = analysis(i, sp_indiv_prob, sp_input_label, False)
        summary.append(res)

    summary = np.asarray(summary)
    np.save("../data/summary_all", summary)  # Save the analysis result


if __name__ == '__main__':
    app.run(main)