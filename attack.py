import tensorflow as tf
import importlib
import utils
import argparse
import numpy as np
import time
import os
import sys
import utils
import foolbox
import eagerpy as ep
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

import sklearn.metrics


def compute_scores(f_model, g_model, A, a, B, b, pics, x_data, y_data, batch_size):
    F_data = f_model.predict(x_data, batch_size=batch_size)
    G_data = g_model.predict(y_data, batch_size=batch_size)

    Fn_data = (F_data - a).dot(A)
    Gn_data = (G_data - b).dot(B)
    scores = np.square(Fn_data - np.sqrt(pics) * Gn_data).mean(axis=1)

    return scores

def compute_pics(Fn, Gn):
    return np.clip(np.real(np.diag(Fn.T.dot(Gn)/Gn.shape[0])), 0, 1)

def generate_adversarials(model, f_model, g_model, pics, A, a, B, b, center_score, epsilon, x_data, y_data, batch_size, threshold, iterations, alpha=.001, tol=1e-6, max_lambda=2048):
    attack = foolbox.attacks.LinfProjectedGradientDescentAttack()
    num_batches = int(np.ceil(x_data.shape[0] / batch_size))
    min_val, max_val = x_data.min(), x_data.max()
    x_advs = np.zeros(x_data.shape)
    fmodel = foolbox.models.TensorFlowModel(model, bounds=(min_val, max_val))
    bt = trange(num_batches)
    for b in bt:
        # get current batch
        start, end = b * batch_size, min((b+1) * batch_size, x_data.shape[0])
        x_orig_batch, y_batch = x_data[start:end].astype(np.float32), y_data[start:end]

        # initialize with PGD
        images, labels = ep.astensors(tf.convert_to_tensor(x_orig_batch), tf.convert_to_tensor(y_batch.argmax(axis=1)))
        _, advs, _ = attack(fmodel, images, labels, epsilons=epsilon)
        x_batch = advs.raw.numpy()

        # use predicted classes on initial adversarials as targets
        y_pred = model.predict(x_batch)
        indices = np.argmax(y_pred, axis=1)
        mask = np.zeros(y_batch.shape)
        mask[:, indices] = 1

        # precompute G-Net values
        G_batch = g_model.predict(mask)
        Gn_batch = (G_batch - b).dot(B)

        # optimize for detector
        optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        lambdas = 1e-3 * np.ones(x_batch.shape[0])
        for _ in range(iterations):
            x_var_batch = tf.Variable(initial_value=x_batch, trainable=True)
            with tf.GradientTape() as tape:
                # C&W loss
                y_pred = model(x_var_batch)
                y_max = tf.reduce_max(y_pred, axis=1)
                y_target = tf.reduce_sum(y_pred * mask, axis=1)
                cw_loss = tf.math.maximum(y_max - y_target + 1e-3, 0)

                # non-conformity penalty
                F_batch = f_model(x_var_batch)
                Fn_batch = tf.tensordot(F_batch - a, A.astype(np.float32), axes=1)

                x_devs = tf.reduce_mean(tf.square(Fn_batch - np.sqrt(pics) * Gn_batch), axis=1)
                non_conformity = tf.square(x_devs - center_score)

                # distance penalty
                dist = tf.reduce_mean(tf.math.maximum(x_var_batch - x_orig_batch - epsilon, 0), axis=[1, 2, 3])

                # full loss
                losses = dist + lambdas * tf.math.maximum(non_conformity - threshold, cw_loss)
                loss = tf.reduce_mean(losses)

                # compute gradients
                grads = tape.gradient(loss, [x_var_batch])
            # apply gradients
            optimizer.apply_gradients(zip(grads, [x_var_batch]))

            # adjust lambdas
            lambda_mask = (losses.numpy() > tol)
            lambdas[lambda_mask] = np.minimum(2*lambdas[lambda_mask], max_lambda)

            # store batch for next iteration
            x_batch = x_var_batch.numpy()

        x_advs[start:end] = tf.clip_by_value(x_var_batch, x_orig_batch - epsilon, x_orig_batch + epsilon).numpy()
    return x_advs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the adaptive attack on the CANN detector.')
    parser.add_argument('dataset', type=str, help='data set to evaluate')
    parser.add_argument('model', type=str, help='model to evaluate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=20200527, help='random seed')
    parser.add_argument('--eps', type=float, help='relative perturbation budget')
    parser.add_argument('--its', type=int, default=100, help='number of iterations of optimization')
    args = parser.parse_args()

    # set some backend parameters
    print('Initializing...')
    tf.keras.backend.set_floatx('float32')
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # regulate GPU usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # load modules
    data_module = importlib.import_module(f'datasets.{args.dataset}')
    model_module = importlib.import_module(f'models.{args.model}')

    path = f'results/{args.dataset}/{args.model}/{args.seed}'

    # load data
    print('Loading data...')
    x_full, y_full = data_module.load_data()
    y_full = tf.keras.utils.to_categorical(y_full)

    # split data into train/test/validation
    print(f'Loading existing train/test/validation split...')
    train_indices = np.load(f'{path}/train_split.npy')
    test_indices = np.load(f'{path}/test_split.npy')
    valid_indices = np.load(f'{path}/valid_split.npy')

    x_train, y_train = x_full[train_indices], y_full[train_indices]
    x_test, y_test = x_full[test_indices], y_full[test_indices]
    x_valid, y_valid = x_full[valid_indices], y_full[valid_indices]

    # standardize data
    mu, sigma = x_train.mean(), x_train.std()
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    x_valid = (x_valid - mu) / sigma

    # baseline model
    print(f'Loading baseline model...')
    baseline_model = model_module.create_model(x_train.shape[1:], y_train.shape[-1])
    baseline_model.load_weights(f'{path}/baseline.h5')
    
    baseline_acc = baseline_model.evaluate(x_test, y_test, batch_size=args.batch_size)[1]
    print(f'Baseline accuracy: {baseline_acc}')

    # CANN model
    print(f'Loading CANN model...')
    f_model, g_model = data_module.create_cann()
    f_model.load_weights(f'{path}/fnet.h5')
    g_model.load_weights(f'{path}/gnet.h5')
    
    # compute CANN stats
    F_train = f_model.predict(x_train, batch_size=128)
    G_train = g_model.predict(y_train, batch_size=128)

    A, a, B, b = utils.normalizeFG(F_train, G_train)
    Fn_train = (F_train - a).dot(A)
    Gn_train = (G_train - b).dot(B)
    pics = compute_pics(Fn_train, Gn_train)

    # compute scores on clean validation set
    scores = compute_scores(f_model, g_model, A, a, B, b, pics, x_valid, y_valid, args.batch_size)
    center_score = np.mean(scores)
    alphas = np.array([
        np.square(score - np.mean(np.concatenate((scores[:i], scores[i+1:]))))
        for i, score in enumerate(tqdm(scores))])
    pvalues = np.array([
        (alphas >= alpha).mean()
        for alpha in alphas])
    
    # compute scores on clean test set
    y_pred_clean = baseline_model.predict(x_test, batch_size=args.batch_size)
    test_scores = compute_scores(f_model, g_model, A, a, B, b, pics, x_test, y_pred_clean, args.batch_size)
    test_pvalues = np.array([
        (alphas >= np.square(test_score - center_score)).mean()
        for test_score in test_scores])
    
    # load threshold
    with open(f'{path}/results.json', 'r') as f:
        d = json.load(f)
    
    # generate adversarials for baseline
    best_tau = float(d['threshold'])
    threshold = .99 * np.quantile(alphas, 1 - best_tau)

    budget = args.eps * (x_train.max() - x_train.min())
    print(f'Evaluating adaptive adversarials @ {budget} with tau = {best_tau} and r = {threshold}...')

    x_advs = generate_adversarials(baseline_model, f_model, g_model, pics, A, a, B, b, center_score, budget, x_test, y_test, args.batch_size, threshold, args.its)
    adv_acc = baseline_model.evaluate(x_advs, y_test, batch_size=args.batch_size)[1]
    print(f'Adversarial accuracy: {adv_acc}')

    # compute scores on adversarial test set
    y_pred = baseline_model.predict(x_advs, batch_size=args.batch_size)
    adv_scores = compute_scores(f_model, g_model, A, a, B, b, pics, x_advs, y_pred, args.batch_size)
    adv_pvalues = np.array([
        (alphas >= np.square(adv_score - center_score)).mean()
        for adv_score in adv_scores])
    
    # compute AUROC
    all_values = np.unique(np.concatenate((pvalues, adv_pvalues)))
    true_rejections, false_rejections = [], []
    correct_flags = np.concatenate((
        y_pred.argmax(axis=1) == y_test.argmax(axis=1),
        y_pred_clean.argmax(axis=1) == y_test.argmax(axis=1)))
    for tau in all_values:
        accept_flags = np.concatenate((
            adv_pvalues > tau,
            test_pvalues > tau))
        
        true_accepts = np.logical_and(correct_flags, accept_flags).sum()
        false_accepts = np.logical_and(np.logical_not(correct_flags), accept_flags).sum()
        false_rejects = np.logical_and(correct_flags, np.logical_not(accept_flags)).sum()
        true_rejects = np.logical_and(np.logical_not(correct_flags), np.logical_not(accept_flags)).sum()

        trr = true_rejects / (true_rejects + false_accepts)
        frr = false_rejects / (false_rejects + true_accepts)
        detection_acc = (true_rejects + true_accepts) / accept_flags.shape[0]

        true_rejections.append(trr)
        false_rejections.append(frr)
    sorted_idx = np.argsort(false_rejections)
    auroc = sklearn.metrics.auc(np.array(false_rejections)[sorted_idx], np.array(true_rejections)[sorted_idx])

    # compute ROC point
    accept_flags = np.concatenate((
        adv_pvalues > best_tau,
        test_pvalues > best_tau))
    
    true_accepts = np.logical_and(correct_flags, accept_flags).sum()
    false_accepts = np.logical_and(np.logical_not(correct_flags), accept_flags).sum()
    false_rejects = np.logical_and(correct_flags, np.logical_not(accept_flags)).sum()
    true_rejects = np.logical_and(np.logical_not(correct_flags), np.logical_not(accept_flags)).sum()

    trr = true_rejects / (true_rejects + false_accepts)
    frr = false_rejects / (false_rejects + true_accepts)
    detection_acc = (true_rejects + true_accepts) / accept_flags.shape[0]

    # save statistics
    with open(f'{path}/results_adaptive.json', 'w') as f:
        json.dump({
            'adversarial_acc': str(adv_acc),
            'auroc': str(auroc),
            'trr': str(trr),
            'frr': str(frr),
            'detection_acc': str(detection_acc)
        }, f)
