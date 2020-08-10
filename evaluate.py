import tensorflow as tf
import importlib
import utils
import argparse
import numpy as np
import time
import foolbox
import os
import utils
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

def generate_adversarials(model, attack, epsilon, x_data, y_data, batch_size):
    num_batches = int(np.ceil(x_data.shape[0] / batch_size))
    min_val, max_val = x_data.min(), x_data.max()
    fmodel = foolbox.models.TensorFlowModel(model, bounds=(min_val, max_val))
    x_advs = np.zeros(x_data.shape)
    bt = trange(num_batches)
    for b in bt:
        start, end = b * batch_size, min((b+1) * batch_size, x_data.shape[0])
        x_batch, y_batch = x_data[start:end].astype(np.float32), y_data[start:end]

        images, labels = ep.astensors(tf.convert_to_tensor(x_batch), tf.convert_to_tensor(y_batch.argmax(axis=1)))
        _, advs, _ = attack(fmodel, images, labels, epsilons=epsilon)
        x_advs[start:end] = advs.raw.numpy()
    return x_advs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CANN detector.')
    parser.add_argument('dataset', type=str, help='data set to evaluate')
    parser.add_argument('model', type=str, help='model to evaluate')
    parser.add_argument('--attack', type=str, default='LinfProjectedGradientDescentAttack', help='attack to use')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=20200527, help='random seed')
    parser.add_argument('--eps', type=float, help='maximal relative perturbation budget')
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
    Path(path).mkdir(parents=True, exist_ok=True)

    # load data
    print('Loading data...')
    x_full, y_full = data_module.load_data()
    y_full = tf.keras.utils.to_categorical(y_full)

    # split data into train/test/validation
    if not os.path.isfile(f'{path}/train_split.npy') or \
        not os.path.isfile(f'{path}/test_split.npy') or \
        not os.path.isfile(f'{path}/valid_split.npy'):
        print(f'Creating new train/test/validation split...')
        indices = range(x_full.shape[0])

        train_indices, rest_indices = train_test_split(indices, test_size=.3)
        test_indices, valid_indices = train_test_split(rest_indices, test_size=.3)

        np.save(f'{path}/train_split.npy', train_indices)
        np.save(f'{path}/test_split.npy', test_indices)
        np.save(f'{path}/valid_split.npy', valid_indices)
    else:
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
    baseline_model = model_module.create_model(x_train.shape[1:], y_train.shape[-1])
    if not os.path.isfile(f'{path}/baseline.h5'):
        print(f'Training baseline model...')
        model_module.train_baseline(baseline_model, x_train, y_train, x_valid, y_valid, args.batch_size)
        baseline_model.save_weights(f'{path}/baseline.h5')
    else:
        print(f'Loading baseline model...')
        baseline_model.load_weights(f'{path}/baseline.h5')
    
    baseline_acc = baseline_model.evaluate(x_test, y_test, batch_size=args.batch_size)[1]
    print(f'Baseline accuracy: {baseline_acc}')

    # CANN model
    if not os.path.isfile(f'{path}/fnet.h5') or not os.path.isfile(f'{path}/gnet.h5'):
        print(f'Training CANN model...')
        while True:
            try:
                f_model, g_model = data_module.create_cann()
                data_module.train_cann(f_model, g_model, x_train, y_train, args.batch_size)
                break
            except ValueError:
                del f_model
                del g_model
        f_model.save_weights(f'{path}/fnet.h5')
        g_model.save_weights(f'{path}/gnet.h5')
    else:
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
    
    # generate adversarials for baseline
    attack = getattr(foolbox.attacks, args.attack)()
    budget = args.eps * (x_test.max() - x_test.min())

    print(f'Evaluating adversarials using {attack.__class__.__name__} @ {budget}...')
    x_valid_advs = generate_adversarials(baseline_model, attack, budget, x_valid, y_valid, args.batch_size)
    x_advs = generate_adversarials(baseline_model, attack, budget, x_test, y_test, args.batch_size)

    adv_acc = baseline_model.evaluate(x_advs, y_test, batch_size=args.batch_size)[1]
    print(f'Adversarial accuracy: {adv_acc}')

    # compute scores on adversarial validation set
    y_pred = baseline_model.predict(x_valid_advs, batch_size=args.batch_size)
    adv_scores = compute_scores(f_model, g_model, A, a, B, b, pics, x_valid_advs, y_pred, args.batch_size)
    adv_pvalues = np.array([
        (alphas >= np.square(adv_score - center_score)).mean()
        for adv_score in adv_scores])

    # compute scores on adversarial test set
    y_pred = baseline_model.predict(x_advs, batch_size=args.batch_size)
    adv_scores = compute_scores(f_model, g_model, A, a, B, b, pics, x_advs, y_pred, args.batch_size)
    test_adv_pvalues = np.array([
        (alphas >= np.square(adv_score - center_score)).mean()
        for adv_score in adv_scores])
    
    # compute AUROC
    true_rejections, false_rejections, detection_accs = [], [], []
    best_tau = None
    best_score = -np.inf
    best_idx = None
    correct_flags = np.concatenate((
        y_pred.argmax(axis=1) == y_test.argmax(axis=1),
        y_pred_clean.argmax(axis=1) == y_test.argmax(axis=1)))
    all_values = np.unique(np.concatenate((test_adv_pvalues, test_pvalues)))
    for idx, tau in enumerate(all_values):
        accept_flags = np.concatenate((
            test_adv_pvalues > tau,
            test_pvalues > tau))
        
        true_accepts = np.logical_and(correct_flags, accept_flags).sum()
        false_accepts = np.logical_and(np.logical_not(correct_flags), accept_flags).sum()
        false_rejects = np.logical_and(correct_flags, np.logical_not(accept_flags)).sum()
        true_rejects = np.logical_and(np.logical_not(correct_flags), np.logical_not(accept_flags)).sum()

        trr = true_rejects / (true_rejects + false_accepts)
        frr = false_rejects / (false_rejects + true_accepts)
        detection_acc = (true_rejects + true_accepts) / accept_flags.shape[0]

        tau_score = trr - frr
        if tau_score > best_score:
            best_score = tau_score
            best_tau = tau
            best_idx = idx

        true_rejections.append(trr)
        false_rejections.append(frr)
        detection_accs.append(detection_acc)
    sorted_idx = np.argsort(false_rejections)
    auroc = sklearn.metrics.auc(np.array(false_rejections)[sorted_idx], np.array(true_rejections)[sorted_idx])

    # current ROC point
    trr = true_rejections[best_idx]
    frr = false_rejections[best_idx]
    detection_acc = detection_accs[best_idx]

    # save statistics
    with open(f'{path}/results.json', 'w') as f:
        json.dump({
            'baseline_acc': str(baseline_acc),
            'adversarial_acc': str(adv_acc),
            'center_score': str(center_score),
            'threshold': str(best_tau),
            'auroc': str(auroc),
            'trr': str(trr),
            'frr': str(frr),
            'detection_acc': str(detection_acc)
        }, f)
