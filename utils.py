import tensorflow as tf
import numpy as np
import scipy as sp

from sklearn.covariance import LedoitWolf

class Welford(object):
    """
    Welford's algorithm for numerically stable online mean.
    """
    def __init__(self, shape):
        self.k = 0
        self.M = np.zeros(shape)
    
    def update(self,x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M)/self.k
        self.M = newM

    @property
    def mean(self):
        return self.M

def compute_ece(y_probs, y_preds, y_true, bins='fd'):
    """
    Compute the expected calibration error (ECE) using a given binning method.
    """
    # check number of bins
    if not isinstance(bins, int):
        try:
            _, edges = np.histogram(y_probs, bins=bins)
            bins = len(edges) - 1
        except IndexError:
            bins = 2

    # bin the confidence levels
    levels = np.linspace(y_probs.min(), y_probs.max(), bins)
    bin_indices = np.digitize(y_probs, levels, right=True)
    result = 0.
    for i in np.unique(bin_indices):
        y_probs_bin, y_preds_bin, y_true_bin = y_probs[bin_indices==i], y_preds[bin_indices==i], y_true[bin_indices==i]

        # update current estimate
        acc = (y_preds_bin == y_true_bin).mean()
        conf = y_probs_bin.mean()
        result += len(y_probs_bin) / y_probs.shape[0] * abs(acc - conf)
    return result

def shrink_cov(cov):
    """
    Estimate sample covariance using the Ledoit-Wolf estimator.
    """
    lw = LedoitWolf(assume_centered=True).fit(cov)
    shrinkage = lw.shrinkage_

    m = cov.shape[0]
    mu = tf.linalg.trace(cov) / m
    return (1 - shrinkage) * cov + shrinkage * mu * tf.eye(m, dtype=mu.dtype)

def compute_loss(f_out, g_out, clip_min=np.float32(-10000), clip_max=np.float32(10000)):
    """
    Compute the loss for the FGNet.
    """
    # number of samples in batch
    nBatch = f_out.shape[0]
    
    # clip to avoid runaway arguments
    f_clip = tf.clip_by_value(f_out,clip_min,clip_max)
    g_clip = tf.clip_by_value(g_out,clip_min,clip_max)
    
    # create regularized correlation matrices
    corrF = shrink_cov(tf.linalg.matmul(f_clip, f_clip, transpose_a=True) / nBatch)
    corrFG = shrink_cov(tf.linalg.matmul(f_clip, g_clip, transpose_a=True) / nBatch)
    
    # Second moment of g
    sqG = tf.reduce_sum(tf.reduce_mean(tf.square(g_clip), axis=0))
    
    # compute Schatten 2-norm
    invCorrF = tf.linalg.pinv(corrF)
    prodGiFG = tf.linalg.matmul(tf.linalg.matmul(corrFG, invCorrF, transpose_a=True), corrFG)
    
    s, v = tf.linalg.eigh(prodGiFG)
    schatNorm = tf.reduce_sum(tf.sqrt(tf.abs(s)))
    
    # define objective
    objective = sqG - 2*schatNorm
    
    #return objective
    return objective

def normalizeFG(F, G):
    """
    Whitening transformation for FGNet output.
    """
    # Values for G
    Gs = G[:,1:]
    b_mean = Gs.mean(axis=0)
    Gs = Gs - b_mean
    corrG = shrink_cov(Gs.transpose().dot(Gs)/Gs.shape[0])
    corrG_sqrt_inv = sp.linalg.sqrtm(np.linalg.pinv(corrG))
    
    b_mean = np.concatenate(([0],b_mean))
    B = sp.linalg.block_diag(1,corrG_sqrt_inv)
    
    nG = (G-b_mean).dot(B)

    # values for F
    Fs = F[:,1:]
    a_mean = Fs.mean(axis=0)
    Fs = Fs - a_mean
    corrF = shrink_cov(Fs.transpose().dot(Fs)/Fs.shape[0])
    U,v,_ = np.linalg.svd(corrF)
    corrF_sqrt_inv = (U*(v)**(-.5)).dot(U.transpose())

    a_mean = np.concatenate(([0],a_mean))
    A = sp.linalg.block_diag(1,corrF_sqrt_inv)
    
    nF = (F-a_mean).dot(A)
    
    # Create proper normalization
    U,s,V = np.linalg.svd(nF.transpose().dot(nG)/G.shape[0])

    return A.dot(U),a_mean,B.dot(V.transpose()),b_mean
