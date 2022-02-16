import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.stats

def construct_tridiag(val, nrows, ncols):
    """
    Creates the probability transition matrix.
    Currently only supports steps +1,-1,0
    (Adapted from Pat Meyers)
    """
    return val * (np.tri(nrows, ncols, 1) + np.tri(nrows, ncols, 1).T - np.tri(nrows, ncols, -1).T - np.tri(nrows, ncols, -1) - np.eye(nrows, ncols))

def loglike(spectrum, errors, freqidx):
    """
    get log-likelihood for a given frequency
    and amplitude in our hidden markov model.
     MM: Is this actually a likelihood ?? Gives the right viterbi path,
        but I don't know if it's useful for calculating scores/detection statistics
    (From Pat Meyers)
    """
    # replace entry with a random uniform gaussian variable
    newspec = spectrum.copy()
    newspec[freqidx] = 0
    return np.nansum(-0.5 * (newspec)**2 / (errors**2))
    
def loglike_2(spectrum, errors, freqidx, ntimes):
    """
    get log-likelihood for a given frequency *path*
    """
    # replace entry with a random uniform gaussian variable
    retval = np.sum(np.log([scipy.stats.norm.pdf(spectrum[_,freqidx[_]],scale=errors[_,freqidx[_]]) for _ in range(0,ntimes)]))
    return retval
    
def loglike_path_gaps(spectrum, errors, freqidx, ntimes, nempty):
    """
    Log-likelihood for a spectrum with missing data
    """
    # replace entry with a random uniform gaussian variable
    retval1 = np.sum(np.log([scipy.stats.norm.pdf(spectrum[_,freqidx[_]],scale=errors[_,freqidx[_]]) for _ in range(0,ntimes)]))
    retval2 = np.sum(np.log([scipy.stats.norm.pdf(spectrum[_,freqidx[_]],scale=errors[_,freqidx[_]]) for _ in range(ntimes+nempty,2*ntimes+nempty)]))
    return retval1, retval2, retval1+retval2
    
def get_loglike_array(spectrum, errors, llike=None):
    """
    get log-likelihood for each individual
    state.
    (Adapted from Pat Meyers)
    """
    if llike is None:
        raise ValueError('need to supply loglikelihood')
    llike_array = [llike(spectrum,errors, ii) for ii in range(spectrum.size)] # ii -- time index
    return np.array(llike_array)
    
def viterbi(specgram, errorgram, freqs, A, llike=None, llike_array=None, ntimes=None, nfreqs=None):
    """
    Run the Viterbi algorithm on a spectrogram, returns best path
    (From Pat Meyers)
    """
    delta = np.zeros((ntimes, nfreqs))
    delta[0, :] = get_loglike_array(specgram[0, :], errorgram[0, :], llike=llike) # uniform prior
#     delta[0, :] = llike_array(specgram[0, :], errorgram[0, :], llike=llike)
    PHI = np.zeros((ntimes, nfreqs))
    for kk in range(1, ntimes):
        for ii in range(nfreqs):
        # convolve probability of current state with best probability of transition
            delta[kk, ii] = loglike(specgram[kk, :] , errorgram[kk, :], freqs, ii) +\
                                         np.nanmax(A[ii,:] + delta[kk-1, :])
            # argument of best previous state * transition
            PHI[kk, ii] = np.nanargmax(A[ii, :] + delta[kk-1, :])
    best_path_idx = np.zeros(ntimes)
    best_path_val = np.zeros(ntimes)
    best_path_idx[-1] = np.nanargmax((delta[-1, :]))
    best_path_val[-1] = freqs[int(best_path_idx[-1])]
    for kk in np.arange(0, ntimes-1)[::-1]:
        best_path_idx[kk] = PHI[kk+1,int(best_path_idx[kk+1])]
        best_path_val[kk] = freqs[int(best_path_idx[kk])]
    return best_path_val, best_path_idx, delta, PHI
    
def viterbi2(specgram, errorgram, freqs, A, llike=None, llike_array=None, ntimes=None, nfreqs=None):
    """
    should be faster
    (From Pat Meyers)
    """
    delta = np.zeros((ntimes, nfreqs))
    delta[0, :] = llike_array(specgram[0, :], errorgram[0, :], llike=llike) # uniform prior
#     print delta
        
    PHI = np.zeros((ntimes, nfreqs))
    
    # fill in paths
    for kk in range(1, ntimes):
    # convolve probability of current state with best probability of transition
        delta[kk, :] = llike_array(specgram[kk, :] , errorgram[kk, :], llike=llike) + np.nanmax(np.log(A[:,:]) + delta[kk-1, :], axis=1)
        # argument of best previous state * transition
        PHI[kk, :] = np.nanargmax(np.log(A[:, :]) + delta[kk-1, :], axis=1)
        
    best_path_idx = np.zeros(ntimes)
    best_path_val = np.zeros(ntimes)
    best_path_idx[-1] = np.nanargmax((delta[-1, :]))
    best_path_val[-1] = freqs[int(best_path_idx[-1])]
    
    # backtrack
    for kk in np.arange(0, ntimes-1)[::-1]:
        best_path_idx[kk] = PHI[kk+1,int(best_path_idx[kk+1])]
        best_path_val[kk] = freqs[int(best_path_idx[kk])]
    return best_path_val, best_path_idx, delta, PHI

def make_fake_data(ntimes, nfreqs, path, expower):
    # generate spectrogram
    specgram = np.random.randn(ntimes, nfreqs) # generate "background" as draws from normal dist
    errorgram = np.ones((ntimes, nfreqs)) # sigma=1
    freqs = np.arange(0, 2*nfreqs, 2)
        
    for ii in range(ntimes):
        # make path! Just add excess power to path
        specgram[ii, path[ii]] += expower
    return(specgram)

def make_fake_gaps(ntimes,nfreqs,nempty,excessPower):
    """
    Make fake data with a gap in the middle (current version has equal length data on either side)
    """
    
    try:
        exp1 = excessPower[0]
        exp2 = excessPower[1]
    except:
        exp1 = excessPower
        exp2 = excessPower
    
    freqs = np.arange(0, 2*nfreqs, 2)
    path_combined = [int(nfreqs/2)] # this just starts the path in the middle of the frequency space
    for ii in range(1,2*ntimes+nempty):
        # move -1,0,+1, and I think this accounts for the edges?
        path_combined.append(np.max([np.min([path_combined[ii-1] + np.random.randint(-1,2), nfreqs-1]), 0]))
    
    path1 = path_combined[0:ntimes]
    path2 = path_combined[ntimes+nempty:]
        
    specgram1 = make_fake_data(ntimes,nfreqs,path1,exp1)
    specgram2 = make_fake_data(ntimes,nfreqs,path2,exp2)
    
    empty = np.zeros((nempty,nfreqs))
    intermediate = np.append(specgram1,empty,axis=0)
    combined = np.append(intermediate,specgram2,axis=0)
    
    
    return(combined,path_combined,path1,path2)
