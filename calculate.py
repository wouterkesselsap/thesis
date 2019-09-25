import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from math import ceil
from itertools import chain, groupby
from operator import itemgetter
from copy import copy
from qutip import *
from process import *
from supports import *
from envelopes import drive_nonosc


def calculate(H, psi0, e_ops, H_args, options, Nc, g, Np, Np_per_batch, verbose=True):
    "Integrate through time evolution."
    
    t0 = H_args['t0']
    t3 = H_args['t3']
    
    batches = create_batches(t0, t3, Np, Np_per_batch)

    ID, folder, now = prepare_folder()

    # Calculate!
    for num, tlist in enumerate(batches):
        if verbose:
            print(num+1, "/", len(batches), ":", int(np.round(100*(num+1)/len(batches))), "%")
        result = mesolve(H, psi0, tlist, c_ops=[], e_ops=e_ops, args=H_args, options=options)
        e0, g1, e1, g0 = combined_probs(result.states, Nc)
        coupling = drive_nonosc(tlist, H_args)  # unitless, peaks at 1
        saveprog(result, e0, g1, e1, g0, coupling, num, folder)
        psi0 = copy(result.states[-1])
        del result, e0, g1, e1, g0
    end_calc = datetime.now()
    if verbose:
        print("Evolution completed in {} s".format((end_calc - now).total_seconds()))
    
    return folder


def combined_probs(states, Nc):
    """Calculates |e,0> - |g,1> and |e,1> - |g,0>."""
    inds = ((1,0), (0,1), (1,1), (0,0))
    probs = list()
    [probs.append(list()) for i in range(len(inds))]
    
    for i, ind in enumerate(inds):
        for state in states:
            probs[i].append((state.data[ind[1] + Nc*ind[0], 0]*state.data[ind[1] + Nc*ind[0], 0].conj()).real)
    
    e0 = np.asarray(probs[0])
    g1 = np.asarray(probs[1])
    e1 = np.asarray(probs[2])
    g0 = np.asarray(probs[3])
    return e0, g1, e1, g0


def extrema(x, times):
    maxima = list()
    t_maxima = list()
    n_maxima = list()
    minima = list()
    t_minima = list()
    n_minima = list()
    
    for n, value, t in zip(range(len(x)), x, times):
        
        # If first element
        if (n == 0 and value > x[1]):
            maxima.append(value)
            t_maxima.append(t)
            n_maxima.append(n)
        elif (n == 0 and value < x[1]):
            minima.append(value)
            t_minima.append(t)
            n_minima.append(n)
        
        # If last element
        elif (n == len(x)-1 and value > x[1]):
            maxima.append(value)
            t_maxima.append(t)
            n_maxima.append(n)
        elif (n == len(x)-1 and value < x[1]):
            minima.append(value)
            t_minima.append(t)
            n_minima.append(n)
        
        # Check if maximum
        elif (value > x[n-1] and value > x[n+1]):
            maxima.append(value)
            t_maxima.append(t)
            n_maxima.append(n)
        
        # Check if minimum
        elif (value < x[n-1] and value < x[n+1]):
            minima.append(value)
            t_minima.append(t)
            n_minima.append(n)
    
    return maxima, t_maxima, n_maxima, minima, t_minima, n_minima


def remove_micromotion(x, times, method, window_length=1001, order=2, **kwargs):
    """Remove micromotion from input signal. by determining all local maxima
    and minima, and subsequently draw the output in the middle of the
    region defined by these local maxima and minima.
    
    Method is either
    - 'bisect' : determines all local maxima
      and minima, and subsequently draw the output on the bisection
      of two subsequent extrema; or
    - 'savgol' : Savitzky-Golay filter; or
    - 'lowpass' : cuts off Fourier spectrum after some value
    """
    if method == 'bisect':
        xnew = list()
        tnew = list()
        maxima, t_maxima, _, minima, t_minima, _ = extrema(x, times)
        
        supports = copy(maxima)
        supports.extend(minima)
        t_supports = copy(t_maxima)
        t_supports.extend(t_minima)
        supports_zipped = sorted(zip(t_supports, supports))
        t_supports, supports = zip(*supports_zipped)
        
        for interval in range(1, len(supports)):
            maxval = max(supports[interval-1], supports[interval])
            minval = min(supports[interval-1], supports[interval])
            xnew.append(minval + (maxval - minval)/2)
            tnew.append(t_supports[interval-1] + (t_supports[interval] - t_supports[interval-1])/2)
            
    elif method == 'savgol':
        xnew = savgol_filter(x, window_length, order)
    
    elif method == 'lowpass':
        spectrum = np.fft.fft(x)
        _, _, _, minima, _, n_minima = extrema(spectrum, times)
        spectrum[n_minima[1]+1 :] = np.zeros(len(spectrum) - n_minima[1]-1).tolist()
        xnew = np.fft.ifft(spectrum)
        print("WARNING: lowpass filter does not yet give appropriate results")
        
    return xnew, times
    


def cluster(x, t, out='extremum'):
    """Determine clusters in data and return a single point per cluster.
    
    out : str
        'centroid' : return cluster centroid
        'extremum' : return maximum or minimum
    """
    
    # Determine clusters
    xmin = min(x)
    xmax = max(x)
    k = 0
    classified = [(0, 0)]
    
    if abs(x[0] - xmax) < abs(x[0] - xmin):
        pole = 1
    elif abs(x[0] - xmax) > abs(x[0] - xmin):
        pole = -1
    
    poles = [pole]
    
    for n, val in enumerate(x):
        if n != 0:
            if abs(val - xmax) < abs(val - xmin):
                newpole = 1
            elif abs(val - xmax) > abs(val - xmin):
                newpole = -1
            if newpole == pole:
                classified.append((n, k))
            elif newpole != pole:
                pole = copy(newpole)
                k += 1
                classified.append((n, k))
                poles.append(pole)
    
    clustered = list()
    for key, group in groupby(classified, key=itemgetter(1)):
        inds = [item[0] for item in group]
        clustered.append(inds)
    
    # Calculate output per cluster
    xlocs = list()
    tlocs = list()
    
    for ic, cluster in enumerate(clustered):
        if out == 'centroid':
            xtot = 0
            ttot = 0
            for i in cluster:
                xtot += x[i]
                ttot += t[i]
            xlocs.append(xtot/len(cluster))
            tlocs.append(ttot/len(cluster))
        
        elif out == 'extremum':
            if poles[ic] == 1:
                xmax = max(x[cluster[0] : cluster[-1] +1])
                tmax = t[x.index(xmax)]
                xlocs.append(xmax)
                tlocs.append(tmax)
            elif poles[ic] == -1:
                xmin = min(x[cluster[0] : cluster[-1] +1])
                tmin = t[x.index(xmin)]
                xlocs.append(xmin)
                tlocs.append(tmin)
    
    return xlocs, tlocs


def sideband_freq(x, times, rm_micromotion=False, method='savgol', **kwargs):
    """Determine the sideband transition frequency [GHz] based on
    expectation values.
    If the micromotion is not already removed from the signal, rm_micromotion
    should be set to True."""
    
    if rm_micromotion:
        x, times = remove_micromotion(x, times, method)
    
    maxima, t_maxima, _, minima, t_minima, _ = extrema(x, times)
    supports = copy(maxima)
    supports.extend(minima)
    t_supports = copy(t_maxima)
    t_supports.extend(t_minima)
    
    supports_zipped = sorted(zip(t_supports, supports))
    t_supports, supports = zip(*supports_zipped)
    supports, t_supports = cluster(supports, t_supports)
    
    supports = supports[1:-1]  # remove first and last element
    t_supports = t_supports[1:-1]  # remove first and last element
    
    if len(supports) < 2:
        print("WARNING: not enough sideband oscillations to determinde frequency,")
        print("         increase the simulation time")
        return 0
    elif len(supports) == 2:
        print("WARNING: not enough sideband oscillations to accurately determinde frequency,")
        print("         increase the simulation time for a more accurate result")
        dts = np.diff(t_supports)
        Tsb = 2*sum(dts)/len(dts)  # sideband transition period [ns]
        wsb = 1/Tsb  # sideband transition frequency [GHz]
        return wsb*2*pi
    else:
        dts = np.diff(t_supports)
        Tsb = 2*sum(dts)/len(dts)  # sideband transition period [ns]
        wsb = 1/Tsb  # sideband transition frequency [GHz]
        return wsb*2*pi