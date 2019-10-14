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


def calculate(H, psi0, e_ops, H_args, options, Nc, Np, Np_per_batch, parallel, verbose=True):
    """
    Integrate through time evolution using qutip's Lindblad master equation solver.
    
    Input
    -----
    H : list
        Full Hamiltonian. Time-dependent terms must be given as
        [qutip.Qobj, callback function].
    psi0 : qutip.Qobj class object
        Initial state
    e_ops : list of qutip.Qubj class objects
        Operators for which to evaluate the expectation value
    H_args : dict
        Parameters for time-dependent Hamiltonians and collapse operators
    options : qutip.Options class object
        Options for the solver
    Nc : int
        Number of cavity levels
    Np : int
        Number of points for which to store the data
    Np_per_batch : int, float
        Number of points per batch
    parallel : bool
        Whether multiple simulations are run in parallel
    verbose : bool
        Print progress
    
    Returns
    -------
    folder : str
        Folder name in which the evolution is stored
    """
    
    t0 = H_args['t0']
    t3 = H_args['t3']
    
    batches = create_batches(t0, t3, Np, Np_per_batch)

    ID, folder, now = prepare_folder(parallel)

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
    """
    Calculates |e,0> - |g,1> and |e,1> - |g,0> through time
    from given quantum states. Assumes |qubit, cavity>.
    
    Input
    -----
    states : list of qutip.Qobj class objects
        Full quantum states
    Nc : int
        Number of cavity levels
    
    Returns
    -------
    e0 : np.array
        Probabilities of |e,0>
    g1 : np.array
        Probabilities of |g,1>
    e1 : np.array
        Probabilities of |e,1>
    g0 : np.array
        Probabilities of |g,0>
    """
    
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
    """
    Determines all extrema in a given sequence with corresponding time values.
    First and last element of x are always returned.
    
    Input
    -----
    x : array-like
        Values from which to determine the extrema
    times : array-like
        Corresponding time values
    
    Returns
    -------
    maxima : list
        All maxima from x
    t_maxima : list
        Corresponding values from times for maxima
    n_maxima : list
        Indeces of maxima in x
    minima : list
        All minima from x
    t_minima : list
        Corresponding values from times for minima
    n_minima : list
        Indeces of minima in x
    """
    
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


def remove_micromotion(x, times, method='savgol', window_length=1001, order=2, **kwargs):
    """
    Removes micromotion from input signal x by a specified method.
    
    Input
    -----
    x : array-like
        Signal to remove the micromotion from
    times : array-like
        Corresponding time values of signal
    method : str
        Method to use to remove micromotion. The options are:
        - 'bisect' : determines all local maxima and minima,
          and subsequently draws the output on the bisection
          of two subsequent extrema; or
        - 'savgol' : Savitzky-Golay filter; or
        - 'lowpass' : cuts off Fourier spectrum after some value
    window_length : int
        Window length in case of Savitzky-Golay filter
    order : int
        Polynomial order in case of Savitzky-Golay filter
    
    Returns
    -------
    xnew : list, np.array
        New signal
    times : list, np.array
        Corresponding time values
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
    """
    Determines clusters of subsequent maxima or minima in the data
    and return a single point per cluster.
    
    Input
    -----
    x : array-like
        Values of maxima and minima
    t : array-like
        Corresponding time values
    out : str
        Location of output points. Options are:
        'centroid' : return cluster centroid, or
        'extremum' : return maximum or minimum
    
    Returns
    -------
    xlocs : list
        Cluster locations
    tlocs : list
        Corresponding time values
    """
    
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(t, np.ndarray):
        t = t.tolist()
    
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


def sideband_freq(x, times, rm_micromotion=False, method='savgol', tg=10, rtol=0.5, **kwargs):
    """
    Determines the sideband transition frequency in [GHz] based on
    expectation values.
    If the micromotion is not already removed from the signal, rm_micromotion
    should be set to True.
    
    Input
    -----
    x : array-like
        Signal to determine the sideband transition frequency from
    times : array-like
        Corresponing time values
    rm_micromotion : bool
        Remove micromotion from x
    method : str
        Method to use for removal of the micromotion. Consult the
        remove_micromotion function for the possible parameters
    tg : float
        Time of Gaussian rise and fall
    rtol : float
        Ratio between distance to mean and distance to extrema to tolerate.
        All points closer to the mean than rtol times the distance to the
        global extrema are removed
    
    Returns
    -------
    wsb*2*pi : float
        Sideband transition frequency [rad/s]
    """
    
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
    
    # Remove supports due to remaining oscillations in filtered signal close to signal mean
    news = list()
    newt = list()
    for s in supports:
        d_to_max = abs(s - max(supports))
        d_to_min = abs(s - min(supports))
        d_to_mid = abs(s - min(supports) - (max(supports)-min(supports))/2)
        if ( d_to_mid < rtol*d_to_max and d_to_mid < rtol*d_to_min ):
            pass
        else:
            news.append(s)
            newt.append(t_supports[supports.index(s)])
    
    supports = copy(news)
    t_supports = copy(newt)
    if max(t_supports) > max(times)-tg:  # if last cluster within Gaussian fall
        supports = supports[1:-1]  # remove first and last element
        t_supports = t_supports[1:-1]  # remove first and last element
    else:
        supports = supports[1:]  # remove first element
        t_supports = t_supports[1:]  # remove first element
    
    if len(supports) < 2:
        print("WARNING: not enough sideband oscillations to determinde frequency,")
        print("         increase the simulation time")
        return 0
    elif len(supports) == 2:
        print("WARNING: not enough sideband oscillations to accurately determinde frequency,")
        print("         increase the simulation time for a more accurate result")
        dts = np.diff(t_supports)
        Tsb = 2*np.mean(dts)  # sideband transition period [ns]
        wsb = 1/Tsb  # sideband transition frequency [GHz]
        return wsb*2*pi
    else:
        dts = np.diff(t_supports)
        Tsb = 2*np.mean(dts)  # sideband transition period [ns]
        wsb = 1/Tsb  # sideband transition frequency [GHz]
        return wsb*2*pi