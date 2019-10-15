import time
import numpy as np
import matplotlib.pyplot as plt
from plots import *
from supports import *
from process import *
from calculate import *
from envelopes import *
from qutip import *
from ipywidgets import widgets
from IPython.display import display

home = "/home/student/thesis/"


def sample_single_tone(Nq, wq, wc, Ec, g, Omega, shift, sb, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, H, options, parallel):
    """
    Performs a single-tone sideband transition simulation
    with the given input parameters. Plots the expectation
    values of the qubit & cavity, and the combined probability
    |e0>-|g1> in the case of red sideband transitions, or
    |e1>-|g0> in the case of blue sideband transitions.
    
    Input
    -----
    The input parameters are equal to the names in 2p_sideband.ipynb.
    
    Returns
    -------
    figqc : matplotlib.pyplot.Figure class object
        Figure with expected qubit and cavity occupation number
    fig : matplot.pyplot.Figure class object
        Figure with combined probabilities
    """
    from envelopes import drive
    
    i = shift[0]
    shift = shift[1]
    
    Nc = 10  # number of levels in resonator 1
    Nt = 1   # number of drive tones
    
    if sb == 'red':
        if wq > wc:
            wd = (wq + shift - wc)/2
        elif wq < wc:
            wd = (wc - wq - shift)/2
    elif sb == 'blue':
            wd = (wq + shift + wc)/2
        
    Np = 100*int(t3)     # number of discrete time steps for which to store the output
    
    # Operators
    b, a, nq, nc = ops(Nq, Nc)

    # Hamiltonian arguments
    H_args = {'t0' : t0, 't1' : t1, 't2' : t2,
              't3' : t3, 'tg' : tg, 'Q'  : Q,
              'smooth' : smooth, 'wd' : wd}

    # Expectation operators
    e_ops = [nq, nc]
        
    progfolder = calculate(H, psi0, e_ops, H_args, options, Nc, Np, Np_per_batch, parallel, verbose=False)
    
    """ SAVE EVOLUTION TEMPORARILY """

    srcfolder =  progfolder
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()
    
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    time.sleep(i*3)
    
    print(" ")
    print("shift = {}, wd = {}".format(shift/2/pi, wd/2/pi))
    if sb == 'red':
        print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
    elif sb == 'blue':
        print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
    
    """ EXPECTATION VALUES """

    if sb == 'red':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wd=wd, wsb=0, title=expect_title)
    elif sb == 'blue':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wd=wd, wsb=0, title=expect_title)
    
    """COMBINED PROBABILITIES"""

    if sb == 'red':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling, e0=e0, g1=g1, wd=wd, wsb=0, title=cp_title)
    elif sb == 'blue':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling, e1=e1, g0=g0, wd=wd, wsb=0, title=cp_title)
    
#     figqc.savefig(srcfolder + "/figqc.png", bbox_inches='tight')
#     fig.savefig(srcfolder + "/fig.png", bbox_inches='tight')
    
    return figqc, fig


def sample_double_tone(Nq, wq, wc, Ec, g, Omegaq, Omegac, shift, dw, sb, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, H, options, parallel):
    """
    Performs a double-tone sideband transition simulation
    with the given input parameters. Plots the expectation
    values of the qubit & cavity, and the combined probability
    |e0>-|g1> in the case of red sideband transitions, or
    |e1>-|g0> in the case of blue sideband transitions.
    
    Input
    -----
    The input parameters are equal to the names in 2p_sideband.ipynb.
    
    Returns
    -------
    figqc : matplotlib.pyplot.Figure class object
        Figure with expected qubit and cavity occupation number
    fig : matplot.pyplot.Figure class object
        Figure with combined probabilities
    """
    i = shift[0]
    shift = shift[1]
    
    Nc = 10  # number of levels in resonator 1
    Nt = 2   # number of drive tones

    delta = wq - wc    # detuning    
    
    if sb == 'red':
        wdq =  wq + shift - dw
        wdc =  wc - dw
    elif sb == 'blue':
        wdq =  wq + shift + dw
        wdc =  wc - dw

    Np = 100*int(t3)     # number of discrete time steps for which to store the output
    
    # Operators
    b, a, nq, nc = ops(Nq, Nc)

    # Hamiltonian arguments
    H_args = {'t0' : t0, 't1' : t1, 't2' : t2, 't3' : t3, 'tg' : tg,
              'Q'  : Q, 'smooth' : smooth, 'Nt' : Nt, 'wdq' : wdq, 'wdc' : wdc}

    # Expectation operators
    e_ops = [nq, nc]
        
    """ CALCULATE! """

    progfolder = calculate(H, psi0, e_ops, H_args, options, Nc, Np, Np_per_batch, parallel, verbose=False)
    
    """ SAVE EVOLUTION TEMPORARILY """
    
    srcfolder =  progfolder
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()
    
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    time.sleep(i*3)
    
    print(" ")
    print("shift = {}".format(shift/2/pi))
    if sb == 'red':
        print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
    elif sb == 'blue':
        print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
        
    """ EXPECTATION VALUES """

    if sb == 'red':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wsb=0, title=expect_title)
    elif sb == 'blue':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wsb=0, title=expect_title)
    
    """COMBINED PROBABILITIES"""

    if sb == 'red':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                xlim=None, ylim=None, figsize=[15,3], e0=e0, g1=g1, wsb=0, title=cp_title)
    elif sb == 'blue':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                xlim=None, ylim=None, figsize=[15,3], e1=e1, g0=g0, wsb=0, title=cp_title)
    
#     figqc.savefig(srcfolder + "/figqc.png", bbox_inches='tight', dpi=500)
#     fig.savefig(srcfolder + "/fig.png", bbox_inches='tight', dpi=500)
    
    return figqc, fig