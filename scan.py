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


def sbsample(Nq, wq, wc, Ec, g, shift, sb, Nt, H, H_args, psi0, Np_per_batch,
             options, home, parallel, *args):
    """
    Performs a single- or double-tone sideband transition simulation
    with the given input parameters. Plots the expectation
    values of the qubit & cavity, and the combined probability
    |e0>-|g1> in the case of red sideband transitions, or
    |e1>-|g0> in the case of blue sideband transitions.
    
    Due to the use of the pool.starmap function, the additional
    arguments of *args, dependent on Nt, have to be passed in
    a definite order.
    Nt = 1 : Omega
    Nt = 2 : Omegaq, Omegac, dw
    
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
    
    if Nt == 1:
        if sb == 'red':
            if wq > wc:
                wd = (wq + shift - wc)/2
            elif wq < wc:
                wd = (wc - wq - shift)/2
        elif sb == 'blue':
                wd = (wq + shift + wc)/2
    elif Nt == 2:
        dw = args[2]
        if sb == 'red':
            wdq =  wq + shift - dw
            wdc =  wc - dw
        elif sb == 'blue':
            wdq =  wq + shift + dw
            wdc =  wc - dw
    
    Np = 100*int(H_args['t3'])  # number of discrete time steps for which to store the output
    b, a, nq, nc = ops(Nq, Nc)  # Operators
    if Nt == 1:
        H_args['wd'] = wd
    elif Nt == 2:
        H_args['wdq'] = wdq
        H_args['wdc'] = wdc
    e_ops = [nq, nc]
        
    srcfolder = calculate(H, psi0, e_ops, H_args, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False)
    
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    if i < 10:
        num = "0" + str(i)
    elif i >= 10:
        num = str(i)
    
    print(" ")
    
    if Nt == 1:
        Omega = args[0]
        print("shift = {}, wd = {}".format(shift/2/pi, wd/2/pi))
        if sb == 'red':
            print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
            expect_title = "shift = {}".format(shift/2/pi)
            cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wd=wd, wsb=0, title=expect_title, Omega=Omega)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    e0=e0, g1=g1, wd=wd, wsb=0, title=cp_title, Omega=Omega)
        elif sb == 'blue':
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            expect_title = "shift = {}".format(shift/2/pi)
            cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wd=wd, wsb=0, title=expect_title, Omega=Omega)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    e1=e1, g0=g0, wd=wd, wsb=0, title=cp_title, Omega=Omega)            
    elif Nt == 2:
        Omegaq = args[0]
        Omegac = args[1]
        print("shift = {}".format(shift/2/pi))
        if sb == 'red':
            print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
            expect_title = "shift = {}".format(shift/2/pi)
            cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wsb=0, title=expect_title, Omegaq=Omegaq, Omegac=Omegac)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    xlim=None, ylim=None, figsize=[15,3], e0=e0, g1=g1, wsb=0,
                                    title=cp_title, Omegaq=Omegaq, Omegac=Omegac)
        elif sb == 'blue':
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            expect_title = "shift = {}".format(shift/2/pi)
            cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wsb=0, title=expect_title, Omegaq=Omegaq, Omegac=Omegac)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    xlim=None, ylim=None, figsize=[15,3], e1=e1, g0=g0, wsb=0,
                                    title=cp_title, Omegaq=Omegaq, Omegac=Omegac)
    fig.savefig(home + "temp/fig{}_{}.png".format(num, shift/2/pi))
    figqc.savefig(home + "temp/figqc{}_{}.png".format(num, shift/2/pi))
    return figqc, fig


def qfs(Nq, wq, Ec, wp, H, H_args, psi0, Nc, Np, Np_per_batch, options, home, parallel):
    i = wp[0]
    wp = wp[1]
    H_args['wp'] = wp
    b, nq = ops(Nq)
    e_ops = [nq]
        
    srcfolder = calculate(H, psi0, e_ops, H_args, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False)
    quants = ['times', 'expect']
    combine_batches(srcfolder, quants=quants, return_data=False)
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    if i < 10:
        num = "0" + str(i)
    elif i >= 10:
        num = str(i)
    
    print("\nshift  =", (wp-wq)/2/pi)
    print("wp     =", wp/2/pi)
    print("max    =", max(expect[0]))
    
    plt.figure(figsize=[15,3])
    plt.plot(times, expect[0], c='k')
    plt.title("shift {}, wp {}, max {}".format(np.round((wp-wq)/2/pi,4), np.round(wp/2/pi,4), np.round(max(expect[0]),4)))
    plt.savefig(home + "temp/fig{}_{}.png".format(num, (wp-wq)/2/pi))
    

def cfs(Nq, Nc, wc, Ec, wp, H, H_args, psi0, Np_per_batch, options, home, parallel):
    i = wp[0]
    wp = wp[1]
    H_args['wp'] = wp
    Np = 100*int(H_args['t3'])     # number of discrete time steps for which to store the output
    b, a, nq, nc = ops(Nq, Nc)  # Operators
    e_ops = [nq, nc]
    
    srcfolder = calculate(H, psi0, e_ops, H_args, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False)
    quants = ['times', 'expect']
    combine_batches(srcfolder, quants=quants, return_data=False)
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    if i < 10:
        num = "0" + str(i)
    elif i >= 10:
        num = str(i)
    
    print("wp     =", wp/2/pi)
    print("max    =", max(expect[1]))
    
    plt.figure(figsize=[15,3])
    plt.plot(times, expect[0], c='b')
    plt.plot(times, expect[1], c='r')
    plt.title("shift {}, wp {}, max {}".format(np.round((wp-wc)/2/pi,4), np.round(wp/2/pi,4), np.round(max(expect[1]),6)))
    plt.savefig(home + "temp/fig{}_{}.png".format(num, (wp-wc)/2/pi))