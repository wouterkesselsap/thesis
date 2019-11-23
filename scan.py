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
             options, home, parallel, *args, **kwargs):
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
    plt.close(fig)
    plt.close(figqc)
    
    return figqc, fig


def sbsample_visualize_sweep(Nq, wq, wc, Ec, g, shift, sb, Nt, H, H_args, psi0,
                             Np_per_batch, options, home, parallel, *args):
    from envelopes import drive
    
    small = 14
    medium = 16
    big = 18
    
    plt.rc('font', size=medium)          # controls default text sizes
    plt.rc('axes', titlesize=big)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium)    # fontsize of the tick labels
    plt.rc('legend', fontsize=medium)    # legend fontsize
    plt.rc('figure', titlesize=big)     # fontsize of the figure title
    plt.rc('lines', linewidth=3)
    
    alpha = 0.6    
    
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
            expect_title = "$\\omega_d / 2\\pi = {}$ GHz".format(np.round(wd/2/pi, 4))
            cp_title = "$\\omega_d / 2\\pi = {}$ GHz".format(np.round(wd/2/pi, 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=[0, 1.02], figsize=[15,4],
                              wd=wd, wsb=0, title=expect_title, Omega=Omega, alpha=alpha)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    e0=e0, g1=g1, wd=wd, wsb=0, ylim=[-1.02, 1.02], title=cp_title, Omega=Omega, alpha=alpha)
        elif sb == 'blue':
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            expect_title = "$\\omega_d / 2\\pi = {}$ GHz".format(np.round(wd/2/pi, 4))
            cp_title = "$\\omega_d / 2\\pi = {}$ GHz".format(np.round(wd/2/pi, 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=[0, 1.02], figsize=[15,4],
                              wd=wd, wsb=0, title=expect_title, Omega=Omega, alpha=alpha)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    e1=e1, g0=g0, wd=wd, wsb=0, ylim=[-1.02, 1.02], title=cp_title, Omega=Omega, alpha=alpha)   
        fig.savefig("/Users/Wouter/Documents/TU/TN/Master/Thesis project/Midterm presentation/freq sweep method/fig_{}.png".format(wd/2/pi))
        figqc.savefig("/Users/Wouter/Documents/TU/TN/Master/Thesis project/Midterm presentation/freq sweep method/figqc_{}.png".format(wd/2/pi))
    
    elif Nt == 2:
        Omegaq = args[0]
        Omegac = args[1]
        print("shift = {}".format(shift/2/pi))
        if sb == 'red':
            print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
            expect_title = "$\\omega_{{d_{{q}}}} / 2\\pi = {}$ GHz".format(np.round(wdq/2/pi, 4))
            cp_title = "$\\omega_{{d{{q}}}} / 2\\pi = {}$ GHz".format(np.round(wdq/2/pi, 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=[0, 1.02], figsize=[15,4],
                              wsb=0, title=expect_title, Omegaq=Omegaq, Omegac=Omegac, alpha=alpha)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    xlim=None, ylim=[-1.02, 1.02], figsize=[15,4], e0=e0, g1=g1, wsb=0,
                                    title=cp_title, Omegaq=Omegaq, Omegac=Omegac, alpha=alpha)
        elif sb == 'blue':
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            expect_title = "$\\omega_{{d_{{q}}}} / 2\\pi = {}$ GHz".format(np.round(wdq/2/pi, 4))
            cp_title = "$\\omega_{{d{{q}}}} / 2\\pi = {}$ GHz".format(np.round(wdq/2/pi, 4))
            figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=[0, 1.02], figsize=[15,4],
                              wsb=0, title=expect_title, Omegaq=Omegaq, Omegac=Omegac, alpha=alpha)
            fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    xlim=None, ylim=[-1.02, 1.02], figsize=[15,4], e1=e1, g0=g0, wsb=0,
                                    title=cp_title, Omegaq=Omegaq, Omegac=Omegac, alpha=alpha)
        fig.savefig("/Users/Wouter/Documents/TU/TN/Master/Thesis project/Midterm presentation/freq sweep method/fig_{}.png".format(wdq/2/pi))
        figqc.savefig("/Users/Wouter/Documents/TU/TN/Master/Thesis project/Midterm presentation/freq sweep method/figqc_{}.png".format(wdq/2/pi))
        
    plt.close(fig)
    plt.close(figqc)
    return figqc, fig