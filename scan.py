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


def sbsample(Nq, wq, wc, Ec, g, wd, sb, Nt, H, H_args, psi0, Np_per_batch,
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
    Nt = 1 : eps
    Nt = 2 : epsq, epsc, dw
    
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
    
    i = wd[0]
    wd = wd[1]
    
    Nc = 10  # number of levels in resonator 1
    
    Np = 100*int(H_args['t3'])  # number of discrete time steps for which to store the output
    b, a, nq, nc = ops(Nq, Nc)  # Operators
    if Nt == 1:
        H_args['wd'] = wd
    elif Nt == 2:
        dw = args[2]
        wdq = wd
        wdc =  wc - dw
        H_args['wdq'] = wdq
        H_args['wdc'] = wdc
    e_ops = [nq, nc]
    c_ops = []
        
    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, options, Nc, Np, Np_per_batch,
                          home, parallel, verbose=False, method='me')
    
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
        eps = args[0]
        print("wd = {}".format(wd/2/pi))
        if sb == 'red':
            print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
            expect_title = "wd = {}".format(wd/2/pi)
            cp_title = "wd = {}, min = {}, max = {}".format(wd/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wd=wd, wsb=0, title=expect_title, eps=eps)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    e0=e0, g1=g1, wd=wd, wsb=0, title=cp_title, eps=eps)
        elif sb == 'blue':
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            expect_title = "wd = {}".format(wd/2/pi)
            cp_title = "wd = {}, min = {}, max = {}".format(wd/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wd=wd, wsb=0, title=expect_title, eps=eps)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    e1=e1, g0=g0, wd=wd, wsb=0, title=cp_title, eps=eps)
        fig.savefig(home + "temp/fig{}_{}.png".format(num, wd/2/pi))
        figqc.savefig(home + "temp/figqc{}_{}.png".format(num, wd/2/pi))
    
    elif Nt == 2:
        epsq = args[0]
        epsc = args[1]
        print("wdq = {}".format(wdq/2/pi))
        if sb == 'red':
            print("min = {}, max = {}".format(round(min(e0-g1), 4), round(max(e0-g1), 4)))
            expect_title = "wdq = {}".format(wdq/2/pi)
            cp_title = "wdq = {}, min = {}, max = {}".format(wdq/2/pi, round(min(e0-g1), 4), round(max(e0-g1), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wsb=0, title=expect_title, epsq=epsq, epsc=epsc)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    xlim=None, ylim=None, figsize=[15,3], e0=e0, g1=g1, wsb=0,
                                    title=cp_title, epsq=epsq, epsc=epsc)
        elif sb == 'blue':
            print("min = {}, max = {}".format(round(min(e1-g0), 4), round(max(e1-g0), 4)))
            expect_title = "wdq = {}".format(wdq/2/pi)
            cp_title = "wdq = {}, min = {}, max = {}".format(wdq/2/pi, round(min(e1-g0), 4), round(max(e1-g0), 4))
            figqc, axqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3],
                              wsb=0, title=expect_title, epsq=epsq, epsc=epsc)
            fig, axp = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                    xlim=None, ylim=None, figsize=[15,3], e1=e1, g0=g0, wsb=0,
                                    title=cp_title, epsq=epsq, epsc=epsc)
        fig.savefig(home + "temp/fig{}_{}.png".format(num, wdq/2/pi))
        figqc.savefig(home + "temp/figqc{}_{}.png".format(num, wdq/2/pi))
    plt.close(fig)
    plt.close(figqc)
    return figqc, fig


def sbsample_visualize_sweep(Nq, wq, wc, Ec, g, wd, sb, Nt, H, H_args, psi0, Np_per_batch,
             options, home, parallel, *args):
    from envelopes import drive
    
    i = wd[0]
    wd = wd[1]
    
    Nc = 10  # number of levels in resonator 1
    
    Np = 100*int(H_args['t3'])  # number of discrete time steps for which to store the output
    b, a, nq, nc = ops(Nq, Nc)  # Operators
    if Nt == 1:
        H_args['wd'] = wd
    elif Nt == 2:
        dw = args[2]
        wdq = wd
        wdc =  wc - dw
        H_args['wdq'] = wdq
        H_args['wdc'] = wdc
    e_ops = [nq, nc]
    c_ops = []
        
    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, options, Nc, Np, Np_per_batch,
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
        eps = args[0]
        print("$\\omega_d /2\\pi = ${} GHz".format(wd/2/pi))
        if sb == 'red':
            expect_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(wd/2/pi, 4))
            cp_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(wd/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              wd=wd, wsb=0, title=expect_title, eps=eps)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling, figsize=[12,4],
                                    e0=e0, g1=g1, wd=wd, wsb=0, xlim=[0,1000], ylim=[-1.02, 1.02], title=cp_title, eps=eps)
        elif sb == 'blue':
            expect_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(wd/2/pi, 4))
            cp_title = "$\\omega_d /2\\pi = ${} GHz".format(np.round(wd/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              wd=wd, wsb=0, title=expect_title, eps=eps)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling, figsize=[12,4],
                                    e1=e1, g0=g0, wd=wd, wsb=0, xlim=[0,1000], ylim=[-1.02, 1.02], title=cp_title, eps=eps)
    
    elif Nt == 2:
        epsq = args[0]
        epsc = args[1]
        print("$\\omega_dq /2\\pi = ${} GHz".format(np.round(wdq/2/pi, 4)))
        if sb == 'red':
            expect_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(wdq/2/pi, 4))
            cp_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(wdq/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              wsb=0, title=expect_title, epsq=epsq, epsc=epsc)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling,
                                   xlim=[0,1000], ylim=[-1.02, 1.02], figsize=[15,3], e0=e0, g1=g1, wsb=0,
                                    title=cp_title, epsq=epsq, epsc=epsc)
        elif sb == 'blue':
            expect_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(wdq/2/pi, 4))
            cp_title = "$\\omega_dq /2\\pi = ${} GHz".format(np.round(wdq/2/pi, 4))
            figqc, axqc = sb_expect_temporary(times, expect, sb, Nt, H_args, coupling, xlim=[0,1000], ylim=[-0.02, 1.02], figsize=[12,4],
                              wsb=0, title=expect_title, epsq=epsq, epsc=epsc)
            fig, axp = sb_combined_probs_temporary(times, sb, Nt, H_args, coupling, 
                                    xlim=[0,1000], ylim=[-1.02, 1.02], figsize=[12,4], e1=e1, g0=g0, wsb=0,
                                    title=cp_title, epsq=epsq, epsc=epsc)
    fig.savefig("/Users/Wouter/Documents/TU/TN/Master/Thesis project/Midterm presentation/freq sweep method/fig_{}.png".format(num))
    figqc.savefig("/Users/Wouter/Documents/TU/TN/Master/Thesis project/Midterm presentation/freq sweep method/figqc_{}.png".format(num))
    plt.close(fig)
    plt.close(figqc)
    return figqc, fig


def qfs(Nq, wq, Ec, wp, H, H_args, psi0, Nc, Np, Np_per_batch, options, home, parallel):
    """
    Applies a probe tone to an uncoupled qubit to find its transition frequency,
    which can be shifted due to a dispersive drive.
    
    Input
    -----
    The input parameters are equal to the names in 2p_sideband.ipynb.
    """
    i = wp[0]
    wp = wp[1]
    H_args['wp'] = wp
    b, nq = ops(Nq)
    e_ops = [nq]
    c_ops = []
        
    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, options, Nc, Np, Np_per_batch,
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
    c_ops = []
    
    srcfolder = calculate(H, psi0, e_ops, c_ops, H_args, options, Nc, Np, Np_per_batch,
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
#    plt.plot(times, expect[0], c='b')
    plt.plot(times, expect[1], c='r')
    plt.title("shift {}, wp {}, max {}".format(np.round((wp-wc)/2/pi,4), np.round(wp/2/pi,4), np.round(max(expect[1]),6)))
    plt.savefig(home + "temp/fig{}_{}.png".format(num, (wp-wc)/2/pi))
