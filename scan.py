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


def sample_single_tone(Nq, wq, wc, Ec, g, Omega, shift, sb, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, options, parallel):
    from envelopes import drive
    
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

    # Jaynes-Cummings Hamiltonian
    Hjc = wq*nq + wc*nc - Ec/12*(b + b.dag())**4
    Hc = g*(a*b + a*b.dag() + b*a.dag() + a.dag()*b.dag())

    # Sideband transitions
    Hd = Omega*(b + b.dag())

    # Hamiltonian arguments
    H_args = {'t0' : t0, 't1' : t1, 't2' : t2,
              't3' : t3, 'tg' : tg, 'Q'  : Q,
              'smooth' : smooth, 'wd' : wd}

    # Expectation operators
    e_ops = [nq, nc]
        
    H = [Hjc, [Hc, drive_nonosc], [Hd, drive]]  # complete Hamiltonian
    
    progfolder = calculate(H, psi0, e_ops, H_args, options, Nc, g, Np, Np_per_batch, parallel, verbose=False)
    
    """ SAVE EVOLUTION TEMPORARILY """

    srcfolder =  progfolder
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()
    
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    print(" ")
    print("shift = {}, wd = {}".format(shift/2/pi, wd/2/pi))
    if sb == 'red':
        print("min = {}, max = {}".format(min(e0-g1), max(e0-g1)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, min(e0-g1), max(e0-g1))
    elif sb == 'blue':
        print("min = {}, max = {}".format(min(e1-g0), max(e1-g0)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, min(e1-g0), max(e1-g0))
    
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


def sample_double_tone(Nq, wq, wc, Ec, g, Omegaq, Omegac, shift, dw, sb, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, options, parallel):
    
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

    # Jaynes-Cummings Hamiltonian
    Hjc = wq*nq + wc*nc - Ec/12*(b + b.dag())**2
    Hc = g*(a*b + a*b.dag() + b*a.dag() + a.dag()*b.dag())

    # Sideband transitions
    Hdq = Omegaq*(b + b.dag())
    Hdc = Omegac*(b + b.dag())

    # Hamiltonian arguments
    H_args = {'t0' : t0, 't1' : t1, 't2' : t2, 't3' : t3, 'tg' : tg,
              'Q'  : Q, 'smooth' : smooth, 'Nt' : Nt, 'wdq' : wdq, 'wdc' : wdc}

    # Expectation operators
    e_ops = [nq, nc]
    
    H = [Hjc, [Hc, drive_nonosc], [Hdq, driveq], [Hdc, drivec]]  # complete Hamiltonian
    
    """ CALCULATE! """

    progfolder = calculate(H, psi0, e_ops, H_args, options, Nc, g, Np, Np_per_batch, parallel, verbose=False)
    
    """ SAVE EVOLUTION TEMPORARILY """
    
    srcfolder =  progfolder
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()
    
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    print(" ")
    print("shift = {}".format(shift/2/pi))
    if sb == 'red':
        print("min = {}, max = {}".format(min(e0-g1), max(e0-g1)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, min(e0-g1), max(e0-g1))
    elif sb == 'blue':
        print("min = {}, max = {}".format(min(e1-g0), max(e1-g0)))
        expect_title = "shift = {}".format(shift/2/pi)
        cp_title = "shift = {}, min = {}, max = {}".format(shift/2/pi, min(e1-g0), max(e1-g0))
        
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