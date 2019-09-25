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


def sample_single_tone(Nq, wq, wc, shift, sb, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, options):
    from envelopes import drive
    
    Nc = 10  # number of levels in resonator 1
    Nt = 1   # number of drive tones

    delta = wq - wc    # detuning
    Ec = 0.16 *2*pi    # anharmonicity (charging energy)
    g = 0.2 *2*pi      # coupling between qubit and resonator
    
    Omega = 0.3 *2 *2*pi  # pump drive amplitude
    if sb == 'red':
        if wq > wc:
            wd = (wq + shift - wc)/2
        elif wq < wc:
            wd = (wc - wq - shift)/2
    elif sb == 'blue':
        wd = (wq + shift + wc)/2
    
    print(shift/2/pi, wd/2/pi)
        
    Np = 100*int(t3)     # number of discrete time steps for which to store the output
    
    # Operators
    b, a, nq, nc = ops(Nq, Nc)

    # Jaynes-Cummings Hamiltonian
    Hjc = wq*nq + wc*nc - Ec/2*b.dag()*b.dag()*b*b
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
    
    progfolder = calculate(H, psi0, e_ops, H_args, options, Nc, g, Np, Np_per_batch, verbose=False)
    
    """ SAVE EVOLUTION TEMPORARILY """

    srcfolder =  progfolder
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()
    
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)

    """ EXPECTATION VALUES """

    if sb == 'red':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wd=wd, wsb=0)
    elif sb == 'blue':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wd=wd, wsb=0)
    
    """COMBINED PROBABILITIES"""

    if sb == 'red':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling, e0=e0, g1=g1, wd=wd, wsb=0)
    elif sb == 'blue':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling, e1=e1, g0=g0, wd=wd, wsb=0)
    
    if sb == 'red':
        print(min(e0-g1), max(e0-g1))
    elif sb == 'blue':
        print(min(e1-g0), max(e1-g0))


def sample_double_tone(Nq, wq, wc, shift, dw, sb, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, options):
    print(shift/2/pi)
    
    Nc = 10  # number of levels in resonator 1
    Nt = 2   # number of drive tones

    delta = wq - wc    # detuning
    Ec = 0.16 *2*pi    # anharmonicity (charging energy)
    g = 0.2 *2*pi      # coupling between qubit and resonator
    
    Omegaq = 0.025 *2 *2*pi  # amplitude of qubit-friendly drive tone
    Omegac = 0.317 *2 *2*pi  # amplitude of cavity-friendly drive tone
    
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
    Hjc = wq*nq + wc*nc - Ec/2*b.dag()*b.dag()*b*b
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

    progfolder = calculate(H, psi0, e_ops, H_args, options, Nc, g, Np, Np_per_batch, verbose=False)
    
    """ SAVE EVOLUTION TEMPORARILY """
    
    srcfolder =  progfolder
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0', 'coupling']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()
    
    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
        
    """ EXPECTATION VALUES """

    if sb == 'red':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wsb=0)
    elif sb == 'blue':
        figqc = sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], wsb=0)
    
    """COMBINED PROBABILITIES"""

    if sb == 'red':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                xlim=None, ylim=None, figsize=[15,3], e0=e0, g1=g1, wsb=0)
    elif sb == 'blue':
        fig = sb_combined_probs(times, sb, Nt, H_args, coupling,
                                xlim=None, ylim=None, figsize=[15,3], e1=e1, g0=g0, wsb=0)
    
    if sb == 'red':
        print(min(e0-g1), max(e0-g1))
    elif sb == 'blue':
        print(min(e1-g0), max(e1-g0))