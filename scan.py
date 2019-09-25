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
options = Options()
options.store_states=True


def sample_single_tone(Nq, wq, wc, wd, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, options):
    from envelopes import drive
    
    print(wd/2/pi)
    
    Nc = 10  # number of levels in resonator 1

    delta = wq - wc  # detuning
    Ec = 0.16 *2*pi  # anharmonicity (charging energy)
    g = 0.2 *2*pi  # drive frequency resonator 1, coupling between qubit and resonator 1
    chi = 30e-6 *2*pi  # photon number-dependent frequency shift of qubit

    Omega = 0.3 *2 *2*pi  # amplitude of sideband transitions
    wq_mod = wq + Omega**2/(2*(wq-wd)) + Omega**2/(2*(wq+wd))
    
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
    
    """ SAVE EVOLUTION """

    srcfolder = progfolder #"/home/student/thesis/blue"
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()

    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)

    fig, ax1 = plt.subplots(figsize=[15,3])
    ax1.plot(times, expect[0], color='b', label='Qubit')
    ax1.plot(times, expect[1], color='r', label='Cavity')
    ax1.plot([times[0], times[-1]], [1, 1], ':', color='k')
    ax1.plot([times[0], times[-1]], [1/2, 1/2], ':', color='k')
    ax1.plot([times[0], times[-1]], [0, 0], ':', color='k')
    ax1.set_xlabel("$t$ [ns]")
    ax1.set_ylabel("$n$")
    ax1.tick_params(axis='y')
    ax1.legend(loc='center left')

    # ax1.set_xlim([10, 70])

    drive = wd/(2*pi)*coupling
    ax2 = ax1.twinx()
    ax2.plot(times, drive, color='g', label='Drive, coupling')
    ax2.set_ylabel('$\Omega$, $g$ [GHz]')
    ax2.tick_params(axis='y')
    ax2.legend(loc='center right')

    plt.show()


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
    
    H = [Hjc, [Hc, drive_nonosc], [Hdq, drive_qubit], [Hdc, drive_cavity]]  # complete Hamiltonian
    
    """ CALCULATE! """

    progfolder = calculate(H, psi0, e_ops, H_args, options, Nc, g, Np, Np_per_batch, verbose=False)
    
    """ SAVE EVOLUTION TEMPORARILY """

    srcfolder =  progfolder #"/home/student/thesis/prog_190922_144723"
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0']

    start_comb = datetime.now()
    new_folder_name = copy(srcfolder)
    ID = getID(srcfolder)
    combine_batches(srcfolder, quants=quants, return_data=False)
    end_comb = datetime.now()

    times, states, expect, e0, g1, e1, g0, coupling = load_data(quants, srcfolder)
    
    fig, ax1 = plt.subplots(figsize=[15,3])
    ax1.plot(times, expect[0], color='b', label='Qubit')
    ax1.plot(times, expect[1], color='r', label='Cavity')
    ax1.plot([times[0], times[-1]], [1, 1], ':', color='k')
    ax1.plot([times[0], times[-1]], [1/2, 1/2], ':', color='k')
    ax1.plot([times[0], times[-1]], [0, 0], ':', color='k')
    ax1.set_xlabel("$t$ [ns]")
    ax1.set_ylabel("$n$")
    ax1.tick_params(axis='y')
    ax1.legend(loc='center left')

    # ax1.set_xlim([10, 70])

    drive = couplingCo
    ax2 = ax1.twinx()
    ax2.plot(times, drive, color='g', label='Drive, coupling')
    ax2.set_ylabel('$\Omega$, $g$ [GHz]')
    ax2.tick_params(axis='y')
    ax2.legend(loc='center right')

    plt.show()
    
    fig, ax1 = plt.subplots(figsize=[15,3])
    if sb == 'red':
        ax1.plot(times, e0-g1, color='r', label='$P(e0) - P(g1)$')
    if sb == 'blue':
        ax1.plot(times, e1-g0, color='b', label='$P(e1) - P(g0)$')
    ax1.plot([times[0], times[-1]], [1, 1], ':', color='k')
    ax1.plot([times[0], times[-1]], [1/2, 1/2], ':', color='k')
    ax1.plot([times[0], times[-1]], [0, 0], ':', color='k')
    ax1.plot([times[0], times[-1]], [-1/2, -1/2], ':', color='k')
    ax1.plot([times[0], times[-1]], [-1, -1], ':', color='k')
    ax1.set_xlabel("$t$ [ns]")
    ax1.set_ylabel("$P$")
    ax1.tick_params(axis='y')
    ax1.legend(loc='center left')

    # ax1.set_xlim([40, 50])

    ax2 = ax1.twinx()
    if Nt == 1:
        drive_coupling = wd/(2*pi)*drive_nonosc(times, H_args)
        ax2.set_ylabel('$\Omega$, $g$ [GHz]')
    elif Nt == 2:
        drive_coupling = drive_nonosc(times, H_args)
        ax2.set_ylabel('$\Omega$, $g$ [a.u.]')
    ax2.plot(times, drive_coupling, color='g', label='Drive, coupling')
    ax2.tick_params(axis='y')
    ax2.legend(loc='center right')
    
    plt.show()
    
    print(min(e0-g1))