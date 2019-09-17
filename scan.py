import numpy as np
import matplotlib.pyplot as plt
import plots
from supports import *
from qutip import *
from scipy.special import erf
from scipy.signal import argrelextrema


def sample(Nq, wq, wc, wd, smooth, Q, t0, t1, t2, t3, tg, psi0, Np_per_batch, options):
    from supports import drive
    
    print(wd/2/pi)
    
    Nc = 10  # number of levels in resonator 1

    delta = wq - wc  # detuning
    Ec = 0.16 *2*pi  # anharmonicity (charging energy)
    g = 0.2 *2*pi  # drive frequency resonator 1, coupling between qubit and resonator 1
    chi = 30e-6 *2*pi  # photon number-dependent frequency shift of qubit

    Omega = 0.3 *2 *2*pi  # amplitude of sideband transitions
    wq_mod = wq + Omega**2/(2*(wq-wd)) + Omega**2/(2*(wq+wd))
    
    Np = 100*int(t3)     # number of discrete time steps for which to store the output
    
    # Qubit operators
    b = tensor(destroy(Nq), qeye(Nc))
    nq = b.dag()*b
    
    # Cavity operators
    a = tensor(qeye(Nq), destroy(Nc))
    nc = a.dag()*a
    
    # Jaynes-Cummings Hamiltonian
    Hjc = wq*nq + wc*nc - Ec/2*b.dag()*b.dag()*b*b - chi*nq*nc
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
    
    now, nowstr, folder = calculate(H, psi0, e_ops, H_args, options, Nc, Np, Np_per_batch, verbose=False)

    srcfolder = folder # "/home/student/thesis/"
    selection = All # (0, t3)
    reduction = 5
    quants = ['times', 'expect', 'e0', 'g1', 'e1', 'g0']

    start_comb = datetime.now()
    combine_batches(srcfolder, selection, reduction,
                    quants=quants, return_data=False)
    end_comb = datetime.now()
    
    times, states, expect, e0, g1, e1, g0 = load_data(quants, srcfolder)

    fig, ax1 = plt.subplots(figsize=[12,3])
    ax1.plot(times, expect[0], color='b', label='Qubit')
    ax1.plot(times, expect[1], color='r', label='Cavity')
    ax1.set_xlabel("$t$ [ns]")
    ax1.set_ylabel("$n$")
    ax1.tick_params(axis='y')
    ax1.legend(loc='center left')

    # ax1.set_xlim([10, 70])

    drive = wd/(2*pi)*drive_nonosc(times, H_args)
    ax2 = ax1.twinx()
    ax2.plot(times, drive, color='g', label='Drive, coupling')
    ax2.set_ylabel('$\Omega$, $g$ [GHz]')
    ax2.tick_params(axis='y')
    ax2.legend(loc='center right')

    plt.show()