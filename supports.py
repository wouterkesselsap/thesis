import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from itertools import chain, groupby
from operator import itemgetter
from glob import glob
from copy import copy
from qutip import *


pi = np.pi
exp = np.exp
sqrt = np.sqrt
hbar = 1.0546e-34*1e-9  # reduced Planck constant
All = 'all'


psi_0pi = lambda Nq : basis(Nq, 0)
psi_0 = psi_0pi
psi_halfpi = lambda Nq : (basis(Nq, 0) + basis(Nq, 1)).unit()
psi_pi = lambda Nq : basis(Nq, 1)
psi_1pi = psi_pi
psi_2pi = psi_0pi


def ops(Nq, Nc):
    # Qubit operators
    b = tensor(destroy(Nq), qeye(Nc))
    nq = b.dag()*b

    # Cavity operators
    a = tensor(qeye(Nq), destroy(Nc))
    nc = a.dag()*a
    
    return b, a, nq, nc