import numpy as np
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


def ops(*args):
    """
    Returns system operators based on input arguments. The order
    of the input arguments defines the convention of order of the
    elements of the system (for example qubits and cavities).
    
    Input
    -----
    *args : int
        Every argument represents the number of levels of a system element
    
    Returns
    -------
    a# : qutip.Qobj class object
        Lowering operator for element #
    n# : qutip.Qobj class object
        Number operator for element #
    """
    if len(args) == 1:
        a0 = destroy(args[0])
        n0 = a0.dag()*a0
        return a0, n0
    
    elif len(args) == 2:
        a0 = destroy(args[0])
        n0 = a0.dag()*a0
        a1 = destroy(args[1])
        n1 = a1.dag()*a1
        return a0, a1, n0, n1
    
    elif len(args) == 3:
        a0 = destroy(args[0])
        n0 = a0.dag()*a0
        a1 = destroy(args[1])
        n1 = a1.dag()*a1
        a2 = destroy(args[2])
        n2 = a2.dag()*a2
        return a0, a1, a2, n0, n1, n2
    
    elif len(args) == 4:
        a0 = destroy(args[0])
        n0 = a0.dag()*a0
        a1 = destroy(args[1])
        n1 = a1.dag()*a1
        a2 = destroy(args[2])
        n2 = a2.dag()*a2
        a3 = destroy(args[3])
        n3 = a3.dag()*a3
        return a0, a1, a2, a3, n0, n1, n2, n3