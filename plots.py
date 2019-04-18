import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from qutip.ipynbtools import plot_animation
from qutip.wigner import wigner as wig


def expect(ex_values, tlist, op=None, ops=None):
    """Plots expectation value of given operator.
    
    Parameters:
    -----------
    ex_values: qutip.Result.expect or list
               expectation values per operator
    tlist: list, or numpy array
           times at which expectation values are evaluated
    op: str
        specified operator of which to plot the expectation values
        'sx', or 'sy', or 'sz', or 'sm', or 'num_b'
    ops: list
         operators of which the expectation values are found in ex_values
    """
    
    if op == None:
        vals = ex_values
    elif op != None:
        vals = ex_values[ops.index(op)]
    plt.plot(tlist*10**9, vals)
    plt.xlabel('Time [ns]')
    if op != None:
        plt.title('Expectation value of {}'.format(op))
        if op == 'sx':
            plt.ylabel('$\\langle\\sigma_x\\rangle$')
        elif op == 'sy':
            plt.ylabel('$\\langle\\sigma_y\\rangle$')
        elif op == 'sz':
            plt.ylabel('$\\langle\\sigma_z\\rangle$')
        elif op == 'sm':
            plt.ylabel('$\\langle\\sigma_m\\rangle$')
        elif op == 'num_b':
            plt.ylabel('Cavity photon number')
    plt.show()


def dmat(rho, ind=None, roff=2):
    """Prints specified density matrix.
    
    Parameters:
    -----------
    rho: qutip.Result.states, or qutip.qobj
         (Collection of) density matrix/matrices
    ind: int
         Index of specific density matrix to plot when rho is of qutip.Result.states format
    roff: int
          Number of decimals to round off to
    """
    
    if rho == []:
        print('dmat: Give density matrix as input')
        return
    
    if (ind == None and len(rho) == 1):
        dm = np.round(rho, roff)
    else:
        dm = np.round(rho[ind].full(), roff)
    print('Density matrix:')
    print(dm)
    return dm


def wigner(rho, ind=None, x=np.linspace(-3,3,200), y=np.linspace(-3,3,200)):
    """Plots Wigner function of specified density matrix.
    
    Parameters:
    -----------
    rho: qutip.Result.states, or qutip.qobj
         (Collection of) density matrix/matrices
    ind: int
         Index of specific density matrix to plot when rho is of qutip.Result.states format
    x: list, or numpy array
       Phase space x-values to plot
    y: list, or numpy array
       Phase space y-values to plot
    """
    
    if rho == []:
        print('wigner: Give density matrix as input')
        return
        
    if (ind == None and len(rho) == 1):
        dm = rho
    else:
        dm = rho[ind]
        
    W = wig(dm, x, y)
    plt.figure(figsize=([6,5]))
    cont = plt.contourf(x, y, W, 100)
    plt.xlabel('x')
    plt.ylabel('p')
    plt.title('Wigner function')
    plt.colorbar()
    plt.show()
    return cont