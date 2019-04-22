import numpy as np
import matplotlib.pyplot as plt
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


def dmat(rho, obj='all', ind=None, roff=2):
    """Prints specified density matrix.
    
    Parameters:
    -----------
    rho: qutip.Result.states, or qutip.qobj
         (Collection of) density matrix/matrices
    obj: int
         Index of desired object in quantum system
    ind: int
         Index of specific density matrix to plot when rho is of qutip.Result.states format
    roff: int
          Number of decimals to round off to
    """
    
    if rho == []:
        print('dmat: Give density matrix as input')
        return
    
    if (ind == None and len(rho) == 1):
        dm = rho
    else:
        dm = rho[ind]
    
    if obj != 'all':
        if not isinstance(obj, int):
            raise ValueError("Give integer value for 'obj'")
        dm = dm.ptrace(obj)
    
    dm = np.round(dm.full(), roff)
    
    print('Density matrix:')
    print(dm)
    return dm


def dmatf(states, tlist, elem, obj='all', obj_descr=None):
    """Plots time evolution of density matrix elements.
    
    Parameters:
    -----------
    states: qutip.Result.states
            density matrices per time instant
    tlist: list, or numpy array
           times at which density matrices are evaluated, should be of same length as states
    elem: list
          matrix elements of which to plot time evolution
    obj: int
         Index of desired object in quantum system
    """
    


def wigner(rho, obj='all', ind=None, x=np.linspace(-3,3,200), y=np.linspace(-3,3,200)):
    """Plots Wigner function of specified density matrix.
    
    Parameters:
    -----------
    rho: qutip.Result.states, or qutip.qobj
         (Collection of) density matrix/matrices
    obj: int
         Index of desired object in quantum system
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
    
    if obj != 'all':
        if not isinstance(obj, int):
            raise ValueError("Give integer value for 'obj'")
        dm = dm.ptrace(obj)
        
    W = wig(dm, x, y)
    plt.figure(figsize=([6,5]))
    cont = plt.contourf(x, y, W, 100)
    plt.xlabel('x')
    plt.ylabel('p')
    plt.title('Wigner function')
    plt.colorbar()
    plt.show()
    return cont