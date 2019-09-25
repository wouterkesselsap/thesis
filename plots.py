"""
This module contains functions to visualize the results from the Landblad master equation solver of the QuTiP package, qutip.mesolve.
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm
from qutip.bloch import Bloch
from qutip.wigner import wigner as wig
from qutip.visualization import hinton, matrix_histogram, matrix_histogram_complex
from envelopes import *
from supports import *



def expect(ex_values, tlist, op=None, ops=None):
    """Plots expectation value of given operator.
    
    Parameters:
    -----------
    ex_values : qutip.Result.expect or list
                expectation values per operator
    tlist : list, or numpy array
            times at which expectation values are evaluated
    op : str
         specified operator of which to plot the expectation values
         'sx', or 'sy', or 'sz', or 'sm', or 'num_b'
    ops : list
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

    
    
def bloch(ex_values, tlist, ops):
    """Plots expectation values of sx, sy and sz on the Bloch sphere.
    If one of these operators is not calculated, zeros are passed for that operator.
    
    Parameters:
    -----------
    ex_values : qutip.Result.expect or list
                expectation values per operator
    tlist : list, or numpy array
            times at which expectation values are evaluated
    ops : list
          operators of which the expectation values are found in ex_values
    
    Remark:
    -------
    Does not plot a color bar yet. The lines from the QuTiP tutorial (https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/bloch_sphere_with_colorbar.ipynb), commented out down here, give an error which I have not been able to solve yet.
    """
    
    if 'sx' in ops:
        sx_exp = ex_values[ops.index('sx')]
    elif 'sx' not in ops:
        sx_exp = np.zeros(len(list(tlist)))
    if 'sy' in ops:
        sy_exp = ex_values[ops.index('sy')]
    elif 'sy' not in ops:
        sy_exp = np.zeros(len(list(tlist)))
    if 'sz' in ops:
        sz_exp = ex_values[ops.index('sz')]
    elif 'sz' not in ops:
        sz_exp = np.zeros(len(list(tlist)))
    
    b = Bloch()
    b.add_points([sx_exp, sy_exp, sz_exp], 'm')
    
    nrm = Normalize(-2,10)
    colors = cm.hot(nrm(tlist))
    b.point_color = list(colors)
    b.point_marker = ['o']
    # TODO: Plot color bar
#     left, bottom, width, height = [0.98, 0.05, 0.05, 0.9]
#     ax2 = b.fig.add_axes([left, bottom, width, height])
#     ColorbarBase(ax2, cmap=cm.hot, norm=nrm, orientation='vertical')
    b.show()



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



def dmatf(states, tlist, elems, obj='all', obj_descr=None):
    """Plots time evolution of density matrix elements.
    
    Parameters:
    -----------
    states: qutip.Result.states
            density matrices per time instant
    tlist: list, or numpy array
           times at which density matrices are evaluated, should be of same length as states
    elems: list of lists
           matrix elements (as [k,l]) of which to plot time evolution
    obj: int
         Index of desired object in quantum system
    obj_descr: string
               Manual description of which object of the quantum system is plotted
    """
    
    rhos = []
    for i, t in enumerate(tlist):
        if obj != 'all':
            if not isinstance(obj, int):
                raise ValueError("Give integer value for 'obj'")
            rhos.append(states[i].ptrace(obj).full())
        else:
            rhos.append(states[i].full())
    dms = np.asarray(rhos)
    
    plt.figure()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(elems)))
    for i, elem in enumerate(elems):
        color = colors[i]
        re = dms[:, elem[0], elem[1]].real
        im = dms[:, elem[0], elem[1]].imag
        plt.plot(tlist*10**9, re, '-',  color=color, label='Re({},{})'.format(elem[0], elem[1]))
        plt.plot(tlist*10**9, im, '--', color=color, label='Im({},{})'.format(elem[0], elem[1]))
    plt.xlabel('Time [ns]')
    if obj_descr != None:
        plt.title('Time evolution of $\\rho$ of the {}'.format(obj_descr))
    plt.legend()
    plt.show


    
def dmat_hinton(rho, obj='all', ind=None):
    """Plots Hinton diagram of specified density matrix.
    
    Parameters:
    -----------
    rho: qutip.Result.states
         density matrices per time instant
    obj: int
         Index of desired object in quantum system
    ind: int
         Index of specific density matrix to plot when rho is of qutip.Result.states format
    
    Remark:
    -------
    Title is not set in correct place.
    """
    
    if isinstance(rho, list):
        dm = rho[ind]
    else:
        dm = rho
    
    if obj != 'all':
        if not isinstance(obj, int):
            raise ValueError("Give integer value for 'obj'")
        dm = dm.ptrace(obj)
    
    # TODO: Show title in correct location above the diagram
    if obj == 0:
        title = "Hinton diagram of qubit"
    elif (isinstance(obj, int) and obj > 0):
        title = "Hinton diagram of cavity {}".format(obj)
    elif obj == 'all':
        title = "Hinton diagram of total system"
    
    fig, ax = hinton(dm, title=title)
    plt.title(title)


    
def dmat_hist(rho, obj='all', ind=None, im=False):
    """Plots 3D histogram of specified density matrix.
    
    Parameters:
    -----------
    rho: qutip.Result.states
         density matrices per time instant
    obj: int
         Index of desired object in quantum system
    ind: int
         Index of specific density matrix to plot when rho is of qutip.Result.states format
    im: boolean
        Include imaginary part
    """
    
    if isinstance(rho, list):
        dm = rho[ind]
    else:
        dm = rho
    
    if obj != 'all':
        if not isinstance(obj, int):
            raise ValueError("Give integer value for 'obj'")
        dm = dm.ptrace(obj)
    
    if obj == 0:
        title = "Histogram of density matrix of qubit"
    elif (isinstance(obj, int) and obj > 0):
        title = "Histogram of density matrix of cavity {}".format(obj)
    elif obj == 'all':
        title = "Histogram of density matrix of total system"
    
    if im:
        matrix_histogram_complex(dm.full(), title=title)
    else:
        matrix_histogram(dm.full().real, title=title)



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
    plt.plot([0, 0], [min(y), max(y)], 'k--')
    plt.plot([min(x), max(x)], [0, 0], 'k--')
    plt.xlabel('x')
    plt.ylabel('p')
    if obj == 0:
        title = "Wigner plot of qubit"
    elif (isinstance(obj, int) and obj > 0):
        title = "Wigner plot of cavity {}".format(obj)
    elif obj == 'all':
        title = "Wigner plot of total system"
    plt.title(title)
    plt.colorbar()
    plt.show()
    return cont


def sb_expect(times, expect, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], **kwargs):
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(times, expect[0], color='b', label='Qubit')
    ax1.plot(times, expect[1], color='r', label='Cavity')
    ax1.plot([times[0], times[-1]], [1, 1], ':', color='k')
    ax1.plot([times[0], times[-1]], [1/2, 1/2], ':', color='k')
    ax1.plot([times[0], times[-1]], [0, 0], ':', color='k')
    ax1.set_xlabel("$t$ [ns]")
    ax1.set_ylabel("$n$")
    ax1.tick_params(axis='y')
    ax1.legend(loc='center left')

    if xlim != None:
        ax1.set_xlim(xlim)
    elif ylim != None:
        ax1.set_ylim(ylim)

    if Nt == 1:
        wd = kwargs['wd']
        drive_coupling = wd/(2*pi)*coupling
    elif Nt == 2:
        drive_coupling = coupling
    ax2 = ax1.twinx()
    ax2.plot(times, drive_coupling, color='g', label='Drive, coupling')
    ax2.set_ylabel('$\Omega$ [GHz]')
    ax2.tick_params(axis='y')
    ax2.legend(loc='center right')

    figqc = plt.gcf()
    
    wsb = kwargs['wsb']
    if sb == 'red':
        if Nt == 1:
            plt.title("Single-tone red sideband transitions at $\\nu_d$ = {} GHz ($\\nu_{{sb}}$ = {} MHz)".format(
                      round(wd/2/pi, 3), int(round(1000*wsb/2/pi))))
        elif Nt == 2:
            plt.title("Double-tone red sideband transitions ($\\nu_{{sb}}$ = {} MHz)".format(
                      int(round(1000*wsb/2/pi))))
    elif sb == 'blue':
        if Nt == 1:
            plt.title("Single-tone blue sideband transitions at $\\nu_d$ = {} GHz ($\\nu_{{sb}}$ = {} MHz)".format(
                      round(wd/2/pi, 3), int(round(1000*wsb/2/pi))))
        elif Nt == 2:
            plt.title("Double-tone blue sideband transitions ($\\nu_{{sb}}$ = {} MHz)".format(
                      int(round(1000*wsb/2/pi))))
    plt.show()
    
    return figqc


def sb_combined_probs(times, sb, Nt, H_args, coupling, xlim=None, ylim=None, figsize=[15,3], **kwargs):
    fig, ax1 = plt.subplots(figsize=figsize)
    if sb == 'red':
        e0 = kwargs['e0']
        g1 = kwargs['g1']
        ax1.plot(times, e0-g1, color='r', label='$P(e0) - P(g1)$')
    if sb == 'blue':
        e1 = kwargs['e1']
        go = kwargs['g0']
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
    
    if xlim != None:
        ax1.set_xlim(xlim)
    if ylim != None:
        ax1.set_ylim(ylim)

    ax2 = ax1.twinx()
    if Nt == 1:
        wd = kwargs['wd']
        drive_coupling = wd/(2*pi)*coupling
        ax2.set_ylabel('$\Omega$, $g$ [GHz]')
    elif Nt == 2:
        drive_coupling = coupling
        ax2.set_ylabel('$\Omega$, $g$ [a.u.]')
    ax2.plot(times, drive_coupling, color='g', label='Drive, coupling')
    ax2.tick_params(axis='y')
    ax2.legend(loc='center right')

    fig = plt.gcf()
    
    wsb = kwargs['wsb']
    if sb == 'red':
        if Nt == 1:
            plt.title("Single-tone red sideband transitions at $\\nu_d$ = {} GHz ($\\nu_{{sb}}$ = {} MHz)".format(
                      round(wd/2/pi, 3), int(round(1000*wsb/2/pi))))
        elif Nt == 2:
            plt.title("Double-tone red sideband transitions ($\\nu_{{sb}}$ = {} MHz)".format(
                      int(round(1000*wsb/2/pi))))
    elif sb == 'blue':
        if Nt == 1:
            plt.title("Single-tone blue sideband transitions at $\\nu_d$ = {} GHz ($\\nu_{{sb}}$ = {} MHz)".format(
                      round(wd/2/pi, 3), int(round(1000*wsb/2/pi))))
        elif Nt == 2:
            plt.title("Double-tone blue sideband transitions ($\\nu_{{sb}}$ = {} MHz)".format(
                      int(round(1000*wsb/2/pi))))
    plt.show()
    
    return fig