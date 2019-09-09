import numpy as np
import pickle
from scipy.special import erf
from math import ceil
from itertools import chain
from glob import glob
from qutip import *



pi = np.pi
exp = np.exp
sqrt = np.sqrt
hbar = 1.0546e-34*1e-9  # reduced Planck constant
All = 'all'


psi_0pi = lambda Nq : basis(Nq, 0)
psi_halfpi = lambda Nq : (basis(Nq, 0) + basis(Nq, 1)).unit()
psi_pi = lambda Nq : basis(Nq, 1)


def pump_strength(args):
    t0 = args['t0']
    t1 = args['t1']
    Q  = args['Q']  # number of std's in Gaussian rise and fall
    tau = t1-t0      # pulse duration
    std = tau/(2*Q)  # standard deviation
    time = lambda t : t/(std*sqrt(2))  # t' to pass to error function
    
    """
    TODO: include qubit decay rate kq
    """
    
    integral = sqrt(2*pi)*std*erf(time(tau)/2)
    Omega = pi/integral
    return Omega


def pump(t, args):
    """Just Gaussian envelope."""
    t0 = args['t0']  # start of pulse
    t1 = args['t1']  # end of pulse
    t6 = args['t6']  # end of cycle
    Q  = args['Q']   # number of std's in Gaussian rise and fall
    
    t = t%t6  # repeat cycle
    mu = (t1-t0)/2  # pulse center in time domain
    std = (t1-t0)/(2*Q)  # standard deviation
    confine = np.heaviside((t-t0), 0) - np.heaviside((t-t1), 0)  # entire pulse
    
    pulse = exp(-(t-mu)**2/(2*std**2))*confine
    return pulse


def pumpdrive(t, args):
    """Gaussian envelope and rotating wave."""
    t0 = args['t0']  # start of pulse
    t1 = args['t1']  # end of pulse
    t6 = args['t6']  # end of cycle
    Q  = args['Q']   # number of std's in Gaussian rise and fall
    wd = args['wd']  # drive frequency
    
    t = t%t6  # repeat cycle
    mu = (t1-t0)/2  # pulse center in time domain
    std = (t1-t0)/(2*Q)  # standard deviation
    confine = np.heaviside((t-t0), 0) - np.heaviside((t-t1), 0)  # entire pulse
    
    pulse = exp(-(t-mu)**2/(2*std**2))*confine
    envelope = pulse*np.cos(wd*t)
    return envelope


def pumpdrive_b(t, args):
    """Gaussian envelope and rotating wave."""
    t0 = args['t0']  # start of pulse
    t1 = args['t1']  # end of pulse
    t6 = args['t6']  # end of cycle
    Q  = args['Q']   # number of std's in Gaussian rise and fall
    wd = args['wd']  # drive frequency
    
    t = t%t6  # repeat cycle
    mu = (t1-t0)/2  # pulse center in time domain
    std = (t1-t0)/(2*Q)  # standard deviation
    confine = np.heaviside((t-t0), 0) - np.heaviside((t-t1), 0)  # entire pulse
    
    pulse = exp(-(t-mu)**2/(2*std**2))*confine
    envelope = pulse*np.exp(1j*wd*t)
    return envelope


def pumpdrive_bdag(t, args):
    """Gaussian envelope and rotating wave."""
    t0 = args['t0']  # start of pulse
    t1 = args['t1']  # end of pulse
    t6 = args['t6']  # end of cycle
    Q  = args['Q']   # number of std's in Gaussian rise and fall
    wd = args['wd']  # drive frequency
    
    t = t%t6  # repeat cycle
    mu = (t1-t0)/2  # pulse center in time domain
    std = (t1-t0)/(2*Q)  # standard deviation
    confine = np.heaviside((t-t0), 0) - np.heaviside((t-t1), 0)  # entire pulse
    
    pulse = exp(-(t-mu)**2/(2*std**2))*confine
    envelope = pulse*np.exp(-1j*wd*t)
    return envelope


def sideband(t, args):
    wsb = args['wsb']
    return np.cos(wsb*t)


def square1(t, args):
    t2 = args['t2']  # start of pulse
    t3 = args['t3']  # end of pulse
    t6 = args['t6']  # end of cycle
    tg = args['tg']  # time of Gaussian rise and fall
    Q  = args['Q']   # number of std's in Gaussian rise and fall
    smooth = args['smooth']  # rise and fall with Gaussian or not
    
    t = t%t6  # repeat cycle
    confine = np.heaviside((t-t2), 0) - np.heaviside((t-t3), 0)  # entire pulse
    
    # Rise and fall with Gaussian
    std = tg/Q  # standard deviation of Gaussian
    gauss = lambda mu : exp(-(t-mu)**2/(2*std**2))  # Gaussian
    
    if smooth:
        block = np.heaviside((t-(t2+tg)), 0) - np.heaviside((t-(t3-tg)), 0)
        rise = gauss(t2+tg) * (1-np.heaviside((t-(t2+tg)), 0))
        fall = gauss(t3-tg) * (np.heaviside((t-(t3-tg)), 0))
        pulse = (rise + block + fall)*confine
    else:
        pulse = confine
    
    return pulse


def square2(t, args):
    t4 = args['t4']  # start of pulse
    t5 = args['t5']  # end of pulse
    t6 = args['t6']  # end of cycle
    tg = args['tg']  # time of Gaussian rise and fall
    g2 = args['g2']  # pulse strength
    Q  = args['Q']   # number of std's in Gaussian rise and fall
    smooth = args['smooth']  # rise and fall with Gaussian or not
    
    t = t%t6  # repeat cycle
    confine = np.heaviside((t-t4), 0) - np.heaviside((t-t5), 0)  # entire pulse
    
    # Rise and fall with Gaussian
    std = tg/Q  # standard deviation of Gaussian
    gauss = lambda mu : exp(-(t-mu)**2/(2*std**2))  # Gaussian
    
    if smooth:
        block = np.heaviside((t-(t4+tg)), 0) - np.heaviside((t-(t5-tg)), 0)
        rise = gauss(t4+tg) * (1-np.heaviside((t-(t4+tg)), 0))
        fall = gauss(t5-tg) * (np.heaviside((t-(t5-tg)), 0))
        pulse = (rise + block + fall)*confine
    else:
        pulse = confine
    
    return pulse


def create_batches(t0, tf, Np, Nppb):
    """Creates a tuple with batches of time points for which to
    sequentially evaluate the LME evolution.
    t0 : start time
    tf : final time
    Np : total number of equidistant time points
    Nppb : number of time points per batch
    """
    tlist = np.linspace(t0, tf, Np)
    Nppb = ceil(Nppb)
    Nb = int(ceil(Np/Nppb))
    tlist = np.append(tlist, np.inf*np.ones(int(ceil(Nb*Nppb - Np))))
    tlist_reshaped = tlist.reshape(Nb, int(ceil(Nppb)))
    batches = list()
    for i in range(Nb):
        if i == 0:
            batches.append(tlist_reshaped[0])
        elif (i > 0 and i < Nb-1):
            batches.append(np.insert(tlist_reshaped[i], 0, tlist_reshaped[i-1][-1]))
        elif i == Nb-1:
            batch = np.insert(tlist_reshaped[i], 0, tlist_reshaped[i-1][-1])
            batches.append(batch[batch != np.inf])
    return batches


def saveprog(result, num, folder):
    name = folder + "/evolution_" + str(num) + ".pkl"
    data = {
        'times': result.times,
        'states': result.states,
        'expect': result.expect,
        'num': num
    }
    
    out_file = open(name, "wb")
    pickle.dump(data, out_file)
    out_file.close()


def combine_batches(folder, quants, return_data=True):
    """
    folder : str
    quants : list of str
        Can contain 'times', 'states' and 'expect'
    """
    if quants == 'all':
        quants = ['times', 'states', 'expect']
    if isinstance(quants, str):
        quants = [quants]
    
    condition = folder + "/evolution_*"
    filecount = len(glob(condition))
    
    for quant in quants:
        
        if quant == 'times':
            times = [None] * filecount
            for file in glob(condition):
                infile = open(file, 'rb')
                data = pickle.load(infile)
                if data['num'] == 0:
                    times[data['num']] = data['times']
                else:
                    times[data['num']] = data['times'][1:]
                infile.close()
            times_combined = np.asarray(list(chain.from_iterable(times)))
            
            name = folder + "/times.pkl"
            data = {
                'quantity' : quant,
                'data' : times_combined
            }
            out_file = open(name, "wb")
            pickle.dump(data, out_file)
            out_file.close()
            
            del times
            if not return_data:
                del times_combined
            
        elif quant == 'states':
            states = [None] * filecount
            for file in glob(condition):
                infile = open(file, 'rb')
                data = pickle.load(infile)
                if data['num'] == 0:
                    states[data['num']] = data['states']
                else:
                    states[data['num']] = data['states'][1:]
                infile.close()
            states_combined = list()
            for lst in states:
                for state in lst:
                    states_combined.append(state)
            
            name = folder + "/states.pkl"
            data = {
                'quantity' : quant,
                'data' : states_combined
            }
            out_file = open(name, "wb")
            pickle.dump(data, out_file)
            out_file.close()
            
            del states
            if not return_data:
                del states_combined
            
        elif quant == 'expect':
            expect = [None] * filecount
            for file in glob(condition):
                infile = open(file, 'rb')
                data = pickle.load(infile)
                if data['num'] == 0:
                    expect[data['num']] = data['expect']
                else:
                    expect[data['num']] = data['expect']
                infile.close()
            expect_combined = list()
            for op in expect[0]:
                expect_combined.append(list())
            for i in range(len(expect[0])):
                for ib, batch in enumerate(expect):
                    [expect_combined[i].append(value) for value in batch[i] if ib == 0]
                    [expect_combined[i].append(value) for iv, value in enumerate(batch[i]) if (ib > 0 and iv > 0)]
            
            name = folder + "/expect.pkl"
            data = {
                'quantity' : quant,
                'data' : expect_combined
            }
            out_file = open(name, "wb")
            pickle.dump(data, out_file)
            out_file.close()
            
            del expect
            if not return_data:
                del expect_combined
    
    if return_data:
        return times_combined, states_combined, expect_combined