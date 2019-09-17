import os
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
psi_0 = psi_0pi
psi_halfpi = lambda Nq : (basis(Nq, 0) + basis(Nq, 1)).unit()
psi_pi = lambda Nq : basis(Nq, 1)
psi_1pi = psi_pi
psi_2pi = psi_0pi


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


def drive(t, args):
    "Sideband drive with oscillating term."
    t1 = args['t1']    # start of pulse
    t2 = args['t2']    # end of pulse
    tg = args['tg']    # time of Gaussian rise and fall
    smooth = args['smooth']  # whether or not to rise and fall with gaussian
    Q  = args['Q']     # number of std's in Gaussian rise and fall
    wd = args['wd']  # drive frequency
    
    confine = np.heaviside((t-t1), 0) - np.heaviside((t-t2), 0)  # entire pulse
    
    # Rise and fall with Gaussian
    std = tg/Q  # standard deviation of Gaussian
    gauss = lambda mu : exp(-(t-mu)**2/(2*std**2))  # Gaussian
    
    if smooth:
        block = np.heaviside((t-(t1+tg)), 0) - np.heaviside((t-(t2-tg)), 0)
        rise = gauss(t1+tg) * (1-np.heaviside((t-(t1+tg)), 0))
        fall = gauss(t2-tg) * (np.heaviside((t-(t2-tg)), 0))
        pulse = (rise + block + fall)*confine
    else:
        pulse = confine
    
    envelope = pulse*np.cos(wd*t)
    return envelope
    
    
def drive_nonosc(t, args):
    "Sideband drive without oscillating term."
    t1 = args['t1']    # start of pulse
    t2 = args['t2']    # end of pulse
    tg = args['tg']    # time of Gaussian rise and fall
    smooth = args['smooth']  # whether or not to rise and fall with gaussian
    Q  = args['Q']     # number of std's in Gaussian rise and fall
    
    confine = np.heaviside((t-t1), 0) - np.heaviside((t-t2), 0)  # entire pulse
    
    # Rise and fall with Gaussian
    std = tg/Q  # standard deviation of Gaussian
    gauss = lambda mu : exp(-(t-mu)**2/(2*std**2))  # Gaussian
    
    if smooth:
        block = np.heaviside((t-(t1+tg)), 0) - np.heaviside((t-(t2-tg)), 0)
        rise = gauss(t1+tg) * (1-np.heaviside((t-(t1+tg)), 0))
        fall = gauss(t2-tg) * (np.heaviside((t-(t2-tg)), 0))
        pulse = (rise + block + fall)*confine
    else:
        pulse = confine
    
    return pulse


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


def saveprog(result, e0, g1, e1, g0, num, folder):
    name = folder + "/evolution_" + str(num) + ".pkl"
    data = {
        'times': result.times,
        'states': result.states,
        'expect': result.expect,
        'e0' : e0,
        'g0' : g0,
        'g1' : g1,
        'e1' : e1,
        'num': num
    }
    
    out_file = open(name, "wb")
    pickle.dump(data, out_file)
    out_file.close()


def combine_batches(folder, selection='all', reduction=1, quants='all', return_data=True):
    """
    folder : str
    selection : tuple (T1, T2)
        Range of time for which to save the data
    reduction : int
        1/fraction of data points to save
    quants : list of str
        Can contain 'times', 'states' and 'expect'
    """
    if quants == 'all':
        quants = ['times', 'states', 'expect', 'g0', 'g1', 'e0', 'e1']
    if isinstance(quants, str):
        quants = [quants]
    
    # Remove already existing files with combined batches
    try:
        os.remove(folder + "/times.pkl")
    except:
        pass
    try:
        os.remove(folder + "/states.pkl")
    except:
        pass
    try:
        os.remove(folder + "/expect.pkl")
    except:
        pass
    try:
        os.remove(folder + "/g0.pkl")
    except:
        pass
    try:
        os.remove(folder + "/g1.pkl")
    except:
        pass
    try:
        os.remove(folder + "/e0.pkl")
    except:
        pass
    try:
        os.remove(folder + "/e1.pkl")
    except:
        pass
    
    condition = folder + "/evolution_*"
    filecount = len(glob(condition))
    
    if 'times' in quants:
        times = [None] * filecount
        for file in glob(condition):
            infile = open(file, 'rb')
            data = pickle.load(infile)
            if data['num'] == 0:
#                     sill = data['times']
#                     if ( min(sill) <= selection[0] and max(sill) >= selection[0] ):
#                         # batch contains lower bound
				
                times[data['num']] = data['times']
            else:
                times[data['num']] = data['times'][1:]
            infile.close()
        times_combined = np.asarray(list(chain.from_iterable(times)))
        
        name = folder + "/times.pkl"
        data = {
            'quantity' : 'times',
            'data'     : times_combined,
        }
        out_file = open(name, 'wb')
        pickle.dump(data, out_file)
        out_file.close()
        
        del times
        if not return_data:
            del times_combined
    else:
        times_combined = None
    
    if 'states' in quants:
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
            'quantity' : 'states',
            'data' : states_combined
        }
        out_file = open(name, "wb")
        pickle.dump(data, out_file)
        out_file.close()
        
        del states
        if not return_data:
            del states_combined
    else:
        states_combined = None
		
    if 'expect' in quants:
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
            'quantity' : 'expect',
            'data' : expect_combined
        }
        out_file = open(name, "wb")
        pickle.dump(data, out_file)
        out_file.close()
        
        del expect
        if not return_data:
            del expect_combined
    else:
        expect_combined = None
	
    if 'g0' in quants:
        g0 = [None] * filecount
        for file in glob(condition):
            infile = open(file, 'rb')
            data = pickle.load(infile)
            if data['num'] == 0:				
                g0[data['num']] = data['g0']
            else:
                g0[data['num']] = data['g0'][1:]
            infile.close()
        g0_combined = np.asarray(list(chain.from_iterable(g0)))
        
        name = folder + "/g0.pkl"
        data = {
            'quantity' : 'g0',
            'data'     : g0_combined,
        }
        out_file = open(name, 'wb')
        pickle.dump(data, out_file)
        out_file.close()
        
        del g0
        if not return_data:
            del g0_combined
    else:
        g0_combined = None
	
    if 'g1' in quants:
        g1 = [None] * filecount
        for file in glob(condition):
            infile = open(file, 'rb')
            data = pickle.load(infile)
            if data['num'] == 0:				
                g1[data['num']] = data['g1']
            else:
                g1[data['num']] = data['g1'][1:]
            infile.close()
        g1_combined = np.asarray(list(chain.from_iterable(g1)))
        
        name = folder + "/g1.pkl"
        data = {
            'quantity' : 'g1',
            'data'     : g1_combined,
        }
        out_file = open(name, 'wb')
        pickle.dump(data, out_file)
        out_file.close()
        
        del g1
        if not return_data:
            del g1_combined
    else:
        g1_combined = None
		
    if 'e0' in quants:
        e0 = [None] * filecount
        for file in glob(condition):
            infile = open(file, 'rb')
            data = pickle.load(infile)
            if data['num'] == 0:				
                e0[data['num']] = data['e0']
            else:
                e0[data['num']] = data['e0'][1:]
            infile.close()
        e0_combined = np.asarray(list(chain.from_iterable(e0)))
        
        name = folder + "/e0.pkl"
        data = {
            'quantity' : 'e0',
            'data'     : e0_combined,
        }
        out_file = open(name, 'wb')
        pickle.dump(data, out_file)
        out_file.close()
        
        del e0
        if not return_data:
            del e0_combined
    else:
        e0_combined = None
	
    if 'e1' in quants:
        e1 = [None] * filecount
        for file in glob(condition):
            infile = open(file, 'rb')
            data = pickle.load(infile)
            if data['num'] == 0:				
                e1[data['num']] = data['e1']
            else:
                e1[data['num']] = data['e1'][1:]
            infile.close()
        e1_combined = np.asarray(list(chain.from_iterable(e1)))
        
        name = folder + "/e1.pkl"
        data = {
            'quantity' : 'e1',
            'data'     : e1_combined,
        }
        out_file = open(name, 'wb')
        pickle.dump(data, out_file)
        out_file.close()
        
        del e1
        if not return_data:
            del e1_combined
    else:
        e1_combined = None
    
    if return_data:
        return times_combined, states_combined, expect_combined, e0_combined, g1_combined, e1_combined, g0_combined


def load_data(quants, srcfolder):
    if quants == 'all':
        quants = ['times', 'states', 'expect', 'g0', 'g1', 'e0', 'e1']
    if isinstance(quants, str):
        quants = [quants]
    
    if 'times' in quants:
        tfile = open(srcfolder + "/times.pkl", 'rb')
        tdata = pickle.load(tfile)
        times = tdata['data']
        tfile.close()
        del tdata
    else:
        times = None
    
    if 'states' in quants:
        sfile = open(srcfolder + "/states.pkl", 'rb')
        sdata = pickle.load(sfile)
        states = sdata['data']
        sfile.close()
        del sdata
    else:
        states = None
    
    if 'expect' in quants:
        efile = open(srcfolder + "/expect.pkl", 'rb')
        edata = pickle.load(efile)
        expect = edata['data']
        efile.close()
        del edata
    else:
        expect = None
    
    if 'e0' in quants:
        pfile = open(srcfolder + "/e0.pkl", 'rb')
        pdata = pickle.load(pfile)
        e0 = pdata['data']
        pfile.close()
        del pdata
    else:
        e0 = None
    
    if 'g1' in quants:
        pfile = open(srcfolder + "/g1.pkl", 'rb')
        pdata = pickle.load(pfile)
        g1 = pdata['data']
        pfile.close()
        del pdata
    else:
        g1 = None

    if 'e1' in quants:
        pfile = open(srcfolder + "/e1.pkl", 'rb')
        pdata = pickle.load(pfile)
        e1 = pdata['data']
        pfile.close()
        del pdata
    else:
        e1 = None
    
    if 'g0' in quants:
        pfile = open(srcfolder + "/g0.pkl", 'rb')
        pdata = pickle.load(pfile)
        g0 = pdata['data']
        pfile.close()
        del pdata
    else:
        g0 = None
    
    return times, states, expect, e0, g1, e1, g0


def combined_probs(states, Nc):
    """Calculates |e,0> - |g,1> and |e,1> - |g,0>."""
    inds = ((1,0), (0,1), (1,1), (0,0))
    probs = list()
    [probs.append(list()) for i in range(len(inds))]
    
    for i, ind in enumerate(inds):
        for state in states:
            probs[i].append((state.data[ind[1] + Nc*ind[0], 0]*state.data[ind[1] + Nc*ind[0], 0].conj()).real)
    
    e0 = np.asarray(probs[0])
    g1 = np.asarray(probs[1])
    e1 = np.asarray(probs[2])
    g0 = np.asarray(probs[3])
    return e0, g1, e1, g0