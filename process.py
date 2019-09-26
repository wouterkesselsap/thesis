import os
import shutil
import pickle
import time
import numpy as np
from math import ceil
from glob import glob
from datetime import datetime
from itertools import chain, groupby
from operator import itemgetter
from supports import *


def prepare_folder():
    # Remove existing progress folder
    for folder in glob("/home/student/thesis/prog_*"):
        shutil.rmtree(folder)

    # Make new progress folder
    now = datetime.now()
    ID = now.strftime("%y%m%d_%H%M%S")
    folder = "/home/student/thesis/prog_" + ID
    os.makedirs(folder)
    saveID(ID, folder)
    
    return ID, folder, now


def saveparams(Nq, Nc, Nt, wq, shift, wc, Ec, g, sb,
               t0, t1, t2, t3, tg, smooth, Q,
               Np, H, psi0, e_ops, options,
               folder, frmt, **kwargs):
    """Save all parameters to file.
    
    frmt : str, or list of str
        Can be "txt" and "pkl"
    """
    if isinstance(frmt, str):
        frmt = [frmt]
    
    if 'pkl' in frmt:
        data = {
            'Nq' : Nq, 'Nc' : Nc, 'Nt' : Nt, 'wq' : wq, 'shift' : shift, 'wc' : wc, 'Ec' : Ec, 'g' : g, 'sb' : sb,
            't0' : t0, 't1' : t1, 't2' : t2, 't3' : t3, 'tg' : tg, 'smooth' : smooth, 'Q' : Q,
            'Np' : Np, 'H' : H, 'psi0' : psi0, 'e_ops' : e_ops, 'options' : options        
        }
        if Nt == 1:
            data['Omega'] = kwargs['Omega']
            data['wd'] = kwargs['wd']
        elif Nt == 2:
            data['Omegaq'] = kwargs['Omegaq']
            data['Omegac'] = kwargs['Omegac']
            data['dw'] = kwargs['dw']
            data['wdq'] = kwargs['wdq']
            data['wdc'] = kwargs['wdc']

        name = folder + "/parameters.pkl"
        pklfile = open(name, "wb")
        pickle.dump(data, pklfile)
        pklfile.close()
    
    if 'txt' in frmt:
        data = ["PRESET PARAMETERS\n",
                "-----------------\n\n"
                "# of qubit levels               Nq     : {}\n".format(Nq),
                "# of cavity levels              Nc     : {}\n".format(Nc),
                "# of tones in drive             Nt     : {}\n".format(Nt),
                "qubit transition frequency      wq     : {} = {} GHz\n".format(wq, wq/2/pi),
                "qubit's ac-Stark shift          shift  : {} = {} GHz\n".format(shift, shift/2/pi),
                "cavity frequency                wc     : {} = {} GHz\n".format(wc, wc/2/pi),
                "anharmonicity                   Ec     : {} = {} GHz\n".format(Ec, Ec/2/pi),
                "qubit-cavity coupling strength  g      : {} = {} GHz\n".format(g, g/2/pi),
                "sideband transition             sb     : {}\n".format(sb),
                "start of simulation             t0     : {} ns\n".format(t0),
                "start of sideband drive         t1     : {} ns\n".format(t1),
                "end of sideband drive           t2     : {} ns\n".format(t2),
                "end of simulation               t3     : {} ns\n".format(t3),
                "rise and fall time              tg     : {} ns\n".format(tg),
                "rise and fall with Gaussian     smooth : {}\n".format(smooth),
                "# of std's in Gaussian          Q      : {}\n".format(Q),
                "# of data points                Np     : {}\n".format(Np)]
                
        if Nt == 1:
            data.append("sideband drive amplitude        Omega  : {} = {} GHz\n".format(kwargs['Omega'], kwargs['Omega']/2/pi))
            data.append("sideband drive frequency        wd     : {} = {} GHz\n".format(kwargs['wd'], kwargs['wd']/2/pi))
        elif Nt == 2:
            data.append("amplitude of qubit-friendly sideband drive tone   Omegaq : {} = {} GHz\n".format(kwargs['Omegaq'], kwargs['Omegaq']/2/pi))
            data.append("frequency of qubit-friendly sideband drive tone   wdq    : {} = {} GHz\n".format(kwargs['wdq'], kwargs['wdq']/2/pi))
            data.append("amplitude of cavity-friendly sideband drive tone  Omegac : {} = {} GHz\n".format(kwargs['Omegac'], kwargs['Omegac']/2/pi))
            data.append("frequency of cavity-friendly sideband drive tone  wdc    : {} = {} GHz\n".format(kwargs['wdc'], kwargs['wdc']/2/pi))
            data.append("detuning delta from wc                            dw     : {} = {} GHz\n".format(kwargs['dw'], kwargs['dw']/2/pi))
        
        data.append("\nCALCULATED PARAMETERS\n")
        data.append("---------------------\n\n")
        
        name = folder + "/parameters.txt"
        txtfile = open(name, "w")
        txtfile.writelines(data)
        txtfile.close()


def getparams(folder):
    file = folder + "/parameters.pkl"
    infile = open(file, 'rb')
    data = pickle.load(infile)
    
    Nq = data['Nq']
    Nc = data['Nc']
    Nt = data['Nt']
    wq = data['wq']
    shift = data['shift']
    wc = data['wc']
    Ec = data['Ec']
    g = data['g']
    sb = data['sb']
    t0 = data['t0']
    t1 = data['t1']
    t2 = data['t2']
    t3 = data['t3']
    tg = data['tg']
    smooth = data['smooth']
    Q = data['Q']
    Np = data['Np']
    H = data['H']
    psi0 = data['psi0']
    e_ops = data['e_ops']
    options = data['options']
    
    if Nt == 1:
        Omega = data['Omega']
        wd = data['wd']
        Omegaq = None
        Omegac = None
        dw = None
        wdq = None
        wdc = None
    elif Nt == 2:
        Omega = None
        wd = None
        Omegaq = data['Omegaq']
        Omegac = data['Omegac']
        dw = data['dw']
        wdq = data['wdq']
        wdc = data['wdc']
    
    infile.close()
            
    return Nq, Nc, Nt, wq, shift, wc, Ec, g, sb, t0, t1, t2, t3, tg, smooth, Q, Np, H, psi0, e_ops, options, Omega, wd, Omegaq, Omegac, dw, wdq, wdc
            
            

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


def saveprog(result, e0, g1, e1, g0, coupling, num, folder):
    name = folder + "/evolution_" + str(num) + ".pkl"
    data = {
        'times': result.times,
        'states': result.states,
        'expect': result.expect,
        'e0' : e0,
        'g0' : g0,
        'g1' : g1,
        'e1' : e1,
        'coupling' : coupling,
        'num': num
    }
    
    out_file = open(name, "wb")
    pickle.dump(data, out_file)
    out_file.close()


def saveID(ID, folder):
    name = folder + "/ID.pkl"
    outfile = open(name, 'wb')
    pickle.dump(ID, outfile)
    outfile.close()


def getID(folder):
    file = folder + "/ID.pkl"
    infile = open(file, 'rb')
    ID = pickle.load(infile)
    infile.close()
    return ID


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
        quants = ['times', 'states', 'expect', 'g0', 'g1', 'e0', 'e1', 'coupling']
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
    try:
        os.remove(folder + "/e1.pkl")
    except:
        pass
    try:
        os.remove(folder + "/coupling.pkl")
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
    
    if 'coupling' in quants:
        coupling = [None] * filecount
        for file in glob(condition):
            infile = open(file, 'rb')
            data = pickle.load(infile)
            if data['num'] == 0:				
                coupling[data['num']] = data['coupling']
            else:
                coupling[data['num']] = data['coupling'][1:]
            infile.close()
        coupling_combined = np.asarray(list(chain.from_iterable(coupling)))
        
        name = folder + "/coupling.pkl"
        data = {
            'quantity' : 'coupling',
            'data'     : coupling_combined,
        }
        out_file = open(name, 'wb')
        pickle.dump(data, out_file)
        out_file.close()
        
        del coupling
        if not return_data:
            del coupling_combined
    else:
        coupling_combined = None
    
    if return_data:
        return times_combined, states_combined, expect_combined, e0_combined, g1_combined, e1_combined, g0_combined, coupling_combined


def load_data(quants, srcfolder):
    if quants == 'all':
        quants = ['times', 'states', 'expect', 'g0', 'g1', 'e0', 'e1', 'coupling']
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
    
    if 'coupling' in quants:
        gfile = open(srcfolder + "/coupling.pkl", 'rb')
        gdata = pickle.load(gfile)
        coupling = gdata['data']
        gfile.close()
        del gdata
    else:
        coupling = None
    
    return times, states, expect, e0, g1, e1, g0, coupling


def getquants(folder):
    quants = list()
    condition = folder + "/*.pkl"
    
    for file in glob(condition):
        
        if ('expect' in file and 'expect' not in quants):
            quants.append('expect')
        elif ('states' in file and 'states' not in quants):
            quants.append('states')
        elif ('times' in file and 'times' not in quants):
            quants.append('times')
        elif ('coupling' in file and 'coupling' not in quants):
            quants.append('coupling')
        elif ('g0' in file and 'g0' not in quants):
            quants.append('g0')
        elif ('g1' in file and 'g1' not in quants):
            quants.append('g1')
        elif ('e0' in file and 'e0' not in quants):
            quants.append('e0')
        elif ('e1' in file and 'e1' not in quants):
            quants.append('e1')
    
    return quants