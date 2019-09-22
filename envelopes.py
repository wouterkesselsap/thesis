import numpy as np
from scipy.special import erf
from math import ceil
from copy import copy
from qutip import *
from supports import *


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