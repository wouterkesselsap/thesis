import numpy as np
import scipy  #Imported for error function


nsigma=2
#Time dependency functions

#define Gaussian pulse
nsigmad1=nsigma;
W=scipy.special.erf(nsigma/np.sqrt(2))

def GaussEnvelope(u,t1,t2,Nsigma):
    mu=(t2+t1)/2
    tpulse=t2-t1
    sigma=tpulse/(2*Nsigma)
    W=scipy.special.erf(Nsigma/np.sqrt(2))

    Env=1/np.sqrt(2*np.pi*sigma**2)*(np.exp(-(u-mu)**2/(2*sigma**2))-np.exp(-(t2-mu)**2/(2*sigma**2)))/W*tpulse/(1-1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(t2-mu)**2/(2*sigma**2))/W*tpulse)

    return Env

def Resonance1(t,args):
    t0=args['t0']
    td1=args['td1']
    t1=args['t1']
    deltau=args['deltau']
    wd1=args['wd1']
    tstep=args['tstep']

    mud1=t0+td1/2
    sigmad1=td1/2/nsigmad1

    u=t%deltau
    #F=np.heaviside(u-t0,0.5)*np.sin(wd1*(u-t0))*np.heaviside(t1-u,0.5)*1/np.sqrt(2*np.pi*sigmad1**2)*(np.exp(-(u-mud1)**2/(2*sigmad1**2))-np.exp(-(t0-mud1)**2/(2*sigmad1**2)))/W*td1/(1-1/np.sqrt(2*np.pi*sigmad1**2)*np.exp(-(t0-mud1)**2/(2*sigmad1**2))/W*td1)
    #F=np.heaviside(u-t0,0.5)*np.heaviside(t1-u,0.5)*np.sin(wd1*(u-t0))
    F=(scipy.special.erf((u-t0)/tstep)-scipy.special.erf((u-t1)/tstep))*np.sin(wd1*(u-t0))
    return F

#define Gaussian pulse
nsigmad2=nsigma;

def Resonance2(t,args):
    t2=args['t2']
    td2=args['td2']
    t3=args['t3']
    deltau=args['deltau']
    wd2=args['wd2']
    tstep=args['tstep']

    mud2=t2+td2/2
    sigmad2=td2/2/nsigmad2;


    u=t%deltau
    #F=np.heaviside(u-t2,0.5)*np.sin(wd2*(t-t2))*np.heaviside(t3-u,0.5)*1/np.sqrt(2*np.pi*sigmad2**2)*(np.exp(-(u-mud2)**2/(2*sigmad2**2))-np.exp(-(t2-mud2)**2/(2*sigmad2**2)))/W*td2/(1-1/np.sqrt(2*np.pi*sigmad2**2)*np.exp(-(t2-mud2)**2/(2*sigmad2**2))/W*td2)
    #F=np.heaviside(u-t2,0.5)*np.sin(wd2*(t-t2))*np.heaviside(t3-u,0.5)*GaussEnvelope(u,t2,t3,nsigmad2)
    F=(scipy.special.erf((u-t2)/tstep)-scipy.special.erf((u-t3)/tstep))*np.sin(wd2*(u-t2))

    return F
def Rotate(t,args):
    return np.exp(0*1j*wp*t)


def Couple1(t,args):
    t0=args['t0']
    t1=args['t1']
    deltau=args['deltau']

    u=t%deltau
    F=np.heaviside(u-t0,0.5)*np.heaviside(t1-u,0.5)
    return F

def Couple2(t,args):
    t2=args['t2']
    t3=args['t3']
    deltau=args['deltau']

    u=t%deltau
    F=np.heaviside(u-t2,0.5)*np.heaviside(t3-u,0.5)
    return F

def One(t,args):
    return 1


#define Gaussian pulse


def PumpQubit(t,args):
    t4=args['t4']
    tpulse=args['tpulse']
    t5=args['t5']
    deltau=args['deltau']

    mu=t4+tpulse/2
    sigma=tpulse/2/nsigma;


    u=t%deltau

    #F=np.heaviside(u-t4,0.5)*np.heaviside(t5-u,0.5)*1/np.sqrt(2*np.pi*sigma**2)*(np.exp(-(u-mu)**2/(2*sigma**2))-np.exp(-(t4-mu)**2/(2*sigma**2)))/W*tpulse/(1-1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(t4-mu)**2/(2*sigma**2))/W*tpulse)
    F=np.heaviside(u-t4,0.5)*np.heaviside(t5-u,0.5)*GaussEnvelope(u,t4,t5,nsigma)
    return F



