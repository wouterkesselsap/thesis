import stlab
from stlab.devices.PNAN5222A import PNAN5222A
from stlab.devices import autodetect_instrument #Import device driver for your desired PNA
import numpy as np
import time
import stlab.advanced_measurements.TwoToneSpectroscopy as tts
from stlab.devices.Keysight_N5183B import Keysight_N5183B


prefix = 'B1' #prefix for measurement folder name.  Can be anything or empty
idstring = 'Two-tone-LT142W4_3D_C_40dB-prove_att_40dB-pump_red-sideband_3'  #Additional info included in m easurement folder name.  Can be anything or empty
pna = PNAN5222A("TCPIP::192.168.1.221::INSTR", reset=False,verb=True) #Initialize device but do not reset.
mw = Keysight_N5183B("TCPIP::192.168.1.91::INSTR", reset=False,verb=True) #Initialize device but do not reset.

################################################################
# power sweeps
# drive setting
###############################################################

mw.RFon() 
mw.setCWpower(19)  
     
fstart = 1.108e9  
fstop = 1.112e9  
steps = 41
flist = np.linspace(fstart,fstop,steps) #generate power sweep steps

pna.SetPower(-20) #set pna power
pna.SetRange(8.44e9,8.46e9) #Set frequency range in Hz
pna.SetIFBW(100) #Set IF bandwidth in Hz
pna.SetPoints(201) #Set number of frequency points   

pna.write('SENS1:AVER ON')  
for i,f in enumerate(flist):
    mw.setCWfrequency(f) #mw source freq
    data = pna.MeasureScreen() 
    pna.write('SENS1:AVER:COUN 1000') 
    naver = int(pna.query('SENS1:AVER:COUN?'))
    for ii in range(naver-1):
        pna.query("INIT1:IMM;*OPC?")
    if i==0: #if on first measurement, create new m easurement file and folder using titles extracted from measurement
        myfile = stlab.newfile(prefix,idstring,data.keys())
    stlab.savedict(myfile, data) #Save measured data to file.  Written as a block for spyview.
    stlab.metagen.fromarrays(myfile,data['Frequency (Hz)'],flist[0:i+1],xtitle='Frequency (Hz)',ytitle='drive freq (Hz)',colnames=data.keys())
mw.RFoff()
myfile.close() #Close file
