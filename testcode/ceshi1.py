import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Ocspy.Channel import NonlinearFiber
from Ocspy.Base import QamSignal
from Ocspy.utilities import  generate_qam_signal
from Ocspy.utilities import to_wdm_array
from scipy.io import savemat
import Ocspy.Instrument as ins
import Ocspy.tool as tool
import joblib
#test when xpm change, whether the correlation will change
from utilities import prop

spans = NonlinearFiber(0.2,16.7,80,1/3,step_length=20/1000)
spans = [spans]*11
edfa = ins.Edfa(gain_db=16,nf=5,is_ase=False)
edfas = [edfa]*11


five_channel = generate_qam_signal([0]*5,35,is_wdm=1)
wdm_signal = ins.Multiplex.mux_signal(five_channel,50e9)
wdm_signal_array = to_wdm_array(wdm_signal,len(wdm_signal.signals)//2)
wdm_signal_array = prop(wdm_signal_array,spans,edfas)
joblib.dump(wdm_signal_array,'five_channel')

seven_channel = generate_qam_signal([0]*7,35,is_wdm=1)
wdm_signal = ins.Multiplex.mux_signal(five_channel,50e9)
wdm_signal_array = to_wdm_array(wdm_signal,len(wdm_signal.signals)//2)
wdm_signal_array = prop(wdm_signal_array,spans,edfas)
joblib.dump(wdm_signal_array,'seven_channel')
