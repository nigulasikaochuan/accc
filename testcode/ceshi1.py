from Ocspy.Channel import NonlinearFiber
from Ocspy.Base import QamSignal
from Ocspy.utilities import  generate_qam_signal
from Ocspy.utilities import to_wdm_array

import Ocspy.Instrument as ins
import Ocspy.tool as tool
#test when xpm change, whether the correlation will change

five_channel = generate_qam_signal([0]*5,35,is_wdm=1)
wdm_signal = ins.Multiplex.mux_signal(five_channel,50e9)
wdm_signal_array = to_wdm_array(wdm_signal,len(wdm_signal.signals)//2)
tool.spectrum_analyzer(wdm_signal_array)


# print(len(five_channel))