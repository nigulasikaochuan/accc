from Ocspy.Channel import NonlinearFiber
from Ocspy.Base import QamSignal
from Ocspy.utilities import  generate_qam_signal

#test when xpm change, whether the correlation will change

five_channel = generate_qam_signal([0]*5,35)
print(len(five_channel))