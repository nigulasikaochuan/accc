from utilities import calc_xcorr
from Ocspy.ReceiverDsp import cd_compensation
from Ocspy.Base.SignalInterface import WdmSignalFromArray
from Ocspy.Instrument import Demultiplex, ADC
from Ocspy.ReceiverDsp import MatchedFilter
from Ocspy.Base import QamSignal
from CONF import ROLL_OFF, DATA_CONFIG
from Ocspy.tool import downsample
from Ocspy.tool import scatterplot
from Ocspy.ReceiverDsp import estimate_snr_using_tx
from Ocspy.ReceiverDsp.dsp_tools import normal_sample
import numpy as np
import joblib
import os


def receiver_onechannel(signal: QamSignal, spans):
    original_symbol = signal.symbol
    signal = cd_compensation(signal, spans, False)
    signal = ADC()(signal)
    signal = MatchedFilter(roll_off=ROLL_OFF, sps=signal.sps_in_fiber)(signal)

    signal = downsample(signal, signal.sps_in_fiber)
    phase = np.angle(np.sum(signal/original_symbol, axis=1, keepdims=True))
    signal = signal * np.exp(-1j*phase)

    snr = estimate_snr_using_tx(signal, original_symbol)
    signal = normal_sample(signal)
    return signal, snr[2]


def receiver_wdmsignal(wdmsignal: WdmSignalFromArray, index, spans):
    center_index2 = len(wdmsignal.signal_under_study)//2
    if wdmsignal.signal_index[center_index2]!=index:
        center_index2 = center_index2 +(index - wdmsignal.signal_index[center_index2])
    original_symbol = wdmsignal.signal_under_study[center_index2].symbol

    wdmsignal = cd_compensation(wdmsignal, spans, False)
    demultiplexed_signal = Demultiplex.demux_signal(
        wdmsignal, center_index2, roll_off=ROLL_OFF,freq_index=index)

    demultiplexed_signal = ADC()(demultiplexed_signal)
    demultiplexed_signal = MatchedFilter(
        roll_off=ROLL_OFF, sps=demultiplexed_signal.sps_in_fiber)(demultiplexed_signal)
    demultiplexed_signal = downsample(
        demultiplexed_signal, demultiplexed_signal.sps_in_fiber)
    phase = np.angle(np.sum(demultiplexed_signal /
                            original_symbol, axis=1, keepdims=True))
    demultiplexed_signal = demultiplexed_signal * np.exp(-1j*phase)

    snr = estimate_snr_using_tx(demultiplexed_signal, original_symbol)
    demultiplexed_signal = normal_sample(demultiplexed_signal)
    return demultiplexed_signal, snr[2]


def get_feature(wdm_signal: WdmSignalFromArray, spans, tocorrc_index, center_demod_symbol, max_lag=100):
    center_index = len(wdm_signal.signal_under_study)//2
    interfer_symbol,_ = receiver_wdmsignal(wdm_signal, tocorrc_index, spans)

    center_ori_symbol = wdm_signal.signal_under_study[center_index].symbol
    new_index = center_index + tocorrc_index-wdm_signal.signal_index[center_index]
    interf_ori_symbol = wdm_signal.signal_under_study[new_index].symbol
    noise_center = np.abs(center_demod_symbol)-np.abs(center_ori_symbol)
    noise_inter = np.abs(interfer_symbol) - np.abs(interf_ori_symbol)
    phase_noise_center = np.angle(center_demod_symbol/center_ori_symbol)
    phase_noise_inter = np.angle(interfer_symbol/interf_ori_symbol)

    amplitude_correlation = calc_xcorr(noise_center, noise_inter, max_lag)
    phase_corr = calc_xcorr(phase_noise_center, phase_noise_inter, max_lag*6)
    return amplitude_correlation, phase_corr


def get_total_snr(data_dir):
    import os
    names = os.listdir(data_dir)
    res = {}
    for name in names:
        all_information = joblib.load(data_dir+name)
        wdm_signal = all_information['wdm_signal_afterprop']
        center_index = len(wdm_signal.signal_under_study)//2
        center_index = wdm_signal.signal_index[center_index]
        spans = all_information['spans']
        center_receiver_symbol, snr = receiver_wdmsignal(
            wdm_signal, center_index, spans)
        res[name] = {
            'center_receive_symbol': center_receiver_symbol, 'total_snr': snr}
        break
    return res


def get_spm(data_dir):

    import os
    names = os.listdir(data_dir)
    res = {}
    for name in names:
        all_information = joblib.load(data_dir+name)
        only_center = all_information['center_signal_afterprop']
        spans = all_information['spans']
        _, snr = receiver_onechannel(only_center, spans)
        res[name] = snr
        break
    return res


def main_function():
    total_snr = get_total_snr(DATA_CONFIG)
    spm_snr = get_spm(DATA_CONFIG)
    xpm_snr = {}
    for key in total_snr:
        total_snr_lin = total_snr[key]['total_snr']
        total_snr_lin = 10**(total_snr_lin/10)
        spm_snr_lin = spm_snr[key]

        spm_snr_lin = 10**(spm_snr_lin/10)
        xpm_lin_nsr = 1/total_snr_lin - 1/spm_snr_lin
        xpm_lin_snr_ = 1/xpm_lin_nsr
        xpm_snr[key] = 10*np.log10(xpm_lin_snr_)
    names = os.listdir(DATA_CONFIG)
    feature_target = np.ones((len(names), 2+2+1))
    for cnt, name in enumerate(names):
        all_information = joblib.load(DATA_CONFIG+name)
        wdm_signal = all_information['wdm_signal_afterprop']
        spans = all_information['spans']

        center_index = len(wdm_signal.signal_under_study)//2
        center_index = wdm_signal.signal_index[center_index]
        left = center_index-1
        right = center_index+1
        center_demod_symbol = total_snr[name]['center_receive_symbol']
        left_feature = get_feature(
            wdm_signal, spans, left, center_demod_symbol)
        right_feature = get_feature(
            wdm_signal, spans, right, center_demod_symbol)
        target = xpm_snr[name]
        feature_target[cnt, :] = np.hstack(
            (np.array(right_feature), np.array(left_feature), target))
    joblib.dump(feature_target, 'dataall')

if __name__ == '__main__':
    main_function()