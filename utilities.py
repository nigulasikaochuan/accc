from typing import List
from pycorrelate import ucorrelate
import numpy as np
import tqdm
from Ocspy.Base import QamSignal
from Ocspy.Channel import NonlinearFiber
from Ocspy.Instrument import Edfa, AWG, Multiplex
from Ocspy.utilities import calc_sps_in_fiber, to_wdm_array

from CONF import PSCF, SPACING, ROLL_OFF
from CONF import SSMF

SPLIT_CONFIG = -100000


def read_config(config: List):
    start_index1 = config.index(SPLIT_CONFIG)
    start_index2 = config.index(SPLIT_CONFIG, start_index1 + 1)
    start_index3 = config.index(SPLIT_CONFIG, start_index2 + 1)

    span_setting = config[:start_index1]

    ch_number = config[start_index1 + 1:start_index2]
    mf = config[start_index2 + 1:start_index3]
    power = config[-1]

    return dict(span_setting=span_setting, ch_number=ch_number[0], mf=mf[0], power=power)


def generate_spans_edfas(span_setting):
    spans = []
    edfas = []
    for setting in span_setting:
        if setting == 0:
            spans.append(NonlinearFiber(**PSCF, length=80,
                                        step_length=20 / 1000, backend='cupy'))

        elif setting == 1:
            spans.append(NonlinearFiber(**SSMF, length=80,
                                        step_length=20 / 1000, backend='cupy'))

    for span in spans:
        edfa = Edfa(gain_db=span.alpha * span.length, nf=5, is_ase=False)
        edfas.append(edfa)

    return spans, edfas


def generate_signal(signal_setting: dict):
    nch = signal_setting['nch']
    power = signal_setting['power']
    mf = signal_setting['mf']
    if mf == 0:
        mf = 'qpsk'
    elif mf == 1:
        mf = '16-qam'
    signals = []
    sps_in_fiber = calc_sps_in_fiber(nch, 50, 35)

    frequences = [193.1e12 + i * SPACING for i in range(nch)]
    for index, _ in enumerate(range(nch)):
        sig = QamSignal(sps_in_fiber=sps_in_fiber, mf=mf,
                        frequence=frequences[index])
        sig = AWG(alpha=ROLL_OFF)(sig)
        sig.set_signal_power(power, 'dbm')
        signals.append(sig)

    return signals


def mux_signal(signals):
    center_ch = len(signals) // 2
    signal_index = [center_ch]

    try:
        _ = signals[center_ch + 1]
        signal_index.append(center_ch + 1)
        _ = signals[center_ch + 2]

        signal_index.append(center_ch + 2)
        _ = signals[center_ch + 3]

        signal_index.append(center_ch + 3)
    except Exception as e:
        pass

    try:
        _ = signals[center_ch - 1]
        assert (center_ch - 1) >= 0
        signal_index.insert(0, center_ch - 1)
        _ = signals[center_ch - 2]
        assert (center_ch - 2) >= 0
        signal_index.insert(0, center_ch - 2)
        _ = signals[center_ch - 3]
        assert (center_ch - 3) >= 0
        signal_index.insert(0, center_ch - 3)
    except Exception as e:
        pass

    wdm_signal = Multiplex.mux_signal(signals, grid_size=SPACING)
    wdm_signal_array = to_wdm_array(wdm_signal, signal_index)
    return wdm_signal_array


def prop(signal, spans, edfas, is_copy=False):
    import copy

    if is_copy:
        signal = copy.deepcopy(signal)

    assert len(spans) == len(edfas)
    # cnt = 0
    for cnt in tqdm.tqdm(range(len(spans))):
        signal = spans[cnt](signal)
        signal = edfas[cnt](signal)

    return signal


def calc_xcorr(sequence1, sequence2, max_lag):
    sequence1 = np.atleast_2d(sequence1)
    sequence2 = np.atleast_2d(sequence2)
    res2 = []
    scale = np.arange(len(sequence1[0, :]), len(sequence2[0, :])-max_lag, -1)
    for i in range(sequence1.shape[0]):
        res = ucorrelate(
            sequence1[i, :], sequence2[i, :], max_lag)/scale  # xx,yy
        res2.append(10*np.log10(1/np.sum(np.abs(res))))
        if i == 0:
            res = ucorrelate(
                sequence1[i, :], sequence2[i+1, :], max_lag)/scale  # xy
            res2.append(10*np.log10(1/np.sum(np.abs(res))))

    return res2


if __name__ == '__main__':
    signals = generate_signal(dict(nch=5, power=0, mf=0))
    wdm_array = mux_signal(signals)

    print('xixi')
