import copy
import random

import numpy as np
import Ocspy.Base as base
import Ocspy.Channel as ch
import Ocspy.Instrument as ins
import joblib
from typing import List
from Ocspy.Base import QamSignal
from Ocspy.Instrument import AWG
from resampy import resample

from utilities import prop, read_config, generate_signal, mux_signal, generate_spans_edfas

SPLIT_CONFIG = -100000

def generate_data_config():

    signal_power = [0, 1, 2, 3]

    span_number = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ch_number = [5, 7, 9, 11, 13, 15, 17, 19]
    # 0: represent:'SSMF' 1:represent:ELEAF 2:represent:Hybrid
    span_type = [0, 1]
    mf = [0, 1]

    config_set = set()
    while True:
        config = []
        chosen_span_number = random.choice(span_number)
        for _ in range(chosen_span_number):
            config.append(random.choice(span_type))
        config.append(SPLIT_CONFIG)
        chosen_ch_number = random.choice(ch_number)
        config.append(chosen_ch_number)
        config.append(SPLIT_CONFIG)
        chosen_mf = random.choice(mf)
        config.append(chosen_mf)
        config.append(SPLIT_CONFIG)
        config_set.add(tuple(config))

        if len(config_set) > 2200:
            break
        else:
            print(len(config_set))

    config_set = list(config_set)
    config_set_power = []

    for config in config_set:
        config = list(config)
        chosen_power = random.choice(signal_power)
        config.append(chosen_power)
        config_set_power.append(config)
    return config_set_power



def simulate_spm(center_signal,spans,edfas):
    after_prop = prop(center_signal,spans,edfas,True)
    return after_prop

def sumulate_all(wdm_signal,spans,edfas):
    wdm_signal = prop(wdm_signal,spans,edfas,False)
    return wdm_signal

def save(obj,name):
    joblib.dump(obj,'./data/'+name)



def main():
    try:
        import os
        os.mkdir('data')
    except Exception as e:
        pass
    cofig_all = joblib.load('config_set')
    cofig_all = cofig_all[:200]


    for conf_ith,conf in enumerate(cofig_all):
        res = read_config(conf)
        span_setting = res['span_setting']
        ch_number = res['ch_number']
        mf = res['mf']
        power = res['power']

        signals = generate_signal(dict(nch=ch_number,mf=mf,power=power))
        spans,edfas = generate_spans_edfas(span_setting)
        wdm_signal = mux_signal(signals)
        assert len(spans) == len(span_setting)
        center_index = len(signals)//2
        center_signal = signals[center_index]

        center_signal_to_prop_obj = copy.deepcopy(center_signal)
        center_signal_to_prop = resample(center_signal[:],center_signal.sps_in_fiber,4)
        center_signal_to_prop_obj.data_sample_in_fiber = center_signal_to_prop
        center_signal_to_prop_obj.sps_in_fiber = 4
        center_signal_to_prop_obj.set_signal_power(power,'dbm')
        center_signal_afterprop = simulate_spm(center_signal_to_prop_obj,spans,edfas)
        wdm_signal_afterprop  = sumulate_all(wdm_signal,spans,edfas)
        to_save = dict(wdm_signal_afterprop=wdm_signal_afterprop,center_signal_afterprop=center_signal_afterprop,spans=spans)
        save(to_save,f'{conf_ith}_th')



if __name__ == '__main__':
    main()




# config = [1, 0, 1, 0, 0, 1, 0, 1, -100000, 9, -100000, 0, -100000, 3]
# read_config(config)
#
    # config_set_power = generate_data_config()
    # joblib.dump(config_set_power, './config_set')
# print('xixi')
# print(joblib.load('./config_set'))
