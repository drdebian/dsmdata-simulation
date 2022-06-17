# -*- coding: utf-8 -*-
"""Config data for Master Thesis Simulation Studies"""

___author__ = "2022 Albert Ulmer"

from copy import deepcopy
#import random
#import math
#import numpy

#pi = math.pi
#cos = math.cos

hours_per_day = 24

def timing_1():
    # timing and scope
    #T = hours_per_day
    M_DT = 1 #.25  #1                  # model length of period in hours
    M_LA = 1*hours_per_day          # model lookahead hours
    T = int(round(M_LA/M_DT, 0))
   
    # assemble variables into dictionary
    conf = {
        "T": T, "M_DT": M_DT, "M_LA": M_LA
    }

    return(deepcopy(conf))    
    
def varpvbess_1(pvprc=1, bessprc=1):
    conf1 = timing_1()
    
    # timing and scope
    T = conf1['T'] 
    M_DT = conf1['M_DT']                # model length of period in hours
    M_LA = conf1['M_LA']             # model lookahead number of periods

    # infrastructure
    # power grid
    P_NE_MAX = 35                # maximum power draw/delivery allowed from/to grid in kW
    P_NE_MIN = 0                 # minimum power draw/delivery allowed from/to grid in kW

    # electric vehicles
    # EV capacity in kWh (Nissan Leaf 2016: (40) or 62 kWh)
    E_EV_CAP = 40
    E_EV_MAX = E_EV_CAP*.95
    E_EV_MIN = E_EV_CAP*.05
    E_EV_BEG = E_EV_CAP*.5       # initial state of charge
    P_EV_MAX = 32*3*230/1000     # max. EV charging power in kW (A*phases*V)
    P_EV_MIN = 6*3*230/1000      # min. EV charging power in kW (A*phases*V)
    P_EV_ETA = .85               # EV charging efficiency
    D_EV = .05                   # self-discharge percentage of electric vehicles per day
    S_EV = 1 - D_EV/30*M_DT      # discharge factor per period
    
    # photovoltaics
    P_PV_CAPMAX = E_EV_CAP
    P_PV_CAP = P_PV_CAPMAX*pvprc # installed kWp (max. charging power of one EV)
    P_PV_ETA = .80               # achievable percentage of peak power

    # electric storage
    E_EL_CAPMAX = E_EV_CAP
    E_EL_CAP = E_EL_CAPMAX*bessprc #battery capacity in kWh
    E_EL_MAX = E_EL_CAP*.99        # max. energy level to charge to
    E_EL_MIN = E_EL_CAP*.01        # min. energy level to drain to
    E_EL_BEG = E_EL_CAP*.5         # initial state of charge
    E_EL_END = E_EL_BEG            # final state of charge
    P_EL_MAX = 12.78               # max. battery (dis)charging power in kW
    P_EL_MIN = 1                   # min. battery charging power in kW
    P_EL_ETA = .95                 # (dis)charging efficiency
    D_EL = .04                     # self-discharge percentage of battery per day
    S_EL = 1 - D_EL/30*M_DT        # discharge factor per period



    conf2 = {
        "T": T, "M_DT": M_DT, "M_LA": M_LA, "P_NE_MAX": P_NE_MAX, "P_NE_MIN": P_NE_MIN, "E_EL_CAP": E_EL_CAP, "E_EL_MAX": E_EL_MAX, "E_EL_MIN": E_EL_MIN, "E_EL_BEG": E_EL_BEG, "E_EL_END": E_EL_END, "P_EL_MAX": P_EL_MAX, "P_EL_MIN": P_EL_MIN, "P_EL_ETA": P_EL_ETA, "D_EL": D_EL, "S_EL": S_EL, "E_EV_CAP": E_EV_CAP, "E_EV_MAX": E_EV_MAX, "E_EV_MIN": E_EV_MIN, "E_EV_BEG": E_EV_BEG, "P_EV_MAX": P_EV_MAX, "P_EV_MIN": P_EV_MIN, "P_EV_ETA": P_EV_ETA, "D_EV": D_EV, "S_EV": S_EV, "P_PV_CAP": P_PV_CAP, "P_PV_ETA": P_PV_ETA
    }
   
    return(deepcopy(conf2))


def basis_1():
    T = hours_per_day

    # timing and scope
    M_DT = 1  # .25                # model length of period in hours
    M_LA = int(T/M_DT)             # model lookahead number of periods

    # infrastructure
    # power grid
    P_NE_MAX = 35                # maximum power draw/delivery allowed from/to grid in kW
    P_NE_MIN = 0                 # minimum power draw/delivery allowed from/to grid in kW

    # electric storage
    E_EL_CAP = 26.56             # battery capacity in kWh
    E_EL_MAX = E_EL_CAP*.99      # max. energy level to charge to
    E_EL_MIN = E_EL_CAP*.01      # min. energy level to drain to
    E_EL_BEG = E_EL_CAP*.5       # initial state of charge
    E_EL_END = E_EL_BEG          # final state of charge
    P_EL_MAX = 12.78             # max. battery (dis)charging power in kW
    P_EL_MIN = 1                 # min. battery charging power in kW
    P_EL_ETA = .95               # (dis)charging efficiency
    D_EL = .04               # self-discharge percentage of battery per day
    S_EL = 1 - D_EL/30*M_DT  # discharge factor per period

    # electric vehicles
    # EV capacity in kWh (Nissan Leaf 2016: (40) or 62 kWh)
    E_EV_CAP = 40
    E_EV_MAX = E_EV_CAP*.95
    E_EV_MIN = E_EV_CAP*.05
    E_EV_BEG = E_EV_CAP*.5      # initial state of charge
    P_EV_MAX = 32*3*230/1000     # max. EV charging power in kW (A*phases*V)
    P_EV_MIN = 6*3*230/1000     # min. EV charging power in kW (A*phases*V)
    P_EV_ETA = .85               # EV charging efficiency
    D_EV = .05               # self-discharge percentage of electric vehicles per day
    S_EV = 1 - D_EV/30*M_DT  # discharge factor per period

    # photovoltaics
    P_PV_CAP = 14.28             # installed kWp
    P_PV_ETA = .6              # achievable percentage of peak power

    conf = {
        "T": T, "M_DT": M_DT, "M_LA": M_LA, "P_NE_MAX": P_NE_MAX, "P_NE_MIN": P_NE_MIN, "E_EL_CAP": E_EL_CAP, "E_EL_MAX": E_EL_MAX, "E_EL_MIN": E_EL_MIN, "E_EL_BEG": E_EL_BEG, "E_EL_END": E_EL_END, "P_EL_MAX": P_EL_MAX, "P_EL_MIN": P_EL_MIN, "P_EL_ETA": P_EL_ETA, "D_EL": D_EL, "S_EL": S_EL, "E_EV_CAP": E_EV_CAP, "E_EV_MAX": E_EV_MAX, "E_EV_MIN": E_EV_MIN, "E_EV_BEG": E_EV_BEG, "P_EV_MAX": P_EV_MAX, "P_EV_MIN": P_EV_MIN, "P_EV_ETA": P_EV_ETA, "D_EV": D_EV, "S_EV": S_EV, "P_PV_CAP": P_PV_CAP, "P_PV_ETA": P_PV_ETA
    }
    return(deepcopy(conf))
