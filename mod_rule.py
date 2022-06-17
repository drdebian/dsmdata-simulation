# -*- coding: utf-8 -*-
"""Rule-based Charging Model"""

___author__ = "2022 Ulmer Albert"

import pulp
import numpy as np


def constructModel(config, production_pv, demand_ev):

    # Time
    T = config["T"]
    assert T >= 1
    Periods = range(0, T)
    Instants = range(0, T + 1)

    # Parameter
    bigm = 1_000

    # timing and scope
    M_DT = config["M_DT"]  # model length of period in hours
    M_LA = config["M_LA"]  # model lookahead number of periods

    # infrastructure
    # power grid
    # maximum power draw/delivery allowed from/to grid in kW
    P_NE_MAX = config["P_NE_MAX"]
    # minimum power draw/delivery allowed from/to grid in kW
    P_NE_MIN = config["P_NE_MIN"]

    # electric storage
    E_EL_CAP = config["E_EL_CAP"]  # battery capacity in kWh
    E_EL_MAX = config["E_EL_MAX"]  # max. energy level to charge to
    E_EL_MIN = config["E_EL_MIN"]  # min. energy level to drain to
    E_EL_BEG = config["E_EL_BEG"]  # initial state of charge
    E_EL_END = config["E_EL_END"]  # final state of charge
    # max. battery (dis)charging power in kW
    P_EL_MAX = config["P_EL_MAX"]
    # min. battery charging power in kW
    P_EL_MIN = config["P_EL_MIN"]
    P_EL_ETA = config["P_EL_ETA"]  # (dis)charging efficiency
    # self-discharge percentage of battery per day
    D_EL = config["D_EL"]
    S_EL = config["S_EL"]  # discharge factor per period

    # electric vehicles
    # EV capacity in kWh (Nissan Leaf 2016: (40) or 62 kWh)
    E_EV_CAP = config["E_EV_CAP"]
    E_EV_MAX = config["E_EV_MAX"]
    E_EV_MIN = config["E_EV_MIN"]
    E_EV_BEG = config["E_EV_BEG"]  # initial state of charge
    # max. EV charging power in kW (A*phases*V)
    P_EV_MAX = config["P_EV_MAX"]
    # min. EV charging power in kW (A*phases*V)
    P_EV_MIN = config["P_EV_MIN"]
    P_EV_ETA = config["P_EV_ETA"]  # EV charging efficiency
    # self-discharge percentage of electric vehicles per day
    D_EV = config["D_EV"]
    S_EV = config["S_EV"]  # discharge factor per period

    # photovoltaics
    P_PV_CAP = config["P_PV_CAP"]  # installed kWp
    # achievable percentage of peak power
    P_PV_ETA = config["P_PV_ETA"]

    # time series
    my_vehicles = demand_ev.index.get_level_values("vehicle").unique()

    # Model creation
    model = pulp.LpProblem("RulebasedChargingModel", pulp.LpMinimize)

    ###################################################################
    # Entscheidungsvariablen
    ###################################################################

    # NetworkGrid
    n_out = pulp.LpVariable.dicts(
        "GridDraw", Periods, lowBound=0, upBound=P_NE_MAX, cat="Continuous"
    )
    n_out_act = pulp.LpVariable.dicts("GridDrawActive", Periods, cat="Binary")
    n_out_ceil = pulp.LpVariable(
        "GridDrawCeiling", lowBound=0, upBound=P_NE_MAX, cat="Continuous"
    )

    n_in = pulp.LpVariable.dicts(
        "GridFeed", Periods, lowBound=0, upBound=P_NE_MAX, cat="Continuous"
    )
    n_in_act = pulp.LpVariable.dicts("GridFeedActive", Periods, cat="Binary")

    # Battery
    b_in = pulp.LpVariable.dicts(
        "BatteryCharge", Periods, lowBound=0, upBound=P_EL_MAX, cat="Continuous"
    )
    b_out = pulp.LpVariable.dicts(
        "BatteryDischarge", Periods, lowBound=0, upBound=P_EL_MAX, cat="Continuous"
    )
    b_in_act = pulp.LpVariable.dicts(
        "BatteryChargeActive", Periods, cat="Binary")
    b_out_act = pulp.LpVariable.dicts(
        "BatteryDischargeActive", Periods, cat="Binary")

    B = pulp.LpVariable.dicts(
        "BatterySOC", Instants, lowBound=E_EL_MIN, upBound=E_EL_MAX, cat="Continuous"
    )

    # Electric Vehicles
    ev_in = pulp.LpVariable.dicts(
        "EVCharge",
        [(v, t) for v in my_vehicles for t in Periods],
        lowBound=0,
        upBound=P_EV_MAX,
        cat="Continuous",
    )
    ev_in_act = pulp.LpVariable.dicts(
        "EVChargeActive", [(v, t) for v in my_vehicles for t in Periods], cat="Binary"
    )
    ev_in_tot = pulp.LpVariable.dicts(
        "EVChargeTotal", Periods, lowBound=0, cat="Continuous"
    )

    # binary variables to couple ev_in to EV based on rules
    I1 = pulp.LpVariable.dicts(
        "Interval1", [(v, t) for v in my_vehicles for t in Periods], cat="Binary"
    )
    I2 = pulp.LpVariable.dicts(
        "Interval2", [(v, t) for v in my_vehicles for t in Periods], cat="Binary"
    )
    I3 = pulp.LpVariable.dicts(
        "Interval3", [(v, t) for v in my_vehicles for t in Periods], cat="Binary"
    )
    I4 = pulp.LpVariable.dicts(
        "Interval4", [(v, t) for v in my_vehicles for t in Periods], cat="Binary"
    )
    I5 = pulp.LpVariable.dicts(
        "Interval5", [(v, t) for v in my_vehicles for t in Periods], cat="Binary"
    )

    EV = pulp.LpVariable.dicts(
        "EVSOC",
        [(v, t) for v in my_vehicles for t in Instants],
        lowBound=E_EV_MIN,
        upBound=E_EV_MAX,
        cat="Continuous",
    )

    # helper variables for PV KPI calculation
    wPro = pulp.LpVariable("PVProducedTotal", lowBound=0, cat="Continuous")
    wEin = pulp.LpVariable("GridFeedTotal", lowBound=0, cat="Continuous")
    wVer = pulp.LpVariable("PowerDemandTotal", lowBound=0, cat="Continuous")
    wBez = pulp.LpVariable("GridDrawTotal", lowBound=0, cat="Continuous")

    # helper variables for postprocessing
    cPeriods = pulp.LpVariable("Periods", lowBound=0, cat="Integer")
    cDeltaT = pulp.LpVariable("DeltaT", lowBound=0, cat="Continuous")

    ###############
    # Objective
    ###############

    model += pulp.lpSum([n_out[t]*t for t in Periods])

    ###############
    # Constraints
    ###############

    for t in Periods:

        model += (
            n_out[t] + b_out[t] + production_pv.PV_kW[t]
            == n_in[t] + b_in[t] + ev_in_tot[t]
        )  # balance equation

        model += n_out[t] <= n_out_ceil  # max. grid draw => Target formulation

        # min./max. grid draw & anti-concurrency
        model += n_out[t] <= n_out_act[t] * P_NE_MAX
        model += n_in[t] <= n_in_act[t] * P_NE_MAX
        model += n_in_act[t] + n_out_act[t] <= 1

        # min./max. (dis)charging power of battery & anti-concurrency
        model += b_in[t] <= b_in_act[t] * P_EL_MAX
        model += b_in[t] >= b_in_act[t] * P_EL_MIN
        model += b_out[t] <= b_out_act[t] * P_EL_MAX
        model += b_out[t] >= b_out_act[t] * P_EL_MIN
        model += b_in_act[t] + b_out_act[t] <= 1

        # sum up all load coming from electric vehicles
        model += ev_in_tot[t] == pulp.lpSum([ev_in[(v, t)]
                                             for v in my_vehicles])

        # keep track of battery SOC including losses
        model += (
            B[t + 1]
            == B[t] * S_EL + (P_EL_ETA * b_in[t] - (1 / P_EL_ETA) * b_out[t]) * M_DT
        )

        for v in my_vehicles:
            model += (
                EV[(v, t + 1)]
                == EV[(v, t)] * S_EV
                + (
                    P_EV_ETA * ev_in[(v, t)]
                    - demand_ev.power.loc[v, t] *
                    demand_ev.driving.loc[v, t] / P_EV_ETA
                )
                * M_DT
            )  # keep track of EV SOC
            # min./max. ev charging power
            model += (
                ev_in[(v, t)]
                >= ev_in_act[(v, t)] * demand_ev.loadable.loc[v, t] * P_EV_MIN
            )
            model += (
                ev_in[(v, t)]
                <= ev_in_act[(v, t)] * demand_ev.loadable.loc[v, t] * P_EV_MAX
            )
            model += ev_in_act[(v, t)] >= 0
            model += ev_in_act[(v, t)] <= 1

            # trickle charge to xx% as soon as plugged in as default
            model += ev_in_act[(v, t)] >= .90 - EV[(v, t)]*(1/E_EV_MAX)
            #model += ev_in_act[(v, t)] >= E_EV_MAX - EV[(v, t)]

            # SOS1 constraint on I1-I5 intervals
            model += I1[(v, t)] + I2[(v, t)] + I3[(v, t)] + \
                I4[(v, t)] + I5[(v, t)] == 1

            # upper bounds
            model += EV[(v, t)] <= E_EV_MAX * \
                (1 - 4/5*I1[(v, t)] - 3/5*I2[(v, t)] -
                 2/5*I3[(v, t)] - 1/5*I4[(v, t)])

            # lower bounds
            model += EV[(v, t)] >= E_EV_MAX * \
                (0 + 1/5*I2[(v, t)] + 2/5*I3[(v, t)] +
                 3/5*I4[(v, t)] + 4/5*I5[(v, t)])

            # limit charging power based on EVSOC-interval
            model += ev_in[(v, t)] <= P_EV_MAX - (P_EV_MAX-P_EV_MIN) * \
                (1/4*I2[(v, t)]+2/4*I3[(v, t)]+3/4*I4[(v, t)]+4/4*I5[(v, t)])

    # initial conditions
    model += B[0] == B[T]  # E_EL_CAP*.1 #E_EL_BEG
    # model += B[0] == E_EL_CAP*.5  # E_EL_BEG
    model += B[0] == E_EL_BEG

    for v in my_vehicles:
        # model += EV[(v, 0)] == E_EV_BEG
        model += EV[(v, 0)] == demand_ev.EVSOC.loc[v, 0]
        # model += EV[(v, 0)] <= EV[(v, T)]
        # model += EV[(v, 0)] >= EV[(v, T)] - P_EV_MIN * M_DT
        # model += EV[(v, 0)] == EV[(v, T)] + EV_under[(v, T)] - EV_over[(v, T)]

    # calculate PV KPIs
    model += wPro == pulp.lpSum(production_pv.PV_kW)
    model += wVer == pulp.lpSum(ev_in_tot)
    model += wBez == pulp.lpSum(n_out)
    model += wEin == pulp.lpSum(n_in)

    # set constants for postprocessing
    model += cPeriods == T
    model += cDeltaT == M_DT

    # Prepare model
    return model
