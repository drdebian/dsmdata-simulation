# -*- coding: utf-8 -*-
"""Stochastic Charging Model"""

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

    # identifiers
    my_vehicles = demand_ev.index.get_level_values("vehicle").unique()
    my_deadlines = demand_ev[demand_ev.loadend == True].index.values
    my_vehicle_scenarios = demand_ev.index.get_level_values(
        "scenario").unique()

    # Model creation
    model = pulp.LpProblem(
        "StochasticChargingModel", pulp.LpMinimize)

    ###################################################################
    # Entscheidungsvariablen
    ###################################################################

    # NetworkGrid
    n_out = pulp.LpVariable.dicts(
        "GridDraw", [(t, s) for t in Periods for s in my_vehicle_scenarios], lowBound=0, upBound=P_NE_MAX, cat="Continuous"
    )
    n_out_act = pulp.LpVariable.dicts("GridDrawActive", [(
        t, s) for t in Periods for s in my_vehicle_scenarios], cat="Binary")
    n_out_ceil = pulp.LpVariable(
        "GridDrawCeiling", lowBound=0, upBound=P_NE_MAX, cat="Continuous"
    )
    n_out_exp = pulp.LpVariable.dicts(
        "GridDrawExpectedValue", Periods, lowBound=0, upBound=P_NE_MAX, cat="Continuous"
    )

    n_in = pulp.LpVariable.dicts(
        "GridFeed", [(t, s) for t in Periods for s in my_vehicle_scenarios], lowBound=0, upBound=P_NE_MAX, cat="Continuous"
    )
    n_in_act = pulp.LpVariable.dicts("GridFeedActive", [(
        t, s) for t in Periods for s in my_vehicle_scenarios], cat="Binary")

    # Battery
    b_in = pulp.LpVariable.dicts(
        "BatteryCharge", [(t, s) for t in Periods for s in my_vehicle_scenarios], lowBound=0, upBound=P_EL_MAX, cat="Continuous"
    )
    b_out = pulp.LpVariable.dicts(
        "BatteryDischarge", [(t, s) for t in Periods for s in my_vehicle_scenarios], lowBound=0, upBound=P_EL_MAX, cat="Continuous"
    )
    b_in_act = pulp.LpVariable.dicts(
        "BatteryChargeActive", [(t, s) for t in Periods for s in my_vehicle_scenarios], cat="Binary")
    b_out_act = pulp.LpVariable.dicts(
        "BatteryDischargeActive", [(t, s) for t in Periods for s in my_vehicle_scenarios], cat="Binary")

    B = pulp.LpVariable.dicts(
        "BatterySOC", [(t, s) for t in Instants for s in my_vehicle_scenarios], lowBound=E_EL_MIN, upBound=E_EL_MAX, cat="Continuous"
    )

    # Electric Vehicles
    ev_in = pulp.LpVariable.dicts(
        "EVCharge",
        [(v, t, s)
         for v in my_vehicles for t in Periods for s in my_vehicle_scenarios],
        lowBound=0,
        upBound=P_EV_MAX,
        cat="Continuous",
    )
    ev_in_act = pulp.LpVariable.dicts(
        "EVChargeActive", [(v, t, s) for v in my_vehicles for t in Periods for s in my_vehicle_scenarios], cat="Binary"
    )
    ev_in_tot = pulp.LpVariable.dicts(
        "EVChargeTotal", [(t, s) for t in Periods for s in my_vehicle_scenarios], lowBound=0, cat="Continuous"
    )

    EV = pulp.LpVariable.dicts(
        "EVSOC",
        [(v, t, s)
         for v in my_vehicles for t in Instants for s in my_vehicle_scenarios],
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

    model += (1/P_NE_MAX)*n_out_ceil - \
        .5 * (1/len(my_vehicles)) * (1/len(my_vehicle_scenarios)) * (1/len(my_deadlines)) * (1/E_EV_MAX)*pulp.lpSum([EV[(v, t, s)] for v, t, s in my_deadlines]) - \
        .5 * (1/len(my_vehicles)) * (1/len(my_vehicle_scenarios)) * (1/E_EV_MAX) * \
        pulp.lpSum([EV[(v, T, s)]
                   for v in my_vehicles for s in my_vehicle_scenarios])

    ###############
    # Constraints
    ###############

    for t in Periods:  # loop through all periods

        # Expected Value of Grid Draw
        model += n_out_exp[t] == pulp.lpSum([n_out[(t, s)]*(1/len(my_vehicle_scenarios))
                                             for s in my_vehicle_scenarios])
        # max. grid draw => Target formulation of MiniMax problem
        ##model += n_out_exp[t] <= n_out_ceil

        for s in my_vehicle_scenarios:  # loop through all EV scenarios

            # strict condition that all n_out values should be minimized
            model += n_out[(t, s)] <= n_out_ceil

            # special constraints to collapse scenarios in stage 1 for ease of implementation
            if t == 0 and s > 0:
                model += n_out[(t, 0)] == n_out[(t, s)]
                model += n_out_act[(t, 0)] == n_out_act[(t, s)]
                model += n_in[(t, 0)] == n_in[(t, s)]
                model += n_in_act[(t, 0)] == n_in_act[(t, s)]

                model += b_out[(t, 0)] == b_out[(t, s)]
                model += b_out_act[(t, 0)] == b_out_act[(t, s)]
                model += b_in[(t, 0)] == b_in[(t, s)]
                model += b_in_act[(t, 0)] == b_in_act[(t, s)]
                model += B[(t, 0)] == B[(t, s)]

                model += ev_in_tot[(t, 0)] == ev_in_tot[(t, s)]

            # balance equation
            model += (
                n_out[(t, s)] + b_out[(t, s)] + production_pv.PV_kW[t]
                == n_in[(t, s)] + b_in[(t, s)] + ev_in_tot[(t, s)]
            )

            # min./max. grid draw & anti-concurrency
            model += n_out[(t, s)] <= n_out_act[(t, s)] * P_NE_MAX
            model += n_in[(t, s)] <= n_in_act[(t, s)] * P_NE_MAX
            model += n_in_act[(t, s)] + n_out_act[(t, s)] <= 1

            # min./max. (dis)charging power of battery & anti-concurrency
            model += b_in[(t, s)] <= b_in_act[(t, s)] * P_EL_MAX
            model += b_in[(t, s)] >= b_in_act[(t, s)] * P_EL_MIN
            model += b_out[(t, s)] <= b_out_act[(t, s)] * P_EL_MAX
            model += b_out[(t, s)] >= b_out_act[(t, s)] * P_EL_MIN
            model += b_in_act[(t, s)] + b_out_act[(t, s)] <= 1

            # sum up all load coming from electric vehicles
            model += ev_in_tot[(t, s)] == pulp.lpSum([ev_in[(v, t, s)]
                                                      for v in my_vehicles])

            # keep track of battery SOC including losses
            model += (
                B[(t+1, s)]
                == B[(t, s)] * S_EL + (P_EL_ETA * b_in[(t, s)] - (1 / P_EL_ETA) * b_out[(t, s)]) * M_DT
            )

            for v in my_vehicles:

                # special constraints to collapse scenarios in stage 1 for ease of implementation
                if t == 0 and s > 0:
                    model += ev_in[(v, t, 0)] == ev_in[(v, t, s)]
                    model += ev_in_act[(v, t, 0)] == ev_in_act[(v, t, s)]
                    model += EV[(v, t, 0)] == EV[(v, t, s)]

                # keep track of EV SOC
                model += (
                    EV[(v, t+1, s)]
                    == EV[(v, t, s)] * S_EV
                    + (
                        P_EV_ETA * ev_in[(v, t, s)]
                        - demand_ev.power.loc[v, t, s] *
                        demand_ev.driving.loc[v, t, s] / P_EV_ETA
                    )
                    * M_DT
                )
                # min./max. ev charging power
                model += (
                    ev_in[(v, t, s)]
                    >= ev_in_act[(v, t, s)] * demand_ev.loadable.loc[v, t, s] * P_EV_MIN
                )
                model += (
                    ev_in[(v, t, s)]
                    <= ev_in_act[(v, t, s)] * demand_ev.loadable.loc[v, t, s] * P_EV_MAX
                )
                model += ev_in_act[(v, t, s)] >= 0
                model += ev_in_act[(v, t, s)] <= 1

                # trickle charge to 50% as soon as plugged in as default
                model += ev_in_act[(v, t, s)] >= .5 - \
                    EV[(v, t, s)]*(1/E_EV_MAX)

    for s in my_vehicle_scenarios:
        # initial conditions
        model += B[0, s] == B[T, s]  # E_EL_CAP*.1 #E_EL_BEG
        # model += B[0] == E_EL_CAP*.5  # E_EL_BEG
        model += B[0, s] == E_EL_BEG

        for v in my_vehicles:
            # model += EV[(v, 0, s)] == E_EV_BEG
            model += EV[(v, 0, s)] == demand_ev.EVSOC.loc[v, 0, s]
            # model += EV[(v, 0, s)] <= EV[(v, T, s)]
            # model += EV[(v, 0, s)] >= EV[(v, T, s)] - P_EV_MIN * M_DT
            # model += EV[(v, 0, s)] == EV[(v, T, s)] + \
            #     EV_under[(v, T, s)] - EV_over[(v, T, s)]

    # calculate PV KPIs
    model += wPro == pulp.lpSum(production_pv.PV_kW)
    model += wVer == pulp.lpSum([ev_in_tot[(t, s)]*(1/len(my_vehicle_scenarios))
                                 for t in Periods for s in my_vehicle_scenarios])
    model += wBez == pulp.lpSum([n_out[(t, s)]*(1/len(my_vehicle_scenarios))
                                 for t in Periods for s in my_vehicle_scenarios])
    model += wEin == pulp.lpSum([n_in[(t, s)]*(1/len(my_vehicle_scenarios))
                                 for t in Periods for s in my_vehicle_scenarios])

    # set constants for postprocessing
    model += cPeriods == T
    model += cDeltaT == M_DT

    # Prepare model
    return model
