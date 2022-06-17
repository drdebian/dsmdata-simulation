# -*- coding: utf-8 -*-
"""Plotting functions"""

import util
___author__ = "2022 Albert Ulmer"

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
# plt.style.use('ggplot')
# plt.style.use("default")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import scipy.linalg
from typing import Any, Dict
import pandas as pd


def bar3d_plot(mydata: pd.DataFrame, xlabel="", ylabel="", zlabel="", title="", color="red"):

    if color == "red":
        cmap = cm.Reds
    elif color == "rainbow":
        cmap = cm.rainbow
    else:
        cmap = cm.Greens

    data = mydata.to_numpy()
    column_names = list(mydata.columns)
    row_names = list(mydata.index.values)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    lx = len(data[0])
    ly = len(data[:, 0])
    width = .22
    xpos = np.arange(0, lx, 1)
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos-width*.35, ypos-width)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx*ly)

    dx = width * np.ones_like(zpos)
    dy = dx.copy()
    dz = data.flatten()

    values = np.linspace(0.2, 1., xpos.ravel().shape[0])
    #colors = cm.rainbow(values)
    colors = cmap(values)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=.5)

    ticksx = np.arange(0., lx, 1)
    plt.xticks(ticksx, column_names, fontsize=8)

    ticksy = np.arange(0., ly, 1)
    plt.yticks(ticksy, row_names, fontsize=8)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_zlabel(zlabel, fontsize=9)
    ax.set_title(title)

    # fig.tight_layout()

    return(fig)


def contour_plot(mydata: np.array, order=0, xlabel="", ylabel="", zlabel="", title="", color="red"):

    if color == "red":
        cmap = cm.Reds
    else:
        cmap = cm.Greens

    xvals = np.unique(mydata[:, 0])
    yvals = np.unique(mydata[:, 1])

    # regular grid covering the domain of the data
    X, Y = np.meshgrid(xvals, yvals)
    XX = X.flatten()
    YY = Y.flatten()

    if order == 1:
        # best-fit linear plane
        A = np.c_[mydata[:, 0], mydata[:, 1], np.ones(mydata.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, mydata[:, 2])    # coefficients

        B = mydata[:, 2]
        SStot = ((B - B.mean())**2).sum()
        SSres = ((B - np.dot(A, C))**2).sum()
        try:
            R2 = 1 - SSres / SStot
        except:
            R2 = 1

        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(mydata.shape[0]), mydata[:, :2], np.prod(
            mydata[:, :2], axis=1), mydata[:, :2]**2]
        C, _, _, _ = scipy.linalg.lstsq(A, mydata[:, 2])

        B = mydata[:, 2]
        SStot = ((B - B.mean())**2).sum()
        SSres = ((B - np.dot(A, C))**2).sum()
        try:
            R2 = 1 - SSres / SStot
        except:
            R2 = 1

        if R2 == -np.Inf:
            R2 = 1

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX *
                   YY, XX**2, YY**2], C).reshape(X.shape)

    elif order == 0:
        Z = mydata[:, 2].reshape(X.shape)
        X, Y = Y, X

    fig, ax0 = plt.subplots(1, 1, figsize=(7, 6))

    # 2d plot
    plt.style.use("default")
    # plt.style.use("ggplot")
    cpf = ax0.contourf(X, Y, Z, 7, cmap=cmap)
    line_colors = ['black' for l in cpf.levels]
    cp = ax0.contour(X, Y, Z, 7, colors=line_colors)
    ax0.clabel(cp, fontsize=10, colors=line_colors, fmt='%.2f')
    plt.colorbar(cpf, ax=ax0, label=zlabel)
    ax0.set_xticks(xvals)
    ax0.set_yticks(yvals)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel(ylabel)
    ax0.set_title(title)

    if order > 0:
        at = AnchoredText("$R^2$ = " + str(round(R2, 2)),
                          frameon=True, loc='upper center')
        at.patch.set_boxstyle("round, pad=0, rounding_size=.2")
        ax0.add_artist(at)

    fig.tight_layout()

    return(fig)


def surface_plot(mydata: np.array, order=0, xlabel="", ylabel="", zlabel="", title="", color="red"):

    if color == "red":
        cmap = cm.Reds
    else:
        cmap = cm.Greens

    xvals = np.unique(mydata[:, 0])
    yvals = np.unique(mydata[:, 1])

    # regular grid covering the domain of the data
    X, Y = np.meshgrid(xvals, yvals)
    XX = X.flatten()
    YY = Y.flatten()

    if order == 1:
        # best-fit linear plane
        A = np.c_[mydata[:, 0], mydata[:, 1], np.ones(mydata.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, mydata[:, 2])    # coefficients

        B = mydata[:, 2]
        SStot = ((B - B.mean())**2).sum()
        SSres = ((B - np.dot(A, C))**2).sum()
        try:
            R2 = 1 - SSres / SStot
        except:
            R2 = 1

        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]

    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(mydata.shape[0]), mydata[:, :2], np.prod(
            mydata[:, :2], axis=1), mydata[:, :2]**2]
        C, _, _, _ = scipy.linalg.lstsq(A, mydata[:, 2])

        B = mydata[:, 2]
        SStot = ((B - B.mean())**2).sum()
        SSres = ((B - np.dot(A, C))**2).sum()
        try:
            R2 = 1 - SSres / SStot
        except:
            R2 = 1

        if R2 == -np.Inf:
            R2 = 1

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX *
                   YY, XX**2, YY**2], C).reshape(X.shape)

    elif order == 0:
        Z = mydata[:, 2].reshape(X.shape)
        X, Y = Y, X

    # 3d plot
    plt.style.use("default")
    # plt.style.use("ggplot")
    fig = plt.figure(figsize=(7, 7))
    ax1 = plt.axes(projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cmap, alpha=.75)
    ax1.scatter(mydata[:, 0], mydata[:, 1], mydata[:, 2], c='blue', s=50)
    ax1.set_xticks(xvals)
    ax1.set_yticks(yvals)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_zlabel(zlabel)
    plt.rcParams['axes.titley'] = 1.0
    ax1.set_title(title)

    if order > 0:
        at = AnchoredText("$R^2$ = " + str(round(R2, 2)),
                          frameon=True, loc='center')
        at.patch.set_boxstyle("round, pad=0, rounding_size=.2")
        ax1.add_artist(at)

    fig.tight_layout()

    return(fig)


def plot_ev_loadable(ev_data):
    # plot loadable
    i = 0
    for v in ev_data.index.get_level_values('vehicle').unique():
        i += 1
        plt.step(ev_data.loc[v, :].index.get_level_values('period'),
                 np.array([ev_data.loadable[v, t]
                           for t in ev_data.loc[v, :].index.get_level_values('period')])*i,
                 '--',
                 label='$ev^{loadable}_t$'+v)
    ticklabels = ['n. l.']
    ticklabels.extend(ev_data.index.get_level_values('vehicle').unique())
    ticklabels
    ticks = np.arange(0, len(ticklabels))
    ticks
    plt.yticks(ticks=ticks, labels=ticklabels)
    plt.xlim(0, len(ev_data.index.get_level_values('period').unique()))
    plt.xlabel('periods')


def plot_sys_timeseries_simple(result0, result1, result2, prodpv, demandev):

    my_periods = range(0, int(result0.Periods[0]))  # c1['T'])
    my_instants = range(0, int(result0.Periods[0])+1)
    my_vehicles = result2.index.get_level_values('vehicle').unique()
    my_maxgriddraw = max(np.array([result1.GridDraw.loc[t]
                                   for t in my_periods]))
    # my_power_ub = 45  # max([c1['P_EL_MAX'], c1['P_EV_MAX']])
    # my_soc_ub = 45  # max([c1['E_EL_CAP'], c1['E_EV_CAP']])

    fig, axs = plt.subplots(3, 1, figsize=(13, 13), sharex="col")

    # plot Overview
    axs[0].step(np.array(my_periods), np.array([result1.GridDraw.loc[t]
                                                for t in my_periods]), '.-r', where='post', label='$GridDraw$', lw=3)
    axs[0].axhline(round(my_maxgriddraw, 2),
                   ls=':', label='$GridDraw Ceiling$')
    axs[0].step(np.array(my_periods), np.array([result1.GridFeed.loc[t]
                                                for t in my_periods]), '.-m', where='post', label='$GridFeed$', lw=3)
    axs[0].step(np.array(my_periods), np.array([result1.BatteryCharge.loc[t]
                                                for t in my_periods]),     '.-g', where='post', label='$BatteryCharge$', lw=3)
    axs[0].step(np.array(my_periods), np.array([result1.BatteryDischarge.loc[t]
                                                for t in my_periods]),     '.-b', where='post', label='$BatteryDischarge$', lw=3)
    axs[0].step(np.array(my_periods), np.array([result1.EVChargeTotal.loc[t] for t in my_periods]),
                '.-k', where='post', label='$EVDraw$', lw=3)
    axs[0].step(np.array(my_periods),
                prodpv.PV_kW,
                ls='-',
                color='yellow',
                where='post',
                label='$PVProduction$',
                lw=3
                )

    at = AnchoredText(
        result0.label[0] + ": $n_{max}$ = " + str(round(my_maxgriddraw, 2)),
        # prop=dict(size=15),
        frameon=True,
        loc='upper center')
    at.patch.set_boxstyle("round, pad=0, rounding_size=.2")
    axs[0].add_artist(at)

    axs[0].set_xlabel('periods')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title('Overview')
    axs[0].legend(loc='upper right')  # , fontsize=7)

    # plt.subplot(4,1,2)
    # for i, v in enumerate(my_vehicles):
    #     color = 'C' + str(i)
    #     plt.step(my_periods, np.array([demand_ev.loadable.loc[v, t] for t in my_periods])*(i+1), '--', where='post', color = color, label='$ev^{loadable}_t$'+v)
    # ticklabels = ['n. l.']
    # ticklabels.extend(my_vehicles)
    # ticklabels
    # ticks = np.arange(0, len(ticklabels))
    # ticks
    # plt.yticks(ticks=ticks, labels=ticklabels)
    # plt.xlim(0, len(my_periods))
    # plt.xlabel('periods')
    # plt.title('Loadability per vehicle')

    # plot individual EV charge/discharge
    for i, v in enumerate(my_vehicles):
        color = 'C' + str(i)
        axs[1].step(my_periods, np.array([result2.EVCharge[v, t] for t in my_periods])
                    - np.array([demandev.power.loc[v, t] for t in my_periods]), '.-', where='post', color=color, label='$'+v+'$')
    axs[1].set_xlabel('periods')
    axs[1].set_ylabel('Power (kW)')
    axs[1].set_title('Power per vehicle')
    axs[1].legend(loc='upper right')

    # plot SOC
    axs[2].plot(my_instants, result1.BatterySOC,
                color='black',
                ls='-',
                label='$Battery$',
                lw=3,
                )
    for i, v in enumerate(my_vehicles):
        color = 'C' + str(i)
        axs[2].plot(my_instants, np.array([result2.EVSOC[v, t] for t in my_instants]),
                    ls='-',
                    color=color,
                    label='$'+v+'$',
                    lw=3
                    )

    axs[2].set_xlabel('instants')
    axs[2].set_ylabel('Energy (kWh)')
    axs[2].set_title('SOC')
    axs[2].legend(loc='upper right')  # , fontsize=7)

    fig.tight_layout()

    return(fig)


def plot_sys_timeseries_stochastic(result0, result1, result2, result3, prodpv, demandev):
    my_periods = range(0, int(result0.Periods[0]))  # c1['T'])
    my_instants = range(0, int(result0.Periods[0])+1)
    my_vehicles = result3.index.get_level_values('vehicle').unique()
    my_scenarios = result2.index.get_level_values('scenario').unique()

    fig, axs = plt.subplots(3, 1, figsize=(13, 13), sharex="col")

    # plot Overview
    # GridDraw average
    avg = np.array([result2.GridDraw.loc[(t)].mean() for t in my_periods])
    lb = np.maximum(
        avg - np.array([result2.GridDraw.loc[(t)].std() for t in my_periods]), 0)
    ub = avg + np.array([result2.GridDraw.loc[(t)].std()
                         for t in my_periods])
    color = 'red'
    axs[0].step(np.array(my_periods), avg,
                ls='--',
                lw=3,
                # color='red',
                color=color,
                where='post',
                label='$\mathbb{E}\left[ GridDraw \\right]$'
                )
    axs[0].step(np.array(my_periods), lb,
                ls=':',
                lw=1,
                where='post',
                color=color  # ,
                )
    axs[0].step(np.array(my_periods), ub,
                ls=':',
                lw=1,
                where='post',
                color=color
                )
    axs[0].fill_between(my_periods, avg, lb, hatch='//',
                        alpha=.1, color=color, step='post')
    axs[0].fill_between(my_periods, avg, ub, hatch='\\\\',
                        alpha=.1, color=color, step='post')

    # GridFeed average
    avg = np.array([result2.GridFeed.loc[(t)].mean() for t in my_periods])
    lb = np.maximum(
        avg - np.array([result2.GridFeed.loc[(t)].std() for t in my_periods]), 0)
    ub = avg + np.array([result2.GridFeed.loc[(t)].std()
                         for t in my_periods])
    color = 'magenta'
    axs[0].step(np.array(my_periods), avg,
                ls='--',
                lw=3,
                # color='red',
                color=color,
                where='post',
                label='$\mathbb{E}\left[ GridFeed \\right]$'
                )
    axs[0].step(np.array(my_periods), lb,
                ls=':',
                lw=1,
                where='post',
                color=color  # ,
                )
    axs[0].step(np.array(my_periods), ub,
                ls=':',
                lw=1,
                where='post',
                color=color
                )
    axs[0].fill_between(my_periods, avg, lb, hatch='//',
                        alpha=.1, color=color, step='post')
    axs[0].fill_between(my_periods, avg, ub, hatch='\\\\',
                        alpha=.1, color=color, step='post')

    # EVChargeTotal average
    avg = np.array([result2.EVChargeTotal.loc[(t)].mean()
                    for t in my_periods])
    lb = np.maximum(
        avg - np.array([result2.EVChargeTotal.loc[(t)].std() for t in my_periods]), 0)
    ub = avg + np.array([result2.EVChargeTotal.loc[(t)].std()
                         for t in my_periods])
    color = 'black'
    axs[0].step(np.array(my_periods), avg,
                ls='--',
                lw=3,
                color=color,
                where='post',
                label='$\mathbb{E}\left[ EVChargeTotal \\right]$'
                )
    axs[0].step(np.array(my_periods), lb,
                ls=':',
                lw=1,
                where='post',
                color=color  # ,
                )
    axs[0].step(np.array(my_periods), ub,
                ls=':',
                lw=1,
                where='post',
                color=color
                )
    axs[0].fill_between(my_periods, avg, lb, hatch='//',
                        alpha=.1, color=color, step='post')
    axs[0].fill_between(my_periods, avg, ub, hatch='\\\\',
                        alpha=.1, color=color, step='post')

    # BatteryCharge average
    avg = np.array([result2.BatteryCharge.loc[(t)].mean()
                    for t in my_periods])
    lb = np.maximum(
        avg - np.array([result2.BatteryCharge.loc[(t)].std() for t in my_periods]), 0)
    ub = avg + np.array([result2.BatteryCharge.loc[(t)].std()
                         for t in my_periods])
    color = 'green'
    axs[0].step(np.array(my_periods), avg,
                ls='--',
                lw=3,
                color=color,
                where='post',
                label='$\mathbb{E}\left[ BatteryCharge \\right]$'
                )
    axs[0].step(np.array(my_periods), lb,
                ls=':',
                lw=1,
                where='post',
                color=color  # ,
                )
    axs[0].step(np.array(my_periods), ub,
                ls=':',
                lw=1,
                where='post',
                color=color
                )
    axs[0].fill_between(my_periods, avg, lb, hatch='//',
                        alpha=.1, color=color, step='post')
    axs[0].fill_between(my_periods, avg, ub, hatch='\\\\',
                        alpha=.1, color=color, step='post')

    # BatteryDischarge average
    avg = np.array([result2.BatteryDischarge.loc[(t)].mean()
                    for t in my_periods])
    lb = np.maximum(
        avg - np.array([result2.BatteryDischarge.loc[(t)].std() for t in my_periods]), 0)
    ub = avg + np.array([result2.BatteryDischarge.loc[(t)].std()
                         for t in my_periods])
    color = 'blue'
    axs[0].step(np.array(my_periods), avg,
                ls='--',
                lw=3,
                color=color,
                where='post',
                label='$\mathbb{E}\left[ BatteryDischarge \\right]$'
                )
    axs[0].step(np.array(my_periods), lb,
                ls=':',
                lw=1,
                where='post',
                color=color  # ,
                )
    axs[0].step(np.array(my_periods), ub,
                ls=':',
                lw=1,
                where='post',
                color=color
                )
    axs[0].fill_between(my_periods, avg, lb, hatch='//',
                        alpha=.1, color=color, step='post')
    axs[0].fill_between(my_periods, avg, ub, hatch='\\\\',
                        alpha=.1, color=color, step='post')

    axs[0].axhline(result0.GridDrawCeiling[0],
                   ls=':', label='$GridDraw Ceiling$')
    axs[0].step(my_periods, prodpv.PV_kW,
                ls='-',
                lw=5,
                color='yellow',
                where='post',
                label='$PVProduction$')
    #color = 'C' + str(1)
    for s in my_scenarios:
        axs[0].step(np.array(my_periods), np.array([result2.loc[(t, s), 'GridDraw'] for t in my_periods]),
                    ls='-',
                    lw=2,
                    color='red',
                    where='post',
                    alpha=1/len(my_scenarios),
                    label='GridDraw Scenarios' if s == 0 else ''
                    )
        axs[0].step(np.array(my_periods), np.array([result2.loc[(t, s), 'GridFeed'] for t in my_periods]),
                    ls='-',
                    lw=2,
                    color='magenta',
                    where='post',
                    alpha=1/len(my_scenarios),
                    label='GridFeed Scenarios' if s == 0 else ''
                    )
        axs[0].step(np.array(my_periods), np.array([result2.loc[(t, s), 'EVChargeTotal'] for t in my_periods]),
                    ls='-',
                    lw=2,
                    color='black',
                    where='post',
                    alpha=1/len(my_scenarios),
                    label='EVChargeTotal Scenarios' if s == 0 else ''
                    )

        axs[0].step(np.array(my_periods), np.array([result2.loc[(t, s), 'BatteryCharge'] for t in my_periods]),
                    ls='-',
                    lw=2,
                    color='green',
                    where='post',
                    alpha=1/len(my_scenarios),
                    label='BatteryCharge Scenarios' if s == 0 else ''
                    )
        axs[0].step(np.array(my_periods), np.array([result2.loc[(t, s), 'BatteryDischarge'] for t in my_periods]),
                    ls='-',
                    lw=2,
                    color='blue',
                    where='post',
                    alpha=1/len(my_scenarios),
                    label='BatteryDischarge Scenarios' if s == 0 else ''
                    )

    axs[0].axvline(0.95, color='white', lw=4, ls='-.', label='Stage 1->2')
    # axs[0].annotate("Stage 1", xy=(-.3, 37), bbox=dict(
    #     facecolor='white', edgecolor='black', boxstyle='square', pad=.2))
    # axs[0].annotate("Stage 2", xy=(1.1, 37), bbox=dict(
    #     facecolor='white', edgecolor='black', boxstyle='rarrow', pad=.2))

    at = AnchoredText(
        result0.label[0] + ": $n_{max}$ = " +
        str(round(result0.GridDrawCeiling[0], 2)),
        # prop=dict(size=15),
        frameon=True,
        loc='upper center')
    at.patch.set_boxstyle("round, pad=0, rounding_size=.2")
    axs[0].add_artist(at)

    axs[0].set_xlabel('periods')
    axs[0].set_ylabel('Power (kW)')
    axs[0].set_title('Overview')
    axs[0].legend(loc='upper right')  # , fontsize=7)

    # plt.subplot(4,1,2)
    # for i, v in enumerate(my_vehicles):
    #     plt.step(my_periods, np.array([demand_ev.loadable.loc[v, t] for t in my_periods])*(i+1), '--', where='post', label='$ev^{loadable}_t$'+v)
    # ticklabels = ['n. l.']
    # ticklabels.extend(my_vehicles)
    # ticklabels
    # ticks = np.arange(0, len(ticklabels))
    # ticks
    # plt.yticks(ticks=ticks, labels=ticklabels)
    # plt.xlim(0, len(my_periods))
    # plt.xlabel('periods')
    # plt.title('Loadability per vehicle')

    # plt.subplot(4,1,3)
    # for v in my_vehicles:
    #     plt.step(my_periods, np.array([result2.EVCharge[v, t] for t in my_periods])
    #                     - np.array([demand_ev.power.loc[v, t] for t in my_periods]), '.-', where='post', label='$EVCharging_t$'+v)
    # plt.xlim(0, len(my_periods))
    # plt.xlabel('periods')
    # plt.ylabel('Power (kW)')
    # plt.title('Power per vehicle')
    # plt.legend(loc='lower right')

    # plot individual EV charge/discharge
    # EVCharge average
    for i, v in enumerate(my_vehicles):
        #np.array([result3.EVCharge[v, t] for t in my_periods]) - np.array([demandev.power.loc[v, t] for t in my_periods])
        avg = np.array([(result3.EVCharge.loc[v, t] -
                         demandev.power.loc[v, t]).mean() for t in my_periods])
        lb = avg - np.array([(result3.EVCharge.loc[v, t] -
                              demandev.power.loc[v, t]).std() for t in my_periods])
        ub = avg + np.array([(result3.EVCharge.loc[v, t] - demandev.power.loc[v, t]).std()
                             for t in my_periods])
        color = 'C' + str(i)
        axs[1].step(np.array(my_periods), avg,
                    ls='--',
                    lw=3,
                    where='post',
                    color=color,
                    label='$\mathbb{E}\left[ '+v+' \\right]$'
                    )
        axs[1].step(np.array(my_periods), lb,
                    ls=':',
                    lw=1,
                    where='post',
                    color=color  # ,
                    )
        axs[1].step(np.array(my_periods), ub,
                    ls=':',
                    lw=1,
                    where='post',
                    color=color
                    )
        axs[1].fill_between(my_periods, avg, lb,
                            hatch='//', alpha=.1, color=color, step='post')
        axs[1].fill_between(my_periods, avg, ub,
                            hatch='\\\\', alpha=.1, color=color, step='post')

    for s in my_scenarios:
        for i, v in enumerate(my_vehicles):
            color = 'C' + str(i)
            axs[1].step(np.array(my_periods), np.array([result3.EVCharge.loc[v, t, s] for t in my_periods]),
                        ls='-',
                        lw=2,
                        where='post',
                        color=color,
                        alpha=1/len(my_scenarios),
                        label=v+' Scenarios' if s == 0 else ''
                        )

    axs[1].axvline(0.95, color='white', lw=4, ls='-.', label='Stage 1->2')

    axs[1].set_xlabel('periods')
    axs[1].set_ylabel('Power (kW)')
    axs[1].set_title('Power per vehicle')
    axs[1].legend(loc='upper right', fontsize=7)

    # plot SOC
    # BatterySOC average
    avg = np.array([result2.BatterySOC.loc[(t)].mean() for t in my_instants])
    lb = np.maximum(
        avg - np.array([result2.BatterySOC.loc[(t)].std() for t in my_instants]), 0)
    ub = avg + np.array([result2.BatterySOC.loc[(t)].std()
                         for t in my_instants])
    color = 'black'
    axs[2].plot(np.array(my_instants), avg,
                ls='--',
                lw=3,
                color=color,
                label='$\mathbb{E}\left[ Battery \\right]$'
                )
    axs[2].plot(np.array(my_instants), lb,
                ls=':',
                lw=1,
                color=color  # ,
                )
    axs[2].plot(np.array(my_instants), ub,
                ls=':',
                lw=1,
                color=color
                )
    axs[2].fill_between(my_instants, avg, lb, hatch='//',
                        alpha=.1, color=color)
    axs[2].fill_between(my_instants, avg, ub, hatch='\\\\',
                        alpha=.1, color=color)

    # EVSOC average
    for i, v in enumerate(my_vehicles):
        avg = np.array([result3.EVSOC.loc[v, t].mean() for t in my_instants])
        lb = np.maximum(
            avg - np.array([result3.EVSOC.loc[v, t].std() for t in my_instants]), 0)
        ub = avg + np.array([result3.EVSOC.loc[v, t].std()
                             for t in my_instants])
        color = 'C' + str(i)
        axs[2].plot(np.array(my_instants), avg,
                    ls='--',
                    lw=3,
                    color=color,
                    label='$\mathbb{E}\left[ '+v+' \\right]$'
                    )
        axs[2].plot(np.array(my_instants), lb,
                    ls=':',
                    lw=1,
                    color=color  # ,
                    )
        axs[2].plot(np.array(my_instants), ub,
                    ls=':',
                    lw=1,
                    color=color
                    )
        axs[2].fill_between(my_instants, avg, lb,
                            hatch='//', alpha=.1, color=color)
        axs[2].fill_between(my_instants, avg, ub,
                            hatch='\\\\', alpha=.1, color=color)

    for s in my_scenarios:
        axs[2].plot(np.array(my_instants), np.array([result2.loc[(t, s), 'BatterySOC'] for t in my_instants]),
                    ls='-',
                    lw=2,
                    color='black',
                    alpha=1/len(my_scenarios),
                    label='Battery Scenarios' if s == 0 else ''
                    )

        for i, v in enumerate(my_vehicles):
            color = 'C' + str(i)
            axs[2].plot(np.array(my_instants), np.array([result3.EVSOC.loc[v, t, s] for t in my_instants]),
                        ls='-',
                        lw=2,
                        color=color,
                        alpha=1/len(my_scenarios),
                        label=v+' Scenarios' if s == 0 else ''
                        )

    axs[2].axvline(0.95, color='white', lw=4, ls='-.', label='Stage 1->2')
    # axs[1].annotate("Stage 1", xy=(-.3, 37), bbox=dict(
    #     facecolor='white', edgecolor='black', boxstyle='square', pad=.2))
    # axs[1].annotate("Stage 2", xy=(1.1, 37), bbox=dict(
    #     facecolor='white', edgecolor='black', boxstyle='rarrow', pad=.2))

    axs[2].set_xlabel('instants')
    axs[2].set_ylabel('Energy (kWh)')
    axs[2].set_title('SOC')
    axs[2].legend(loc='upper right', fontsize=7)
    fig.tight_layout()

    return(fig)
