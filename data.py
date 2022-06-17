# import datetime
# import pytz
import pandas as pd
import pickle
import numpy as np


def load_pv_data(freq=60):
    my_pv = pd.read_pickle(
        "../../data/dsmdata/aggregate/processed/my_df_sync.pickle")
    my_pv.drop(columns=['RG_MW', 'DA_EUR/MWh', 'PV_kW_pred'],
               inplace=True)  # drop unneeded columns
    # my_pv = my_pv.resample('H').mean()  # resample to hourly
    my_pv = my_pv.resample(str(freq)+'T').mean().pad()  # resample to hourly
    return my_pv


def load_ev_data(num_vehicles=8, freq=60):
    # get all electric vehicles:
    with open('../../data/dsmdata/aggregate/processed/ev_participant_IDs.pickle', 'rb') as fp:
        all_vehicles = pickle.load(fp)

    my_vehicles = all_vehicles[0:num_vehicles]
    appended_data = []
    for v in my_vehicles:
        my_vehicle = pd.read_pickle(
            f"../../data/dsmdata/aggregate/processed/ev_df_{v}.pickle")
        # my_vehicle = my_vehicle.resample('H').mean()  # resample to hourly
        my_vehicle.drop(columns=['charging_bool'],
                        inplace=True)  # drop unneeded columns
        my_vehicle = my_vehicle.resample(str(freq)+'T').agg(
            {'driving_bool': 'max', 'power_kW': 'mean', 'loadable_bool': 'min'}).pad().round(2)  # resample to hourly
        oldindex = my_vehicle.index
        # remove original index
        my_vehicle.reset_index(inplace=True, drop=True)
        newindex = pd.MultiIndex.from_tuples(
            list(zip([v]*len(oldindex), oldindex)), names=['vehicle', 'timestamp'])
        # set new index of vehicle and period
        my_vehicle = my_vehicle.set_index(newindex)
        appended_data.append(my_vehicle)
    my_ev = pd.concat(appended_data, axis=0)
    my_ev.rename(columns={"driving_bool": "driving", "power_kW": "power",
                          "loadable_bool": "loadable"}, inplace=True)
    my_ev.driving = my_ev.driving.astype(int)
    my_ev.power = my_ev.power.astype(float)
    my_ev.loadable = my_ev.loadable.astype(int)
    return my_ev


def get_ev_demand_real(ev_data, from_ts, to_ts):
    # prepare data based on periods
    my_vehicles = ev_data.index.get_level_values('vehicle').unique()
    my_periods = pd.date_range(start=from_ts,
                               end=to_ts,
                               freq='H'
                               # freq='15min'
                               )
    periods = np.arange(0, len(my_periods))

    ev_temp = ev_data.loc[(ev_data.index.get_level_values('timestamp') >= from_ts) &
                          (ev_data.index.get_level_values('timestamp') <= to_ts)].copy()
    ev_temp.reset_index(inplace=True, drop=True)  # remove original index
    newindex = pd.MultiIndex.from_product(
        [my_vehicles, periods], names=['vehicle', 'period'])
    # set new index of vehicle and period
    ev_temp = ev_temp.set_index(newindex)

    # detect end of loadable period and generate loadend flag
    ev_temp['loadend'] = 0
    ev_temp.loc[(ev_temp.loadable > ev_temp.loadable.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'loadend'] = 1
    return ev_temp.sort_index()


def get_ev_demand_scenarios(ev_data, from_ts, to_ts, num_scenarios):
    # get scenario data from past periods and compile into one dataframe

    # get current data for patching in actual period into scenarios
    ev_now = get_ev_demand_real(ev_data, from_ts, to_ts)
    ev_now.reset_index(inplace=True)  # reset index on current data
    # only keep current period, discarding future values
    ev_now = ev_now.loc[ev_now.period == 0]

    appended_data = []

    for s in np.arange(0, num_scenarios):

        s_from_ts = from_ts - pd.Timedelta((s+1), "days")
        s_to_ts = to_ts - pd.Timedelta((s+1), "days")

        ev_temp = get_ev_demand_real(ev_data, s_from_ts, s_to_ts)
        ev_temp.reset_index(inplace=True)
        # discard current period to make room for actual values
        ev_temp = ev_temp.loc[ev_temp.period > 0]
        # merge current period 0 with scenarios to make them all start in the same place
        ev_temp = pd.concat([ev_now, ev_temp])

        # add new index column for scenario numbering
        ev_temp['scenario'] = s
        ev_temp.set_index(['vehicle', 'period', 'scenario'], inplace=True)

        # detect end of loadable period and generate loadend flag
        ev_temp['loadend'] = 0
        ev_temp.loc[(ev_temp.loadable > ev_temp.loadable.shift(-1)) &
                    (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'loadend'] = 1

        appended_data.append(ev_temp)

    ev_final = pd.concat(appended_data)

    return ev_final.sort_index()


# functions for simulation

def get_current_pv_data(my_pv: pd.DataFrame, my_timestamp: pd.Timestamp) -> pd.DataFrame:
    pv_temp = my_pv.loc[my_timestamp:my_timestamp].copy()
    pv_temp.reset_index(inplace=True, drop=True)
    return pv_temp


def get_current_ev_data(my_ev: pd.DataFrame, my_timestamp: pd.Timestamp) -> pd.DataFrame:
    ev_temp = my_ev.xs(my_timestamp, level='timestamp').copy()
    # alternative: ev_data.loc[pd.IndexSlice[:, start:start],:]
    ev_temp['period'] = 0
    ev_temp.reset_index(inplace=True)
    ev_temp.set_index(['vehicle', 'period'], inplace=True)
    return ev_temp  # .astype(float)


def get_history_pv_data(my_pv: pd.DataFrame, my_from: pd.Timestamp, my_to: pd.Timestamp) -> pd.DataFrame:
    pv_temp = my_pv.loc[my_from:my_to].copy()
    return pv_temp


def get_history_ev_data(my_ev: pd.DataFrame, my_from: pd.Timestamp, my_to: pd.Timestamp) -> pd.DataFrame:
    ev_temp = my_ev.loc[pd.IndexSlice[:, my_from:my_to], :].copy()
    return ev_temp


def predict_pv_data(my_pv: pd.DataFrame, my_lookahead: int) -> pd.DataFrame:
    pv_temp = my_pv.copy()
    pv_temp['timestamp'] = pv_temp.index
    pv_temp.reset_index(drop=True, inplace=True)
    pv_temp['period'] = pv_temp.index % my_lookahead
    pv_temp = pv_temp.groupby('period').mean()
    return pv_temp


def predict_ev_data(my_ev: pd.DataFrame, my_lookahead: int) -> pd.DataFrame:
    ev_temp = my_ev.copy()
    #ev_temp['timestamp'] = pv_temp.index
    ev_temp.reset_index(drop=False, inplace=True)
    ev_temp['period'] = ev_temp.groupby('vehicle').cumcount() % my_lookahead
    #ev_temp['period'] = ev_temp.index % my_lookahead
    ev_temp = ev_temp.groupby(['vehicle', 'period']).mean()
    ev_temp['driving'] = np.sign(ev_temp['driving'])
    ev_temp['loadable'] = np.sign(ev_temp['loadable'])
    ev_temp.loc[ev_temp['driving'] > 0, 'loadable'] = 0
    # todo: loadend?
    return ev_temp


def get_model_pv_data(my_current_pv: pd.DataFrame, my_predicted_pv: pd.DataFrame) -> pd.DataFrame:
    pv_temp = pd.concat(
        [my_current_pv, my_predicted_pv.loc[pd.IndexSlice[1:], :]]).sort_index().copy()
    return pv_temp


def get_model_ev_data(my_current_ev: pd.DataFrame, my_predicted_ev: pd.DataFrame) -> pd.DataFrame:
    ev_temp = pd.concat(
        [my_current_ev, my_predicted_ev.loc[pd.IndexSlice[:, 1:], :]]).sort_index().copy()

    # detect end of loadable period and generate loadend flag
    ev_temp['loadend'] = False
    ev_temp.loc[(ev_temp.loadable > ev_temp.loadable.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'loadend'] = True

    # detect end of driving period and generate driveend flag
    ev_temp['driveend'] = False
    ev_temp.loc[(ev_temp.driving > ev_temp.driving.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'driveend'] = True

    # detect beginning of loadable period and generate loadbeg flag
    ev_temp['loadbeg'] = False
    ev_temp.loc[(ev_temp.loadable < ev_temp.loadable.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'loadbeg'] = True

    # detect beginning of driving period and generate driveend flag
    ev_temp['driveend'] = False
    ev_temp.loc[(ev_temp.driving < ev_temp.driving.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'drivebeg'] = True

    return ev_temp


def scenarios_ev_data(my_ev: pd.DataFrame, my_lookahead: int) -> pd.DataFrame:
    ev_temp = my_ev.sort_index().copy()
    ev_temp.reset_index(drop=False, inplace=True)
    ev_temp['period'] = ev_temp.groupby('vehicle').cumcount() % my_lookahead
    ev_temp['scenario'] = ev_temp.groupby(['vehicle', 'period']).cumcount()
    ev_temp = ev_temp.groupby(['vehicle', 'period', 'scenario']).mean()
    return ev_temp.sort_index()


def get_model_ev_scenarios(current_ev: pd.DataFrame, scenarios_ev: pd.DataFrame) -> pd.DataFrame:
    my_scenarios = scenarios_ev.index.get_level_values('scenario').unique()
    my_current_ev = current_ev.sort_index().copy()

    appended_data = list()
    for i in my_scenarios:
        my_temp = my_current_ev.copy()
        my_temp['scenario'] = i
        appended_data.append(my_temp)

    ev_temp = pd.concat(appended_data)
    ev_temp.reset_index(drop=False, inplace=True)
    ev_temp.set_index(['vehicle', 'period', 'scenario'], inplace=True)

    ev_temp = pd.concat(
        [ev_temp, scenarios_ev.loc[pd.IndexSlice[:, 1:, :], :]]).copy().sort_index()

    # change index from vehicle-period-scenario to vehicle-scenario-period
    ev_temp.reset_index(drop=False, inplace=True)
    ev_temp.set_index(['vehicle', 'scenario', 'period'], inplace=True)
    ev_temp = ev_temp.sort_index()

    # detect end of loadable period and generate loadend flag
    ev_temp['loadend'] = False
    ev_temp.loc[(ev_temp.loadable > ev_temp.loadable.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'loadend'] = True

    # detect end of driving period and generate driveend flag
    ev_temp['driveend'] = False
    ev_temp.loc[(ev_temp.driving > ev_temp.driving.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'driveend'] = True

    # detect beginning of loadable period and generate loadbeg flag
    ev_temp['loadbeg'] = False
    ev_temp.loc[(ev_temp.loadable > ev_temp.loadable.shift(-1)) &
                (ev_temp.index.get_level_values('period') < max(ev_temp.index.get_level_values('period'))-1), 'loadbeg'] = True

    # change index back to vehicle-period-scenario
    ev_temp.reset_index(drop=False, inplace=True)
    ev_temp.set_index(['vehicle', 'period', 'scenario'], inplace=True)
    ev_temp = ev_temp.sort_index()

    return ev_temp
