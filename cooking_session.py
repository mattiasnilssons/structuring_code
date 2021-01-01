#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 08:30:08 2020

@author: Mattias Nilsson
"""
# Packages
import pandas as pd
import numpy as np


def remove_spikes(df_raw, min_size_of_spikes=1):
    # (1) Why range(100)?: There are more than 50 recordings in some of the spikes.
    # (2) Why not drop rows?: To save the dates of when the spikes occur.
    # (3) Why min_size_of_spikes? = To avoid losing small energy movements, incl. timestamp issue.
    for _ in range(100):
        df_raw.loc[(df_raw.energy > df_raw.energy.shift(-1) + min_size_of_spikes) &
                   (df_raw.meter_number == df_raw.meter_number.shift(-1)), 'energy'] = df_raw.energy.shift(-1)
    return df_raw


def cooking_event(
        df_raw,
        min_cooking_event=0.05,
        power_mean_min=0.05):

    df_processed = df_raw.copy()

    # Format 'timestamp' column
    if 'timestamp' in df_processed.columns:
        df_processed.timestamp = pd.to_datetime(df_processed.timestamp)
        df_processed.timestamp = np.int64(df_processed.timestamp)
        df_processed.timestamp = pd.to_datetime(df_processed.timestamp)
    else:
        df_processed.reset_index(inplace=True)

    # Check 'UTC+00:00' in column 'timezone'
    boolean_zone = df_processed['timezone'].str.contains('UTC+00:00').any()
    if boolean_zone:
        # adding +3 hr to timestamp
        df_processed.timestamp += pd.Timedelta(hours=3)
        df_processed.timezone = 'UTC+03:00'

    # Create columns based on columns 'meter_number' and 'timestamp' by
    # selecting the time difference between rows for each meter_number to
    # conduct the further analysis
    df_processed.loc[(df_processed.meter_number.diff() == 0),
                     'diff_prev_timestamp'] = df_processed.timestamp.diff()
    df_processed.loc[(df_processed.meter_number.diff(-1) == 0),
                     'diff_next_timestamp'] = df_processed.timestamp.shift(-1) - df_processed.timestamp

    # Create columns for Cooking 'start' & 'end'
    df_processed['cooking_start'] = False
    df_processed['cooking_end'] = False

    # Create distinct cooking events
    df_processed = event_conditions(df_processed)

    # Create a column 'cooking_event' for accumulated numbering of cooking
    # events
    df_processed['cooking_event'] = 0
    df_processed.cooking_event += df_processed['cooking_start']
    df_processed.cooking_event = df_processed['cooking_event'].cumsum()

    # Create columns to show start & end timestamp of each cooking event
    start_cooking = df_processed.groupby('cooking_event').first()
    start_cooking.reset_index(inplace=True)
    df_processed['time_start'] = df_processed.cooking_event.map(
        start_cooking.set_index('cooking_event')['timestamp'].to_dict())
    df_processed['energy_start'] = df_processed.cooking_event.map(
        start_cooking.set_index('cooking_event')['energy'].to_dict())

    end_cooking = df_processed.copy()
    end_cooking = end_cooking.loc[(end_cooking['cooking_end'])]
    end_cooking = end_cooking.groupby(['cooking_event']).first()
    end_cooking_lack_regular_endpoint = df_processed.copy()
    end_cooking_lack_regular_endpoint = end_cooking_lack_regular_endpoint.groupby([
                                                                                  'cooking_event']).last()
    end_cooking = end_cooking.append(end_cooking_lack_regular_endpoint)
    end_cooking.reset_index(inplace=True)

    df_processed['time_end'] = df_processed.cooking_event.map(
        end_cooking.set_index('cooking_event')['timestamp'].to_dict())
    df_processed['energy_end'] = df_processed.cooking_event.map(
        end_cooking.set_index('cooking_event')['energy'].to_dict())

    df_processed.loc[((df_processed.timestamp > df_processed.time_end)
                      ), 'cooking_event'] = np.nan

    df_processed.loc[((df_processed.timestamp > df_processed.time_end)
                      ), 'time_start'] = np.nan

    # Create columns for getting duration of cooking event and sequence time
    # during cooking event
    df_processed['cooking_time'] = (
        df_processed.time_end - df_processed.time_start) / np.timedelta64(1, 'm')
    df_processed['seq_time'] = (
        df_processed.timestamp - df_processed.time_start) / np.timedelta64(1, 'm')

    # Disqualify too short cooking events
    df_processed.loc[
        (
            (df_processed.cooking_event != df_processed.cooking_event.shift())
            & (df_processed.cooking_event != df_processed.cooking_event.shift(-1))
            & (df_processed.energy.diff() < min_cooking_event)
        ), 'cooking_event'] = np.nan

    # Disqualify cooking events of 'too low' average energy
    df_processed.loc[((df_processed.energy_end -
                       df_processed.energy_start < min_cooking_event) | ((df_processed.energy_end -
                                                                          df_processed.energy_start) /
                                                                         (df_processed.cooking_time /
                                                                          60) < power_mean_min)), 'cooking_event'] = np.nan

    df_processed.set_index('timestamp', inplace=True)

    df_processed.loc[((df_processed.cooking_event.isnull())
                      ), 'cooking_time'] = np.nan

    df_processed.loc[((df_processed.cooking_event.isnull())
                      ), 'seq_time'] = np.nan

    return df_processed


def event_conditions(df_processed,
                     min_active_load=0.15,
                     power_capacity=1,
                     time_resolution=5,
                     t_between=15):

    # (i): create coefficients to indicate when an EPC is turned ON
    power_threshold = min_active_load * power_capacity
    energy_threshold = power_threshold * time_resolution / 60

    # (ii): Create column 'load' for when a load is applied.
    df_processed.loc[(
        (
            (df_processed.energy.diff() > energy_threshold)
            | (df_processed.power > min_active_load * power_capacity))
        & (df_processed.meter_number == df_processed.meter_number.shift())
    ), 'load'] = df_processed.energy.diff()

    # (iii): Create a column 'load_count' for accumulated numbering of when a load is applied.
    df_processed['load_count'] = 0  # start
    df_processed.loc[(df_processed.load.isnull()
                      == False), 'load_count'] += 1
    df_processed.load_count = df_processed.load_count.cumsum()

    # (iv): Create a column 'timestamp_load' for a timestamp of each load instance
    load_instance = df_processed.groupby('load_count').first()
    load_instance.reset_index(inplace=True)
    df_processed['timestamp_load'] = df_processed.load_count.map(
        load_instance.set_index('load_count')['timestamp'].to_dict())

    # (v): Cooking_start = TRUE: if timestamp_load - current timestamp is more than t_between and above energy_threshold OR new meter_number
    df_processed.loc[
        (
            (
                (df_processed.timestamp -
                 df_processed.timestamp_load.shift() > pd.to_timedelta(
                     t_between,
                     unit='m'))
                & (df_processed.energy.diff() >= energy_threshold))
            | (
                df_processed.meter_number != df_processed.meter_number.shift())
        ), 'cooking_start'] = True

    # (vi): Cooking_start = FALSE: if energy increase is above energy threshold and diff_prev_timestamp is less than t_between
    df_processed.loc[
        (
            (df_processed.energy.diff() >= energy_threshold)
            & (df_processed.diff_prev_timestamp < pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_start'] = False

    # (vii): Cooking_start = TRUE: if previous to current timestamp_load difference is above t_between + time_resolution AND power level is above 'power threshold', i.e. min_active_load * power_capacity
    df_processed.loc[
        (
            (df_processed.timestamp_load.diff() > pd.to_timedelta(t_between + time_resolution, unit='m'))
            & (df_processed.power >= power_threshold)
        ), 'cooking_start'] = True

    # (viii): Cooking_end = TRUE: if difference between current timestamp and timestamp_load is above t_between AND power is above threshold on current and previous row AND same meter_number are all TRUE.
    df_processed.loc[
        (
            (df_processed.timestamp - df_processed.timestamp_load > pd.to_timedelta(
                t_between, unit='m'))
            & ((df_processed.power < power_threshold)
               & (df_processed.power.shift() < power_threshold)
               & (df_processed.meter_number == df_processed.meter_number.shift())
               )
            | (df_processed.energy - df_processed.energy.shift(-1) == 0)
        ), 'cooking_end'] = True

    # (ix): Cooking_start = TRUE: if difference between current timestamp and timestamp_load is above t_between AND power above power_threshold
    df_processed.loc[
        (
            (df_processed.timestamp - df_processed.timestamp_load > pd.to_timedelta(
                t_between, unit='m'))
            & (df_processed.power >= power_threshold)
        ), 'cooking_start'] = True

    # (x): Cooking_end = TRUE: if cooking_start in next row is TRUE OR new meter_number
    df_processed.loc[
        (
            (df_processed.cooking_start.shift(-1))
            | (df_processed.meter_number != df_processed.meter_number.shift(-1))
        ), 'cooking_end'] = True

    # (xi): Cooking_end = TRUE: if cooking_end in next row is TRUE AND diff_next_timestamp is above t_between
    df_processed.loc[
        (
            (df_processed.cooking_end.shift(-1))
            & (df_processed.diff_next_timestamp > pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_end'] = True

    # (xii): Cooking_start = TRUE: if cooking_end on prev row AND cooking_end on current row
    df_processed.loc[
        (
            (df_processed.cooking_end.shift())
            & (df_processed.cooking_end)
        ), 'cooking_start'] = True

    # (xiii): Cooking_start = FALSE: if cooking_start on prev row AND cooking_start = TRUE in current row AND diff_prev_timestamp is less than t_between AND diff_next_timestamp is more than t_between.
    df_processed.loc[
        (
            (df_processed.cooking_start.shift())
            & (df_processed.cooking_start)
            & (df_processed.diff_prev_timestamp <= pd.to_timedelta(
                t_between, unit='m'))
            & (df_processed.diff_next_timestamp > pd.to_timedelta(
                t_between, unit='m'))
        ), 'cooking_start'] = False

    # (xiv): Cooking_end = FALSE: if cooking_end on prev row AND cooking_end in current row == TRUE AND diff_prev_timestamp is more than t_between AND diff_next_timestamp is less than t_between.
    df_processed.loc[
        (
            (df_processed.cooking_end.shift())
            & (df_processed.cooking_end)
            & (df_processed.diff_prev_timestamp > pd.to_timedelta(t_between, unit='m'))
            & (df_processed.diff_next_timestamp <= pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_end'] = False

    # (xv): Cooking_start = FALSE: if cooking_start in prev row = TRUE AND cooking_start in current row = TRUE AND diff_prev_timestamp is less than t_between AND prev row has power above threshold.
    df_processed.loc[
        (
            (df_processed.cooking_start.shift())
            & (df_processed.cooking_start)
            & (df_processed.diff_prev_timestamp < pd.to_timedelta(
                t_between, unit='m'))
            & (df_processed.power.shift() >= power_threshold)
        ), 'cooking_start'] = False

    # (xvi):
    df_processed.loc[
        (
            (df_processed.cooking_end.shift(-1) == 0)
            & (df_processed.cooking_end.shift() == 0)
            & (df_processed.diff_next_timestamp > pd.to_timedelta(
                t_between, unit='m'))
        ), 'cooking_end'] = True

    # (xvii): if new meter number Cooking_start = TRUE, Cooking_end = FALSE
    df_processed.loc[
        (df_processed.meter_number.diff() != 0), 'cooking_start'] = True

    df_processed.loc[
        (df_processed.meter_number.diff() != 0), 'cooking_end'] = False

    return df_processed


def timestamp_issue(df_processed, error_margin=0.04):
    df_epc = df_processed.copy()

    # checking start of events
    start_of_event = df_epc.copy()
    start_of_event = start_of_event.groupby(
        ['meter_number', 'cooking_event']).head(1)
    start_of_event.loc[
        ((start_of_event['energy'] -
          error_margin <= start_of_event['energy'].shift()) & (
            start_of_event.cooking_event.isnull() == False) & (
            start_of_event.meter_number == start_of_event.meter_number.shift())),
        'timestamp_issue'] = True

    df_epc['timestamp_issue'] = df_epc.cooking_event.map(
        start_of_event.set_index('cooking_event')['timestamp_issue'].to_dict())

    # checking end of events
    end_of_event = df_epc.copy()
    end_of_event = end_of_event.groupby(
        ['meter_number', 'cooking_event']).tail(1)
    end_of_event.loc[
        ((end_of_event['energy'] -
          error_margin <= end_of_event['energy'].shift()) & (
            end_of_event.cooking_event.isnull() == False) & (
            end_of_event.meter_number == end_of_event.meter_number.shift())),
        'timestamp_issue'] = True

    df_epc['timestamp_issue'] = df_epc.cooking_event.map(
        end_of_event.set_index('cooking_event')['timestamp_issue'].to_dict())
    df_epc.drop(df_epc[(df_epc['timestamp_issue'] == 1)].index, inplace=True)
    return df_epc


def only_events(
        df_epc,
        TZS_per_kWh=100):
    df_only_events = df_epc.copy()
    df_only_events.reset_index(inplace=True)
    df_only_events['energy_gen'] = df_only_events.energy

    df_only_events = df_only_events.groupby(['meter_number',
                                             'cooking_event']).agg({'energy': 'max',
                                                                    'energy_gen': 'min',
                                                                    'power': 'mean',
                                                                    'cooking_time': 'max',
                                                                    'timestamp': 'min',
                                                                    'current': 'mean',
                                                                    'voltage': 'count',
                                                                    'id': 'mean'})

    # (a): counting number of recordings in each cooking event
    df_only_events.rename(
        columns={
            'voltage': 'no_recordings'},
        inplace=True)

    # (b): calculating the energy usage of each cooking event
    df_only_events.energy_gen = df_only_events.energy - df_only_events.energy_gen

    # (c): calculating the average power level during a cooking event
    df_only_events['power_mean'] = df_only_events.energy_gen / \
        (df_only_events.cooking_time / 60)

    # df_only_events.drop(
    #    df_only_events[(df_only_events.energy_gen == 0)].index, inplace=True)

    # (e): calculating the cost of cooking (Tanzanian Shilling)
    df_only_events['cooking_cost'] = df_only_events.energy_gen * TZS_per_kWh

    df_only_events.reset_index(inplace=True)
    df_only_events['event_count'] = 0
    df_only_events.loc[(df_only_events.cooking_event.diff()
                        != 0), 'event_count'] += 1
    df_only_events.event_count = df_only_events['event_count'].cumsum()

    df_only_events.set_index('timestamp', inplace=True)
    return df_only_events


def period(df, start='2020-03-09', end='2020-11-15'):
    if 'timestamp' in df.columns:
        df.timestamp = pd.to_datetime(df.timestamp)
        df.timestamp = np.int64(df.timestamp)
        df.timestamp = pd.to_datetime(df.timestamp)
        df.set_index('timestamp', inplace=True)
    df_period = df.copy()
    df_period.drop(df_period[(((df_period.index < pd.to_datetime(
        str(start) + ' 00:00:00'))))].index, inplace=True)
    df_period.drop(df_period[(((df_period.index >= pd.to_datetime(
        str(end) + ' 00:00:00'))))].index, inplace=True)
    return df_period

def addtoevent_ending(df_epc, 
                      power_capacity=1,
                     time_resolution=5):

    if 'timestamp' in df_epc.columns:
        print('timestamp is not in index')
    else:
        df_epc.reset_index(inplace=True)

    df_epc.loc[
        (
            (df_epc.cooking_event.isnull() == False)
            & (df_epc.cooking_event != df_epc.cooking_event.shift(-1))
            & (df_epc.meter_number == df_epc.meter_number.shift(-1))
        ), 'energy_gap_to_next'] = df_epc.energy.shift(-1) - df_epc.energy

    df_epc_energy_gaps = df_epc.copy()
    df_epc_energy_gaps = df_epc_energy_gaps.loc[df_epc['energy_gap_to_next'] > 0]

    df_epc_energy_gaps['energy_gap_time'] = df_epc.energy_gap_to_next / \
        power_capacity * 60
    df_epc_energy_gaps['energy_gap_time_datetime'] = df_epc_energy_gaps['energy_gap_time'] * \
        60 * np.timedelta64(1, 's')

    # Check the occurences of energy gaps with a timedelta below 5 minutes    
    energy_gap_less_5_min = (
        (df_epc_energy_gaps.energy_gap_time <= time_resolution) 
                             & (df_epc_energy_gaps.energy_gap_time > 0)
                             )
    # Check the occurences of energy gaps with a timedelta above 5 minutes   
    energy_gap_above_5_min = (df_epc_energy_gaps.energy_gap_time > time_resolution)
    
    # Update row of extended cooking event at ending, comprising updating 6 columns
    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'timestamp'] += df_epc_energy_gaps.energy_gap_time_datetime

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'time_end'] += df_epc_energy_gaps.energy_gap_time_datetime

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'cooking_time'] += df_epc_energy_gaps.energy_gap_time

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'seq_time'] += df_epc_energy_gaps.energy_gap_time

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'energy'] += df_epc_energy_gaps.energy_gap_to_next

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'energy_end'] += df_epc_energy_gaps.energy_gap_to_next

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'timestamp'] += pd.Timedelta(minutes=time_resolution)

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'time_end'] += pd.Timedelta(minutes=time_resolution)

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'cooking_time'] += time_resolution

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'seq_time'] += time_resolution

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'energy'] += time_resolution / 60

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'energy_end'] -= time_resolution / 60

    df_epc_energy_gaps = df_epc_energy_gaps.drop(['energy_gap_time', 'energy_gap_time_datetime', 'energy_gap_to_next'], axis=1)
    df_epc = df_epc.append(df_epc_energy_gaps)
    
    df_epc['time_end'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['time_end'].to_dict())
    df_epc['cooking_time'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['cooking_time'].to_dict())
    df_epc['energy_end'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['energy_end'].to_dict())
    df_epc.sort_values(by=['meter_number', 'timestamp'],
                       ascending=[True, True], inplace=True)
    df_epc.set_index('timestamp', inplace=True)
    return df_epc


def addtoevent_beginning(df_epc, 
                      power_capacity=1,
                     time_resolution=5):

    if 'timestamp' in df_epc.columns:
        print('timestamp is not in index')
    else:
        df_epc.reset_index(inplace=True)

    df_epc.loc[
        (
            (df_epc.cooking_event.isnull() == False)
            & (df_epc.cooking_event != df_epc.cooking_event.shift())
            & (df_epc.meter_number == df_epc.meter_number.shift())
        ), 'energy_gap_to_prev'] = df_epc.energy.diff()

    df_epc_energy_gaps = df_epc.copy()
    df_epc_energy_gaps = df_epc_energy_gaps.loc[df_epc['energy_gap_to_prev'] > 0]

    df_epc_energy_gaps['energy_gap_time'] = df_epc.energy_gap_to_prev / \
        power_capacity * 60
    df_epc_energy_gaps['energy_gap_time_datetime'] = df_epc_energy_gaps['energy_gap_time'] * \
        60 * np.timedelta64(1, 's')

    # Check the occurences of energy gaps with a timedelta below 5 minutes    
    energy_gap_less_5_min = (
        (df_epc_energy_gaps.energy_gap_time <= time_resolution) 
                             & (df_epc_energy_gaps.energy_gap_time > 0)
                             )
    # Check the occurences of energy gaps with a timedelta above 5 minutes   
    energy_gap_above_5_min = (df_epc_energy_gaps.energy_gap_time > time_resolution)

    # Update row of extended cooking event at beginning, comprising updating 6 columns
    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'timestamp'] -= df_epc_energy_gaps.energy_gap_time_datetime

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'time_start'] -= df_epc_energy_gaps.energy_gap_time_datetime

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'cooking_time'] += df_epc_energy_gaps.energy_gap_time

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'seq_time'] -= df_epc_energy_gaps.energy_gap_time

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'energy'] -= df_epc_energy_gaps.energy_gap_to_prev

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'energy_start'] -= df_epc_energy_gaps.energy_gap_to_prev

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'timestamp'] -= pd.Timedelta(minutes=5)

    df_epc_energy_gaps.loc[energy_gap_less_5_min, 'time_start'] -= pd.Timedelta(minutes=5)

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'cooking_time'] += 5

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'seq_time'] -= 5

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'energy'] -= 5 / 60

    df_epc_energy_gaps.loc[energy_gap_above_5_min, 'energy_start'] -= 5 / 60

    df_epc_energy_gaps = df_epc_energy_gaps.drop(['energy_gap_time', 'energy_gap_time_datetime', 'energy_gap_to_prev'], axis=1)
    df_epc = df_epc.append(df_epc_energy_gaps)

    df_epc['time_start'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['time_start'].to_dict())
    df_epc['cooking_time'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['cooking_time'].to_dict())
    df_epc['energy_start'] = df_epc.cooking_event.map(
        df_epc_energy_gaps.set_index('cooking_event')['energy_start'].to_dict())

    df_epc.sort_values(by=['meter_number', 'timestamp'],
                       ascending=[True, True], inplace=True)
    df_epc.set_index('timestamp', inplace=True)
    return df_epc


# Source file
df_raw = pd.read_csv('dataframe_raw.csv', sep=',')

df_raw = remove_spikes(df_raw)
df_processed = cooking_event(df_raw)
df_epc = timestamp_issue(df_processed)
df_epc = addtoevent_ending(df_epc)
df_epc = addtoevent_beginning(df_epc)
df_epc = cooking_event(df_epc)
df_only_events = only_events(df_epc)
df_period = period(df_only_events)

#df_epc = df_epc.loc[df_epc['meter_number']==546375]
