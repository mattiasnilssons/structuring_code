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
    # (2) Why not drop rows?: To save the dates of when the spikes occured.
    # (3) Why min_size_of_spikes? = To avoid losing small energy movements, incl. timestamp issue.
    for _ in range(100):
        df_raw.loc[(df_raw.energy > df_raw.energy.shift(-1) + min_size_of_spikes) &
                   (df_raw.meter_number == df_raw.meter_number.shift(-1)), 'energy'] = df_raw.energy.shift(-1)
    return df_raw


def cooking_event(
        df_raw,
        t_between=15,
        min_active_load=0.15,
        power_capacity=1,
        time_resolution=5,
        min_cooking_event=0.1):

    df_processed = df_raw.copy()
    df_processed.timestamp = pd.to_datetime(df_processed.timestamp)
    df_processed.timestamp = np.int64(df_processed.timestamp)
    df_processed.timestamp = pd.to_datetime(df_processed.timestamp)

    # add +3 hr to timestamp
    df_processed.timestamp = df_processed.timestamp + pd.Timedelta(hours=3)
    df_processed.timezone = 'UTC+03:00'

    energy_threshold = min_active_load * power_capacity * time_resolution / 60

    # (a): column of when a load is applied
    df_processed.loc[(
        (
            (df_processed.energy.diff() > energy_threshold)
            | (df_processed.power > min_active_load * power_capacity))
        & (df_processed.meter_number == df_processed.meter_number.shift())
    ), 'load'] = df_processed.energy.diff()

    # (b): arrays for start & end
    df_processed['cooking_start'] = False
    df_processed['cooking_end'] = False

    # (c): accumulated numbering of load instances
    df_processed['load_count'] = 0  # start
    df_processed.loc[(df_processed.load.isnull()
                      == False), 'load_count'] += 1
    df_processed.load_count = df_processed.load_count.cumsum()

    # (d): timestamp of load instance
    load_instance = df_processed.groupby('load_count').first()
    load_instance.reset_index(inplace=True)
    df_processed['timestamp_load'] = df_processed.load_count.map(
        load_instance.set_index('load_count')['timestamp'].to_dict())

    # (e): defining cooking events

    df_processed.loc[(df_processed.meter_number.diff(-1) == 0),
                     'next_timestamp'] = df_processed.timestamp.shift(-1) - df_processed.timestamp

    df_processed.loc[
        (((df_processed.timestamp -
           df_processed.timestamp_load.shift() > pd.to_timedelta(
               t_between,
               unit='m')) & (
            df_processed.energy.diff() >= energy_threshold)) | (
            df_processed.meter_number != df_processed.meter_number.shift())),
        'cooking_start'] = True

    df_processed.loc[
        (
            (df_processed.energy.diff() >= energy_threshold)
            & (df_processed.timestamp.diff() < pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_start'] = False

    df_processed.loc[
        (
            (df_processed.timestamp_load.diff() > pd.to_timedelta(t_between + time_resolution, unit='m')) 
            & (df_processed.power >= min_active_load * power_capacity)
            ), 'cooking_start'] = True

    df_processed.loc[
        (
            (df_processed.cooking_start.shift(-1))
            | (df_processed.meter_number != df_processed.meter_number.shift(-1))
        ), 'cooking_end'] = True

    df_processed.loc[
        (
            (df_processed.cooking_end.shift(-1))
            & (df_processed.next_timestamp > pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_end'] = True

    df_processed.loc[
        (
            (df_processed.cooking_end.shift())
            & (df_processed.cooking_end)
        ), 'cooking_start'] = True

    df_processed.loc[
        ((df_processed.cooking_start.shift()) & (
            df_processed.cooking_start) & (
            df_processed.timestamp.diff() <= pd.to_timedelta(
                t_between,
                unit='m')) & (
                    df_processed.next_timestamp > pd.to_timedelta(
                        t_between,
                        unit='m'))),
        'cooking_start'] = False

    df_processed.loc[
        (
            (df_processed.cooking_end.shift())
            & (df_processed.cooking_end)
            & (df_processed.timestamp.diff() > pd.to_timedelta(t_between, unit='m'))
            & (df_processed.next_timestamp <= pd.to_timedelta(t_between, unit='m'))
        ), 'cooking_end'] = False

    df_processed.loc[
        ((df_processed.cooking_start.shift()) & (
            df_processed.cooking_start) & (
            df_processed.timestamp.diff() < pd.to_timedelta(
                t_between,
                unit='m')) & (
                    df_processed.power.shift() >= min_active_load *
                    power_capacity)),
        'cooking_start'] = False

    df_processed.loc[
        (df_processed.meter_number.diff() != 0), 'cooking_start'] = True
    df_processed.loc[
        (df_processed.meter_number.diff() != 0), 'cooking_end'] = False

    # (f): accumulated numbering of cooking events
    df_processed['cooking_event'] = 0
    df_processed.cooking_event += df_processed['cooking_start']
    df_processed.cooking_event = df_processed['cooking_event'].cumsum()

    # (g): get start & end timestamp of cooking events
    start_cooking = df_processed.groupby('cooking_event').first()
    start_cooking.reset_index(inplace=True)
    df_processed['time_start'] = df_processed.cooking_event.map(
        start_cooking.set_index('cooking_event')['timestamp'].to_dict())

    end_cooking = df_processed.groupby('cooking_event').last()
    end_cooking.reset_index(inplace=True)
    df_processed['time_end'] = df_processed.cooking_event.map(
        end_cooking.set_index('cooking_event')['timestamp'].to_dict())

    # (h): get duration of cooking event and sequence time during cooking event
    df_processed['cooking_time'] = (
        df_processed.time_end - df_processed.time_start) / np.timedelta64(1, 'm')
    df_processed['seq_time'] = (
        df_processed.timestamp - df_processed.time_start) / np.timedelta64(1, 'm')

    # (i): disqualify too short cooking events
    df_processed.loc[((df_processed.cooking_event != df_processed.cooking_event.shift())
                      & (df_processed.cooking_event != df_processed.cooking_event.shift(-1))
                      & (df_processed.energy.diff() < min_cooking_event)), 'cooking_event'] = np.nan
    df_processed.set_index('timestamp', inplace=True)
    return df_processed


def timestamp_issue(df_processed, error_margin=0.01):
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
        max_cooking_time=175,
        power_mean_min=0.1,
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

    # (d): removing cooking events with too low mean power level, too long or 0 kWh energy generation
    df_only_events.drop(df_only_events[((df_only_events.power_mean < power_mean_min) | (
        df_only_events.cooking_time > max_cooking_time))].index, inplace=True)
    df_only_events.drop(
        df_only_events[(df_only_events.energy_gen == 0)].index, inplace=True)

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

'''
# Source file
df_raw = pd.read_csv('dataframe_raw.csv', sep=',')

df_raw = remove_spikes(df_raw)
df_processed = cooking_event(df_raw)
df_epc = timestamp_issue(df_processed)
df_only_events = only_events(df_epc)
df_period = period(df_only_events)

#df_epc = df_epc.loc[df_epc['meter_number']==546375]
'''