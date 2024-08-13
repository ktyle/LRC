#!/usr/bin/env python3
# Create the Large Radiosonde Collection archive from IGRA2 archive data
# Created 14 July 2024 by Sam Gardner <samuel.gardner@ttu.edu>

from datetime import datetime as dt, timedelta
import tarfile
from io import BytesIO
from zipfile import ZipFile
from os import path, listdir
import numpy as np
import polars as pl
from time import sleep

from dask.distributed import Client


def parse_zipped_text(z, txt):
    from metpy.units import units
    from metpy import calc as mpcalc
    # Each zip file represents a single station, which has many launches.
    all_dfs = []
    z = ZipFile(BytesIO(z))
    with z.open(txt) as this_txt:
        this_txt = this_txt.readlines()
        lines = [line.decode('utf-8') for line in this_txt]
        lines_arr = np.array(lines)
        # All headers start with a '#'
        header_indices = np.argwhere(np.strings.startswith(lines_arr, '#')).flatten()

        for header_idx in header_indices:
            # Parse the header line
            # https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-data-format.txt
            header_line = lines[header_idx]
            station = header_line[1:11]
            valid_year = int(header_line[13:17])
            valid_month = int(header_line[18:20])
            valid_day = int(header_line[21:23])
            valid_hour = int(header_line[24:26])
            release_hhmm = int(header_line[27:31])
            num_rec = int(header_line[32:36])
            pressure_source = header_line[37:45]
            nonpressure_source = header_line[46:54]
            launch_lat = float(header_line[55:62])/10000
            launch_lon = float(header_line[64:71])/10000
            data_lines = lines_arr[header_idx+1:header_idx+num_rec+1]

            # Create a dataframe from the data lines. Currently the dataframe only has one column, representing each record of the data
            df = pl.read_csv(''.join(data_lines).encode('utf-8'), has_header=False, new_columns=['row'])
            # Split the single column into multiple columns, representing each variable.
            names = ['major_level_indicator', 'minor_level_indicator', 'elapsed_time', 'air_pressure', 'pflag', 'geopotential_height', 'zflag', 'air_temperature',
                    'tflag', 'relative_humidity', 'dewpoint_depression', 'wind_from_direction', 'wind_speed']
            # Each tuple is the start index and length of the variable in the row.
            # Documented at https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-data-format.txt
            starts_and_lengths = [(0, 1), # col 1, LVLTYP1
                                (1, 1), # col 2, LVLTYP2
                                (3, 5), # col 4 through 8, ETIME
                                (9, 6), # col 10 through 15, PRESS
                                (15, 1), # col 16, PFLAG
                                (16, 5), # col 17 through 21, GPH
                                (21, 1), # col 22, ZFLAG
                                (22, 5), # col 23 through 27, TEMP
                                (27, 1), # col 28, TFLAG
                                (28, 5), # col 29 through 33, RH
                                (34, 5), # col 35 through 39, DPDP
                                (40, 5), # col 41 through 45, WDIR
                                (46, 6), # col 47 through 51, WSPD
                                ]
            names_and_locs = dict(zip(names, starts_and_lengths))
            # datatypes also from the documentation above.
            types = [pl.UInt8, pl.UInt8, pl.Int32, pl.Int32, pl.String, pl.Int32, pl.String, pl.Int32, pl.String, pl.Int32, pl.Int32, pl.Int32, pl.Int32]
            # Split the row into columns and drop the original column representing the entire row.
            df = df.with_columns(
                [pl.col('row').str.slice(*start_and_length).str.replace_all(' ', '').alias(name) for name, start_and_length in names_and_locs.items()]
            ).drop('row')
            # Cast the columns to the correct datatypes
            df = df.with_columns(
                [pl.col(name).cast(dtype) for name, dtype in zip(names, types)]
            )
            # Replace missing values with NaN
            df = df.with_columns(
                [pl.when(pl.col(name).is_in([-9999, -8888])).then(np.nan).otherwise(pl.col(name)).alias(name) for name in names if df[name].dtype != pl.String]
            )
            # Data is delivered in pascals, tenths of degrees celsius, and tenths of meters per second.
            df = df.with_columns(
                pl.col('air_pressure')/100
            )
            df = df.with_columns(
                (pl.col('air_temperature')/10)
            )
            # Calculate dew point from dewpoint depression
            df = df.with_columns(
                (pl.col('air_temperature') - pl.col('dewpoint_depression')/10).alias('dew_point_temperature')
            )
            df = df.with_columns(
                (pl.col('wind_speed')/10)
            )

            # Calculate wind components
            u, v = mpcalc.wind_components(df['wind_speed'].to_numpy() * units('m/s'), df['wind_from_direction'].to_numpy() * units('degrees'))

            df = df.with_columns(
                eastward_wind=u.magnitude,
                northward_wind=v.magnitude
            )

            # Try to find the surface record and record the launch altitude above MSL
            launch_msl = df.filter(pl.col('minor_level_indicator') == 1)['geopotential_height'].to_numpy()
            if len(launch_msl) == 1:
                launch_msl = launch_msl[0]
            elif len(launch_msl) == 0:
                launch_msl = np.nan
            else:
                raise ValueError('More than one surface record found')

            # Drop columns that are all NaN or empty strings
            for this_col_name in df.columns:
                this_col = df[this_col_name]
                if this_col.dtype == pl.Float64:
                    if this_col.is_nan().all():
                        df = df.drop(this_col_name)
                elif this_col.dtype == pl.String:
                    if (this_col == '').all():
                        df = df.drop(this_col_name)
            
            # Drop unused columns
            df = df.drop('pflag', 'zflag', 'tflag')
            df = df.drop('major_level_indicator', 'minor_level_indicator', 'dewpoint_depression')

            # Calculate the launch valid time
            launch_valid_time = dt(valid_year, valid_month, valid_day, valid_hour)
            # Some soundings have a specific release time included
            if release_hhmm != 9999:
                if release_hhmm % 100 == 99:
                    release_hhmm -= 99
                release_time = dt(valid_year, valid_month, valid_day, release_hhmm//100, release_hhmm%100)
                # Many 0z launches are launched late in the previous day. Correct for this.
                if launch_valid_time.hour == 0 and release_time.hour >= 21:
                    release_time = release_time - timedelta(days=1)
            else:
                release_time = np.nan

            # Some soundings have the exact time of each record.
            if 'elapsed_time' in df.columns:
                elapsed_hours = pl.Series(df['elapsed_time']//100)
                elapsed_minutes = pl.Series(df['elapsed_time']%100)
                if release_time is not np.nan:
                    rec_valid_times = pl.Series([release_time + timedelta(hours=hours, minutes=minutes) for hours, minutes in zip(elapsed_hours, elapsed_minutes)])
                else:
                    rec_valid_times = pl.Series([launch_valid_time + timedelta(hours=hours, minutes=minutes) for hours, minutes in zip(elapsed_hours, elapsed_minutes)])
            else:
                if release_time is not np.nan:
                    rec_valid_times = np.full(num_rec, release_time, dtype=object)
                else:
                    rec_valid_times = np.full(num_rec, launch_valid_time, dtype=object)
                    
            # Add our new columns to the overall dataset
            df = df.with_columns(
                    site=np.full(num_rec, station),
                    launch_valid_time=np.full(num_rec, launch_valid_time),
                    release_time=np.full(num_rec, release_time),
                    launch_lat=np.full(num_rec, launch_lat),
                    launch_lon=np.full(num_rec, launch_lon),
                    launch_msl=np.full(num_rec, launch_msl),
                    record_valid=rec_valid_times
                )
            all_dfs.append(df)
    # Concatenate all dataframes
    this_station_df = pl.concat(all_dfs)
    return this_station_df


def get_soundings_from_tar(t, dask_client):
    all_txts = []
    # Each tarfile has many zip files inside, each representing a different station
    for member in t:
        if member.name.endswith('.zip'):
            zip_bytes = t.extractfile(member).read()
            zp = BytesIO(zip_bytes)
            with ZipFile(zp) as z:
                txts_to_read = [txt for txt in z.namelist() if txt.endswith('.txt')]
                all_txts.extend(txts_to_read)
    all_dfs = dask_client.map(parse_zipped_text, [zip_bytes]*len(all_txts), txts_to_read)
    # Concatenate all dataframes
    tar_df = dask_client.submit(pl.concat, all_dfs).result()
    print(tar_df)
    return tar_df

if __name__ == '__main__':
    dask_client = Client()
    print(dask_client.dashboard_link)
    sleep(5)
    # Create container for final archive
    all_dfs = []
    # Read in data from all tar files in input_data/
    input_path = 'input_data/'
    for in_filename in sorted(listdir(input_path)):
        if not in_filename.endswith('.tar'):
            continue
        input_filepath = path.join(input_path, in_filename)
        # Create dataframe from the tar file
        with tarfile.open(input_filepath) as t:
            this_tar_df = get_soundings_from_tar(t, dask_client)
        all_dfs.append(this_tar_df)
    # Concatenate all dataframes
    all_radiosondes = all_dfs[0]
    for i in range(1, len(all_dfs)):
        all_radiosondes = all_radiosondes.append(all_dfs[i])
    # Write to parquet
    all_radiosondes.write_parquet('radiosondes.parquet')