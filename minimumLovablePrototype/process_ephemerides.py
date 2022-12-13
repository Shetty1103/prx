import georinex as gr
import xarray
from gnss_lib_py.utils.constants import WEEKSEC
from gnss_lib_py.utils.time_conversions import datetime_to_tow, get_leap_seconds, tow_to_datetime
from gnss_lib_py.utils.sim_gnss import find_sat
# from gnss_lib_py.parsers.precise_ephemeris import parse_sp3
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from datetime import datetime, timedelta


def load_rnx3_nav_ds(path):
    """load rnx3 nav file specified in path

    If zipped, unzip.
    Load the file using the georinex package.
    return as an xarray.dataset object
    """
    # unzip file if necessary
    if not path.exists():
        print("Unzipping RNX3 NAV file...")
        # unzip file
        path_to_zip_file = path.with_suffix(".zip")
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            print(path.resolve().parent)
            zip_ref.extractall(path.resolve().parent)

    # parse RNX3 NAV file using georinex module
    nav_ds = gr.load(path)
    return nav_ds


def load_rnx3_nav_df(path):
    """load rnx3 nav file specified in path

    If zipped, unzip.
    Load the file using the georinex package.
    Convert it to pandas
    """
    nav_ds = load_rnx3_nav_ds(path)

    # convert ephemerides from xarray.Dataset to pandas.DataFrame
    nav_df = convert_nav_ds_to_df(nav_ds)

    return nav_df


def convert_nav_ds_to_df(nav_ds):
    # convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py
    nav_df = nav_ds.to_dataframe()
    nav_df.dropna(how='all', inplace=True)
    nav_df.reset_index(inplace=True)
    nav_df['source'] = nav_ds.filename
    nav_df['t_oc'] = pd.to_numeric(nav_df['time'] - datetime(1980, 1, 6, 0, 0, 0))
    nav_df['t_oc'] = 1e-9 * nav_df['t_oc'] - WEEKSEC * np.floor(1e-9 * nav_df['t_oc'] / WEEKSEC)
    nav_df['time'] = nav_df['time'].dt.tz_localize('UTC')

    nav_df.rename(
        columns={'M0': 'M_0', 'Eccentricity': 'e', 'Toe': 't_oe', 'DeltaN': 'deltaN', 'Cuc': 'C_uc', 'Cus': 'C_us',
                 'Cic': 'C_ic', 'Crc': 'C_rc', 'Cis': 'C_is', 'Crs': 'C_rs', 'Io': 'i_0', 'Omega0': 'Omega_0'},
        inplace=True)
    return nav_df


def convert_nav_single_ds_to_df(nav_ds):
    # convert ephemerides from xarray.Dataset to pandas.DataFrame, as required by gnss_lib_py
    time = pd.to_datetime(nav_ds['time'].values)
    sv = nav_ds['sv'].values
    SVclockBias = nav_ds['SVclockBias'].values
    SVclockDrift = nav_ds['SVclockDrift'].values
    SVclockDriftRate = nav_ds['SVclockDriftRate'].values
    IODE = nav_ds['IODE'].values
    C_rs = nav_ds['Crs'].values
    deltaN = nav_ds['DeltaN'].values
    M_0 = nav_ds['M0'].values
    C_uc = nav_ds['Cuc'].values
    e = nav_ds['Eccentricity'].values
    C_us = nav_ds['Cus'].values
    sqrtA = nav_ds['sqrtA'].values
    t_oe = nav_ds['Toe'].values
    C_ic = nav_ds['Cic'].values
    Omega_0 = nav_ds['Omega0'].values
    C_is = nav_ds['Cis'].values
    i_0 = nav_ds['Io'].values
    C_rc = nav_ds['Crc'].values
    omega = nav_ds['omega'].values
    OmegaDot = nav_ds['OmegaDot'].values
    IDOT = nav_ds['IDOT'].values
    CodesL2 = nav_ds['CodesL2'].values
    GPSWeek = nav_ds['GPSWeek'].values
    L2Pflag = nav_ds['L2Pflag'].values
    SVacc = nav_ds['SVacc'].values
    health = nav_ds['health'].values
    TGD = nav_ds['TGD'].values
    IODC = nav_ds['IODC'].values
    TransTime = nav_ds['TransTime'].values
    source = nav_ds.filename
    td = nav_ds['time'].values.astype('datetime64[ms]').astype(datetime) - datetime(1980, 1, 6, 0, 0, 0)
    td_np = np.timedelta64(td)
    t_oc = pd.to_numeric(td_np).astype('float')
    t_oc = 1e-9 * t_oc - WEEKSEC * np.floor(1e-9 * t_oc / WEEKSEC)

    dataframe_data = {
        'time': time, 'sv': sv, 'SVclockBias': SVclockBias, 'SVclockDrift': SVclockDrift,
        'SVclockDriftRate': SVclockDriftRate, 'IODE': IODE, 'C_rs': C_rs, 'deltaN': deltaN, 'M_0': M_0,
        'C_uc': C_uc, 'e': e, 'C_us': C_us, 'sqrtA': sqrtA, 't_oe': t_oe, 'C_ic': C_ic, 'Omega_0': Omega_0,
        'C_is': C_is, 'i_0': i_0, 'C_rc': C_rc, 'omega': omega, 'OmegaDot': OmegaDot, 'IDOT': IDOT, 'CodesL2': CodesL2,
        'GPSWeek': GPSWeek, 'L2Pflag': L2Pflag, 'SVacc': SVacc, 'health': health, 'TGD': TGD, 'IODC': IODC,
        'TransTime': TransTime, 'source': source, 't_oc': t_oc
    }

    nav_df = pd.DataFrame(dataframe_data, index=[0])

    return nav_df


def select_nav_ephemeris(nav_ds, sv, date):
    """select an ephemeris from a rinex nav dataset for a particular sv and time

    Input examples:
    nav_ds = load_rnx3_nav_ds(path_to_rnx3_nav_file)
    sv: np.array('G01', dtype='<U3')
    date: np.datetime64('2022-01-01T00:00:00.000')

    Output:
    nav_df: a pandas.dataframe containing the selected ephemeris
    """
    # select ephemeris for right satellite
    nav_prn = nav_ds.sel(sv=sv)
    # find first ephemeris before date of interest
    indexEph = np.searchsorted(nav_prn.time.values, date)
    nav_time = nav_prn.isel(time=indexEph-1)

    # call findsat from gnss_lib_py
    nav_df = convert_nav_single_ds_to_df(nav_time)

    return nav_df

if __name__ == "__main__":

    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001",
                                                                            "BRDC00IGS_R_20220010000_01D_GN.rnx")
    # load RNX3 NAV file
    nav_ds = load_rnx3_nav_ds(path_to_rnx3_nav_file)

    # select sv and time
    sv = np.array('G01', dtype='<U3')
    gps_week = 2190
    gps_tow = 523800
    sv_pos_magnitude = np.array([13053451.235, -12567273.060, 19015357.126])

    # select right ephemeris
    date = np.datetime64(tow_to_datetime(gps_week, gps_tow))
    nav_df = select_nav_ephemeris(nav_ds, sv, date)

    # call findsat from gnss_lib_py
    sv_posvel_rnx3_df = find_sat(nav_df, gps_tow, gps_week)
    sv_pos_rnx3 = np.array([sv_posvel_rnx3_df["x"].values[0],
                            sv_posvel_rnx3_df["y"].values[0],
                            sv_posvel_rnx3_df["z"].values[0]])
    print(sv_pos_rnx3)
    print(f"RNX3 sat pos: {sv_pos_rnx3}")
    print(f"MAGNITUDE sat pos: {sv_pos_magnitude}")

    # filepath towards SP3 file
    path_to_sp3_file = Path(__file__).resolve().parent.parent.joinpath("datasets", "TLSE_2022001", "igs21906.sp3")

    # load sp3
    sp3_ds = gr.load(path_to_sp3_file)

    # convert sp3 time from utc to gps_time
    ls = get_leap_seconds(date.astype(datetime))

    # TODO : cannot import parse_sp3 function from gnss_lib_py/parsers/precise_ephemeris
    # sp3_data = parse_sp3(path_to_sp3_file)

    # retrieve position for specified date and sv
    # # large difference between sp3 and rx3 sat pos, maybe due to different timescales
    # sv_pos_sp3 = sp3_ds.sel(time=date, sv=sv)['position'].values*1000
    # # added leap second, but sel() does not work if the date is not part of the time array
    # sv_pos_sp3 = sp3_ds.sel(time=date + np.timedelta64(ls, 's'), sv=sv)['position'].values*1000
    sp3_sv = sp3_ds.sel(sv=sv)
    sv_pos_sp3 = sp3_sv.interp(time=date + np.timedelta64(ls, 's'))['position'].values*1000

    print(f"SP3 sat pos:  {sv_pos_sp3}")
    print(f"Difference RNX3 - SP3: {sv_pos_rnx3 - sv_pos_sp3}")

    # TODO : large difference may come from the fact that date is a UTC timestamp, while the time in sp3_ds is a gps time