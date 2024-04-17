import shutil
import os
from pathlib import Path
from prx import helpers, converters, parse_rinex
from prx import constants
import pandas as pd
import math
import numpy as np
import pytest


def test_rinex_header_time_string_2_timestamp_ns():
    assert helpers.timestamp_2_timedelta(
        helpers.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     6     0     0    0.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(0)
    assert helpers.timestamp_2_timedelta(
        helpers.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     6     0     0    1.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(constants.cNanoSecondsPerSecond, unit="ns")
    timestamp = helpers.rinex_header_time_string_2_timestamp_ns(
        "  1980     1     6     0     0    1.0000001     GPS"
    )
    timedelta = helpers.timestamp_2_timedelta(timestamp, "GPST")

    assert timedelta == pd.Timedelta(constants.cNanoSecondsPerSecond + 100, unit="ns")
    assert helpers.timestamp_2_timedelta(
        helpers.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     7     0     0    0.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(
        constants.cSecondsPerDay * constants.cNanoSecondsPerSecond, unit="ns"
    )


def test_ecef_to_geodetic():
    tolerance_rad = 1e-3 / 6400e3  # equivalent to one mm at Earth surface
    tolerance_alt = 1e-3

    ecef_coords = [6378137.0, 0.0, 0.0]
    expected_geodetic = [0.0, 0.0, 0.0]
    computed_geodetic = helpers.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [0.0, 6378137.0, 0.0]
    expected_geodetic = [np.deg2rad(0.0), np.deg2rad(90), 0.0]
    computed_geodetic = helpers.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [4624518, 116590, 4376497]  # Toulouse, France
    expected_geodetic = [
        np.deg2rad(43.604698100243851),
        np.deg2rad(1.444193786348353),
        151.9032,
    ]
    computed_geodetic = helpers.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [
        -4.646050004314417e06,
        2.553206120634516e06,
        -3.534374202256767e06,
    ]  # Sidney
    expected_geodetic = [np.deg2rad(-33.8688197), np.deg2rad(151.2092955), 0]
    computed_geodetic = helpers.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [
        1.362205559782862e06,
        -3.423584689115747e06,
        -5.188704112366104e06,
    ]  # Ushuaia, Argentina
    expected_geodetic = [np.deg2rad(-54.8019121), np.deg2rad(-68.3029511), 0]
    computed_geodetic = helpers.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt


def test_satellite_elevation_and_azimuth():
    tolerance = np.deg2rad(1e-3)

    sat_pos_ecef = np.array([[26600e3, 0.0, 0.0]])
    rx_pos_ecef = np.array([6400e3, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(90), np.deg2rad(0)
    computed_el, computed_az = helpers.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array([[2.066169397996826e07, 0.0, 1.428355697996826e07]])
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(0)
    computed_el, computed_az = helpers.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array(
        [[2.066169397996826e07, 7.141778489984130e06, 1.236992320105505e07]]
    )
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(30)
    computed_el, computed_az = helpers.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance


def test_sagnac_effect():
    # load validation data
    path_to_validation_file = (
        helpers.prx_repository_root() / f"tools/validation_data/sagnac_effect.csv"
    )

    # satellite position (from reference CSV header)
    sat_pos = np.array([[28400000, 0, 0]])

    # read data
    data = np.loadtxt(
        path_to_validation_file,
        delimiter=",",
        skiprows=3,
    )

    sagnac_effect_reference = np.zeros((data.shape[0],))
    sagnac_effect_computed = np.zeros((data.shape[0],))
    for ind in range(data.shape[0]):
        rx_pos = data[ind, 0:3]
        sagnac_effect_reference[ind] = data[ind, 3]
        sagnac_effect_computed[ind] = helpers.compute_sagnac_effect(sat_pos, rx_pos)

    # errors come from the approximation of cos and sin for small angles
    # millimeter accuracy should be sufficient
    tolerance = 1e-3
    assert np.max(np.abs(sagnac_effect_computed - sagnac_effect_reference)) < tolerance

@pytest.fixture
def rnx3_input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        # file decompression not working properly
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    rnx3_nav_test_file = test_directory.joinpath("BRDC00IGS_R_20230010000_01D_MN.rnx")
    shutil.copy(
        helpers.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{rnx3_nav_test_file.name}",
        rnx3_nav_test_file,
    )
    assert rnx3_nav_test_file.exists()

    yield {"rnx3_nav_file": rnx3_nav_test_file}
    shutil.rmtree(test_directory)

def test_compute_inter_constellation_bias_from_rinex3(rnx3_input_for_test):
    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(
        rnx3_input_for_test["rnx3_nav_file"]
    )
    # Parse the RNX3 NAV file
    computed_time_system_corr_dict  = helpers.parse_rinex_nav_file(path_to_rnx3_nav_file)
    computed_icb_dict = helpers.compute_icb_all_constellations(computed_time_system_corr_dict)
    # Manually defined ICB dictionary values
    manual_icb_dict = {
        'G': 0.0,
        'R': 1.256415677,
        'E': -0.35531,
        'C': -0.25,
        'I': -1.396,
        'J': np.nan,
        'S': np.nan
    }

    # Ensure that the returned dictionary is not empty
    assert computed_icb_dict

    # Ensure that the returned dictionary contains entries for all constellations
    assert all(constellation in computed_icb_dict for constellation in ['G', 'R', 'E', 'C', 'I', 'J', 'S'])

    # Ensure that each entry in the dictionary matches the corresponding manual value
    for constellation in manual_icb_dict:
        if math.isnan(computed_icb_dict[constellation]) and math.isnan(manual_icb_dict[constellation]):
            continue  # Skip if both values are nan
        assert math.isclose(computed_icb_dict[constellation], manual_icb_dict[constellation], abs_tol=1e-3)
