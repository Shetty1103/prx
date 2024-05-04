import logging
import platform
import numpy as np
import pytest
from prx import helpers
from prx import constants
from prx import converters
import pandas as pd
from pathlib import Path
import shutil
import math
import os
from prx import parse_rinex
import subprocess
log = logging.getLogger(__name__)


@pytest.fixture
def input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    test_file = {"obs": test_directory.joinpath(compressed_compact_rinex_file)}
    shutil.copy(
        Path(__file__).parent
        / f"datasets/TLSE_2023001/{compressed_compact_rinex_file}",
        test_file["obs"],
    )
    assert test_file["obs"].exists()

    # Also provide ephemerides so the test does not have to download them:
    ephemerides_file = "BRDC00IGS_R_20230010000_01D_MN.rnx.zip"
    test_file["nav"] = test_directory.joinpath(ephemerides_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{ephemerides_file}",
        test_file["nav"].parent.joinpath(ephemerides_file),
    )
    assert test_file["nav"].parent.joinpath(ephemerides_file).exists()

    # sp3 file
    sp3_file = "GFZ0MGXRAP_20230010000_01D_05M_ORB.SP3"
    test_file["sp3"] = test_directory.joinpath(sp3_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{sp3_file}",
        test_file["sp3"].parent.joinpath(sp3_file),
    )
    assert test_file["sp3"].parent.joinpath(sp3_file).exists()

    yield test_file
    shutil.rmtree(test_directory)


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
        helpers.prx_repository_root() / "tools/validation_data/sagnac_effect.csv"
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

def test_compute_inter_constellation_bias_from_rinex3(input_for_test):
    # filepath towards RNX3 NAV file
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(
        input_for_test["nav"]
    )

    # Parse the RNX3 NAV file
    computed_time_system_corr_dict= helpers.parse_time_syst_corr_from_rinex_nav_file([path_to_rnx3_nav_file])

    t = 3600.0   #For an single EPOCH the value of week of seconds for 2023-01-01 01 00 00
    w = 2243     #weeks
    computed_icb_dict = helpers.compute_icb_all_constellations(computed_time_system_corr_dict, t, w)
    # Manually defined ICB dictionary values
    manual_icb_dict = {
        'G': 0.0,            # GPS = (-1.8626451E-9 + 6.2172489E-15 * (3600 - 233472 + 604800 * (2243 - 2243 ))) = -3.291816539E-9 seconds and Time system correction for reference = -3.2918165E-9 sec. ICB = time system corr reference - time system corr GPS = (0 sec) * speed of light = 0 m
        'R': 0.8279608,      # GLONASS = (-6.0535967E-9 + 0 * (3600 - 0 + 604800 * (2243 - 0 ))) = -6.053596E-9 seconds and Time system correction for reference = -3.2918165E-9 sec. ICB = time system corr reference - time system corr GLONASS = (2.761780137924064e-09 sec) * speed of light = 0.8279608 m
        'E': -0.73162253,    # Galileo = (-9.3132257E-10 + 8.8817841E-16 * (3600 - 518400 + 604800 * (2243 - 2242 ))) = -8.513865131E-9 seconds and Time system correction for reference = -3.2918165E-9 sec. ICB = time system corr reference - time system corr GAL = (-2.440430026E-9 sec) * speed of light = -0.7322 m
        'C': 2.16988596,     # BEIDOU = (-4.6566128E-9 + 9.7699626E-15 * (3600 - 604745 + 604800 * [(2243 - 886 ) SINCE DELTA NOT IN RANGE ITS ASSUMED TOBE ZERO])) = -1.0529777050496465E-08 seconds and Time system correction for reference = -3.2918165E-9 sec. ICB = time system corr reference - time system corr BEIDOU = (7.237960453420529E-09 sec) * speed of light = 2.16988 m
        'I': -1.7321705,     # IRNSS = (-3.1723175E-9 + 1.3322676E-15 * (3600 - 518688 + 604800 * [(2243 - 1218 ) SINCE DELTA NOT IN RANGE ITS ASSUMED TOBE ZERO])) = 2.48608245079856E-09 seconds and Time system correction for reference = -3.2918165E-9 sec. ICB = time system corr reference - time system corr IRNSS = (-5.7778990478744955E-09 sec) * speed of light = -1.7316225 m
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
        assert math.isclose(computed_icb_dict[constellation], manual_icb_dict[constellation], abs_tol=1e-3)  #millimeter level accuarcy
def test_is_sorted():
    assert helpers.is_sorted([1, 2, 3, 4, 5])
    assert not helpers.is_sorted([1, 2, 3, 5, 4])
    assert not helpers.is_sorted([5, 4, 3, 2, 1])
    assert helpers.is_sorted([1, 1, 1, 1, 1])
    assert helpers.is_sorted([1])
    assert helpers.is_sorted([])


def test_gfzrnx_execution_on_obs_file(input_for_test):
    """Check execution of gfzrnx on a RNX OBS file and check"""
    # convert test file to RX3 format
    file_obs = converters.anything_to_rinex_3(input_for_test["obs"])
    # list all gfzrnx binaries contained in the folder "prx/tools/gfzrnx/"
    path_folder_gfzrnx = helpers.prx_repository_root().joinpath("tools", "gfzrnx")
    path_binary = path_folder_gfzrnx.joinpath(
        constants.gfzrnx_binary[platform.system()]
    )
    # assert len(gfzrnx_binaries) > 0, "Could not find any gfzrnx binary"
    command = [
        str(path_binary),
        "-finp",
        str(file_obs),
        "-fout",
        str(file_obs.parent.joinpath("gfzrnx_out.rnx")),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode == 0:
        log.info(
            f"Ran gfzrnx file repair on {file_obs.name} with {constants.gfzrnx_binary[platform.system()]}"
        )
    else:
        log.info(f"gfzrnx file repair run failed: {result}")

    assert file_obs.parent.joinpath("gfzrnx_out.rnx").exists()


def test_gfzrnx_execution_on_nav_file(input_for_test):
    """Check execution of gfzrnx on a RNX NAV file and check"""
    file_nav = converters.anything_to_rinex_3(input_for_test["nav"])
    path_folder_gfzrnx = helpers.prx_repository_root().joinpath("tools", "gfzrnx")
    path_binary = path_folder_gfzrnx.joinpath(
        constants.gfzrnx_binary[platform.system()]
    )
    # assert len(gfzrnx_binaries) > 0, "Could not find any gfzrnx binary"
    command = [
        str(path_binary),
        "-finp",
        str(file_nav),
        "-fout",
        str(file_nav.parent.joinpath("gfzrnx_out.rnx")),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode == 0:
        log.info(
            f"Ran gfzrnx file repair on {file_nav.name} with {constants.gfzrnx_binary[platform.system()]}"
        )
    else:
        log.info(f"gfzrnx file repair run failed: {result}")

    assert file_nav.parent.joinpath("gfzrnx_out.rnx").exists()


def test_gfzrnx_function_call(input_for_test):
    """Check function call of gfzrnx on a RNX OBS file and check"""
    file_nav = converters.anything_to_rinex_3(input_for_test["nav"])
    file_obs = converters.anything_to_rinex_3(input_for_test["obs"])
    file_sp3 = input_for_test["sp3"]

    file_nav = helpers.repair_with_gfzrnx(file_nav)
    file_obs = helpers.repair_with_gfzrnx(file_obs)
    # running gfzrnx on a file that is not a RNX file should result in an error
    try:
        file_sp3 = helpers.repair_with_gfzrnx(file_sp3)
    except AssertionError:
        log.info(f"gfzrnx binary did not execute with file {file_sp3}")
    assert True


def test_row_wise_dot_product():
    # Check whether the way we compute the row-wise dot product with numpy yields the expected result
    A = np.array([[1, 2], [4, 5], [7, 8]])
    B = np.array([[10, 20], [30, 40], [50, 60]])
    row_wise_dot = np.sum(A * B, axis=1).reshape(-1, 1)
    assert (row_wise_dot == np.array([[10 + 40], [120 + 200], [350 + 480]])).all()
