import argparse
import json

from pathlib import Path
import georinex
import pandas as pd
import numpy as np
import git
import joblib
from prx import atmospheric_corrections as atmo
from prx.rinex_nav import nav_file_discovery
from prx import constants, helpers, converters
from prx.rinex_nav import evaluate as rinex_evaluate
memory = joblib.Memory(Path(__file__).parent.joinpath("diskcache"), verbose=0)

log = helpers.get_logger(__name__)


def write_prx_file(
    prx_header: dict,
    prx_records: pd.DataFrame,
    file_name_without_extension: Path,
    output_format: str,
):
    output_writers = {"jsonseq": write_json_text_sequence_file, "csv": write_csv_file}
    if output_format not in output_writers.keys():
        assert (
            False
        ), f"Output format {output_format} not supported,  we can do {list(output_writers.keys())}"
    output_writers[output_format](prx_header, prx_records, file_name_without_extension)


def write_json_text_sequence_file(
    prx_header: dict, prx_records: pd.DataFrame, file_name_without_extension: Path
):
    output_file = Path(
        f"{str(file_name_without_extension)}.{constants.cPrxJsonTextSequenceFileExtension}"
    )
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\u241E" + json.dumps(prx_header, ensure_ascii=False) + "\n")
        drop_columns = [
            "time_of_reception_in_receiver_time",
            "satellite",
            "time_of_emission_in_satellite_time",
        ]
        for epoch in prx_records["time_of_reception_in_receiver_time"].unique():
            epoch = pd.Timestamp(epoch)
            epoch_obs = prx_records[
                prx_records["time_of_reception_in_receiver_time"] == epoch
            ]
            record = {
                "time_of_reception_in_receiver_time": epoch.strftime(
                    "%Y:%m:%dT%H:%M:%S.%f"
                ),
                "satellites": {},
            }
            for idx, row in epoch_obs.iterrows():
                sat = row["satellite"]
                row = row.dropna().to_frame().transpose()
                record["satellites"][sat] = {"observations": {}}
                for col in row.columns:
                    if len(col) == 3:
                        record["satellites"][sat]["observations"][col] = row[
                            col
                        ].values[0]
                        continue
                    if col in drop_columns:
                        continue
                    if type(row[col].values[0]) is np.ndarray:
                        row[col].values[0] = row[col].values[0].tolist()
                    record["satellites"][sat][col] = row[col].values[0]
            file.write("\u241E" + json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Generated JSON Text Sequence prx file: {output_file}")


def write_csv_file(
    prx_header: dict, flat_records: pd.DataFrame, file_name_without_extension: Path
):
    output_file = Path(
        f"{str(file_name_without_extension)}.{constants.cPrxCsvFileExtension}"
    )
    # write header
    with open(output_file, "w", encoding="utf-8") as file:
        for key in prx_header.keys():
            file.write("# %s,%s\n" % (key, prx_header[key]))
    flat_records["sat_elevation_deg"] = np.rad2deg(flat_records.elevation_rad.to_numpy())
    flat_records["sat_azim_deg"] = np.rad2deg(flat_records.azimuth_rad.to_numpy())
    flat_records = flat_records.drop(columns=["elevation_rad", "azimuth_rad"])
    # Re-arrange records to have one  line per code observation, with the associated carrier phase and
    # Doppler observation, and auxiliary information such as satellite position, velocity, clock offset, etc.
    # write records
    # Start with code observations, as they have TGDs, and merge in other observation types one by one
    flat_records["tracking_id"] = flat_records.observation_type.str[1:3]
    records = flat_records.loc[flat_records.observation_type.str.startswith("C")]
    records["C_obs_m"] = records.observation_value
    records = records.drop(columns=["observation_value", "observation_type"])
    type_2_unit = {"D": "mps", "L": "m", "S": "dBHz", "C": "m"}
    for obs_type in ["D", "L", "S"]:
        obs = flat_records.loc[flat_records.observation_type.str.startswith(obs_type)][
            [
                "satellite",
                "time_of_reception_in_receiver_time",
                "observation_value",
                "tracking_id",
            ]
        ]
        obs[f"{obs_type}_obs_{type_2_unit[obs_type]}"] = obs.observation_value
        obs = obs.drop(
            columns=[
                "observation_value",
            ]
        )
        records = records.merge(
            obs,
            on=["satellite", "time_of_reception_in_receiver_time", "tracking_id"],
            how="left",
        )
    records["constellation"] = records.satellite.str[0]
    records["prn"] = records.satellite.str[1:]
    records = records.rename(columns={"tracking_id": "rnx_obs_identifier"})
    records = records.drop(
        columns=[
            "satellite",
            "time_of_emission_in_satellite_time",
            "time_of_emission_isagpst",
            "time_of_emission_weeksecond_system_time",
        ]
    )
    records = records.sort_values(
        by=[
            "time_of_reception_in_receiver_time",
            "constellation",
            "prn",
            "rnx_obs_identifier",
        ]
    )

    records.to_csv(
        path_or_buf=output_file,
        index=False,
        mode="a",
    )
    log.info(f"Generated CSV prx file: {file}")


def build_header(input_files):
    prx_header = {}
    prx_header["input_files"] = [
        {
            "name": file.name,
            "hash": helpers.hash_of_file_content(file, use_sampling=False),
        }
        for file in input_files
    ]
    prx_header["prx_git_commit_id"] = git.Repo(
        search_parent_directories=True
    ).head.object.hexsha
    return prx_header


def check_assumptions(rinex_3_obs_file, rinex_3_nav_file):
    obs_header = georinex.rinexheader(rinex_3_obs_file)
    if "RCV CLOCK OFFS APPL" in obs_header.keys():
        assert (
            obs_header["RCV CLOCK OFFS APPL"].strip() == "0"
        ), "Handling of 'RCV CLOCK OFFS APPL' != 0 not implemented yet."
    assert (
        obs_header["TIME OF FIRST OBS"].split()[-1].strip() == "GPS"
    ), "Handling of observation files using time scales other than GPST not implemented yet."


def build_records(
    rinex_3_obs_file,
    rinex_3_ephemerides_file,
    receiver_ecef_position_m=np.full(shape=(3,), fill_value=np.nan),
):
    return _build_records_cached(
        rinex_3_obs_file,
        helpers.hash_of_file_content(rinex_3_obs_file),
        rinex_3_ephemerides_file,
        helpers.hash_of_file_content(rinex_3_ephemerides_file),
        receiver_ecef_position_m,
    )


# @memory.cache
def _build_records_cached(
    rinex_3_obs_file,
    rinex_3_obs_file_hash,
    rinex_3_ephemerides_file,
    rinex_3_ephemerides_file_hash,
    receiver_ecef_position_m,
):
    check_assumptions(rinex_3_obs_file, rinex_3_ephemerides_file)
    obs = helpers.parse_rinex_obs_file(rinex_3_obs_file)

    # if receiver_ecef_position_m has not been initialized, get it from the RNX OBS header
    obs_header = georinex.rinexheader(rinex_3_obs_file)
    if np.isnan(receiver_ecef_position_m).any():
        receiver_ecef_position_m = np.fromstring(
            obs_header["APPROX POSITION XYZ"], sep=" "
        )

    if "GLONASS SLOT / FRQ #" in obs_header.keys():
        glonass_slot_dict = helpers.build_glonass_slot_dictionary(
            obs_header["GLONASS SLOT / FRQ #"]
        )
    else:
        glonass_slot_dict = None

    # Flatten the xarray DataSet into a pandas DataFrame:
    log.info("Converting Dataset into flat Dataframe of observations")
    flat_obs = pd.DataFrame()
    for obs_label, sat_time_obs_array in obs.data_vars.items():
        df = sat_time_obs_array.to_dataframe(name="obs_value").reset_index()
        df = df[df["obs_value"].notna()]
        df = df.assign(obs_type=lambda x: obs_label)
        flat_obs = pd.concat([flat_obs, df])

    def format_flat_rows(row):
        return [
            pd.Timestamp(row[0]),
            str(row[1]),
            row[2],
            str(row[3]),
        ]

    flat_obs = flat_obs.apply(format_flat_rows, axis=1, result_type="expand")
    flat_obs = flat_obs.rename(
        columns={
            0: "time_of_reception_in_receiver_time",
            1: "satellite",
            2: "observation_value",
            3: "observation_type",
        },
    )

    log.info("Computing times of emission in satellite time")
    per_sat = flat_obs.pivot(
        index=["time_of_reception_in_receiver_time", "satellite"],
        columns=["observation_type"],
        values="observation_value",
    ).reset_index()
    per_sat["time_scale"] = (
        per_sat["satellite"].str[0].map(constants.constellation_2_system_time_scale)
    )
    per_sat["system_time_scale_epoch"] = per_sat["time_scale"].map(
        constants.system_time_scale_2_rinex_utc_epoch
    )
    # When calling georinex.load() with useindicators=True, there are additional ssi columns such as C1Cssi.
    # To exclude them, we check the length of the column name
    code_phase_columns = [c for c in per_sat.columns if c[0] == "C" and len(c) == 3]
    # TODO Find a more self-explanatory name for the following variable
    #
    #  The following term contains, for each satellite
    #  - time-of-flight: around 70 ms for MEO orbits. This includes small
    #  terms (up to tens of meters in units of distance, ten meters correspond to 34 nanoseconds)
    #  such as satellite code bias and atmospheric delays
    #  - receiver clock offset w.r.t. the satellite's constellation time (GPST, GST, BDT etc.)
    #  - satellite clock offset w.r.t. its constellation time
    # By subtracting it from the receiver time of reception, we get the time of emission in
    # the satellite's constellation time frame, plus the satellite system time clock offset plus those small terms
    # of a few tens of nanoseconds.
    tof_dtrx = pd.to_timedelta(
        per_sat[code_phase_columns]
        .mean(axis=1, skipna=True)
        .divide(constants.cGpsSpeedOfLight_mps),
        unit="s",
    )
    per_sat["time_of_emission_in_satellite_time"] = (
        per_sat["time_of_reception_in_receiver_time"]
        - per_sat.system_time_scale_epoch
        - tof_dtrx
    )
    per_sat["time_of_emission_weeksecond_system_time"] = per_sat.apply(
        lambda row: helpers.timedelta_2_weeks_and_seconds(
            row.time_of_emission_in_satellite_time
        )[1],
        axis=1,
    )
    per_sat["time_of_emission_isagpst"] = rinex_evaluate.to_isagpst(
        per_sat.time_of_emission_in_satellite_time,
        per_sat.time_scale,
    )

    flat_obs = flat_obs.merge(
        per_sat[
            [
                "time_of_reception_in_receiver_time",
                "satellite",
                "time_of_emission_in_satellite_time",
                "time_of_emission_isagpst",
                "time_of_emission_weeksecond_system_time",
            ]
        ],
        on=["time_of_reception_in_receiver_time", "satellite"],
    )

    # Compute broadcast position, velocity, clock offset, clock offset rate and TGDs
    query = flat_obs[flat_obs["observation_type"].str.startswith("C")]
    query[["signal", "sv", "query_time_isagpst"]] = query[
        ["observation_type", "satellite", "time_of_emission_isagpst"]
    ]

    sat_states = rinex_evaluate.compute(
        rinex_3_ephemerides_file,
        query,
    )
    sat_states = sat_states.rename(
        columns={
            "sv": "satellite",
            "signal": "observation_type",
            "query_time_isagpst": "time_of_emission_isagpst",
        }
    )
    # We need Timestamps to compute tropo delays
    sat_states = sat_states.merge(
        flat_obs[
            [
                "satellite",
                "time_of_emission_isagpst",
                "time_of_reception_in_receiver_time",
            ]
        ].drop_duplicates(),
        on=["satellite", "time_of_emission_isagpst"],
        how="left",
    )
    # Compute anything else that is satellite-specific
    sat_states[
        "relativistic_clock_effect_m"
    ] = helpers.compute_relativistic_clock_effect(
        sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(),
        sat_states[["sat_vel_x_mps", "sat_vel_y_mps", "sat_vel_z_mps"]].to_numpy(),
    )
    sat_states["sagnac_effect_m"] = helpers.compute_sagnac_effect(
        sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(), receiver_ecef_position_m
    )
    [latitude_user_rad, longitude_user_rad, height_user_m] = helpers.ecef_2_geodetic(
        receiver_ecef_position_m
    )
    days_of_year = np.array(
        sat_states["time_of_reception_in_receiver_time"]
        .apply(lambda element: element.timetuple().tm_yday)
        .to_numpy()
    )
    (
        sat_states["elevation_rad"],
        sat_states["azimuth_rad"],
    ) = helpers.compute_satellite_elevation_and_azimuth(
        sat_states[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].to_numpy(), receiver_ecef_position_m
    )
    (
        tropo_delay_m,
        __,
        __,
        __,
        __,
    ) = atmo.compute_unb3m_correction(
        latitude_user_rad * np.ones(days_of_year.shape),
        height_user_m * np.ones(days_of_year.shape),
        days_of_year,
        sat_states.elevation_rad.to_numpy(),
    )
    sat_states["tropo_delay_m"] = tropo_delay_m
    # Compute time_system_corr_dict somewhere in your code before using it
    sat_states["constellation"] = sat_states["satellite"].str[0]
    time_system_corr_dict = helpers.parse_rinex_nav_file(rinex_3_ephemerides_file)  # Define your dictionary here
    icb_all_constellations = helpers.compute_icb_all_constellations(time_system_corr_dict)
    # Call the compute_icb_all_constellations function
    sat_states['inter_constellation_bias_m'] = sat_states.apply(lambda row: icb_all_constellations.get(row['constellation'], np.nan), axis=1)

    # rows with Doppler and carrier phase observations
    # TODO understand why dropping duplicates with
    #  subset=['satellite', 'time_of_emission_isagpst']
    #  leads to fewer rows here, looks like there are multiple position/velocity/clock values for
    #  the same satellite and the same time of emission
    sat_specific = sat_states[
        sat_states.columns.drop(
            ["observation_type", "sat_instrumental_delay_m", "time_of_reception_in_receiver_time","constellation"]
        )
    ].drop_duplicates(subset=["satellite", "time_of_emission_isagpst"])
    # Group delays are signal-specific, so we merge them in separately
    code_specific = sat_states[
        ["satellite", "observation_type", "time_of_emission_isagpst", "sat_instrumental_delay_m"]
    ].drop_duplicates(
        subset=["satellite", "observation_type", "time_of_emission_isagpst"]
    )
    flat_obs = flat_obs.merge(
        sat_specific, on=["satellite", "time_of_emission_isagpst"], how="left"
    )
    flat_obs = flat_obs.merge(
        code_specific,
        on=["satellite", "observation_type", "time_of_emission_isagpst"],
        how="left",
    )

    flat_obs.loc[
        flat_obs.satellite.str[0] != "R", "carrier_frequency_hz"
    ] = flat_obs.apply(
        lambda row: constants.carrier_frequencies_hz()[row.satellite[0]][
            "L" + row.observation_type[1]
        ],
        axis=1,
    )
    nav_header = georinex.rinexheader(rinex_3_ephemerides_file)
    flat_obs.loc[
        flat_obs.observation_type.str.startswith("C"), "iono_delay_m"
    ] = -atmo.compute_klobuchar_l1_correction(
        flat_obs[
            flat_obs.observation_type.str.startswith("C")
        ].time_of_emission_weeksecond_system_time.to_numpy(),
        nav_header["IONOSPHERIC CORR"]["GPSA"],
        nav_header["IONOSPHERIC CORR"]["GPSB"],
        flat_obs[flat_obs.observation_type.str.startswith("C")].elevation_rad,
        flat_obs[flat_obs.observation_type.str.startswith("C")].azimuth_rad,
        latitude_user_rad,
        longitude_user_rad,
    ) * (
        constants.carrier_frequencies_hz()["G"]["L1"] ** 2
        / flat_obs[flat_obs.observation_type.str.startswith("C")].carrier_frequency_hz
        ** 2
    )
    flat_obs.loc[
        flat_obs.observation_type.str.startswith("L"), "carrier_iono_delay_klobuchar_m"
    ] = atmo.compute_klobuchar_l1_correction(
        flat_obs[
            flat_obs.observation_type.str.startswith("L")
        ].time_of_emission_weeksecond_system_time.to_numpy(),
        nav_header["IONOSPHERIC CORR"]["GPSA"],
        nav_header["IONOSPHERIC CORR"]["GPSB"],
        flat_obs[flat_obs.observation_type.str.startswith("L")].elevation_rad,
        flat_obs[flat_obs.observation_type.str.startswith("L")].azimuth_rad,
        latitude_user_rad,
        longitude_user_rad,
    ) * (
        constants.carrier_frequencies_hz()["G"]["L1"] ** 2
        / flat_obs[flat_obs.observation_type.str.startswith("L")].carrier_frequency_hz
        ** 2
    )

    return flat_obs

def process(observation_file_path: Path, output_format="jsonseq"):
    # We expect a Path, but might get a string here:
    observation_file_path = Path(observation_file_path)
    log.info(
        f"Starting processing {observation_file_path.name} (full path {observation_file_path})"
    )
    rinex_3_obs_file = converters.anything_to_rinex_3(observation_file_path)
    prx_file = str(rinex_3_obs_file).replace(".rnx", "")
    aux_files = nav_file_discovery.discover_or_download_auxiliary_files(
        rinex_3_obs_file
    )

    write_prx_file(
        build_header([rinex_3_obs_file, aux_files["broadcast_ephemerides"]]),
        build_records(rinex_3_obs_file, aux_files["broadcast_ephemerides"]),
        prx_file,
        output_format,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="prx",
        description="prx processes RINEX observations, computes a few useful things such as satellite position, "
        "relativistic effects etc. and outputs everything to a text file in a convenient format.",
        epilog="P.S. GNSS rules!",
    )
    parser.add_argument(
        "--observation_file_path", type=str, help="Observation file path", default=None
    )
    parser.add_argument(
        "--output_format",
        type=str,
        help="Output file format",
        choices=["jsonseq", "csv"],
        default="jsonseq",
    )
    args = parser.parse_args()
    if (
        args.observation_file_path is not None
        and Path(args.observation_file_path).exists()
    ):
        process(Path(args.observation_file_path), args.output_format)
