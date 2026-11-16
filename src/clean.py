
import json
from pathlib import Path
import pandas as pd


# ==================================================
# PATHS AND CONFIGURATION
# ==================================================

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
QUARANTINE_DIR = Path("data/quarantine")
REPORT_DIR = Path("reports")

for folder in [
    INTERIM_DIR,
    PROCESSED_DIR,
    QUARANTINE_DIR,
    REPORT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


TRAFFIC_KEYS = ["road_id", "date", "time"]
WEATHER_KEYS = ["station_id", "date", "time"]

TRAFFIC_COLUMNS = [
    "traffic_volume",
    "avg_speed",
    "occupancy"
]

PHYSICAL_LIMITS = {
    "traffic_volume": (0, float("inf")),
    "avg_speed": (0, 200),
    "occupancy": (0, 100)
}


# ==================================================
# LOAD AND SCHEMA VALIDATION
# ==================================================

def load_data():

    traffic = pd.read_csv(
        RAW_DIR / "traffic_sensor_log.csv"
    )

    weather = pd.read_csv(
        RAW_DIR / "weather_observations.csv"
    )

    calendar = pd.read_csv(
        RAW_DIR / "calendar_events.csv"
    )

    required_columns = {
        "traffic": [
            "road_id", "date", "time",
            "traffic_volume", "avg_speed",
            "occupancy", "road_capacity",
            "weather_station_id", "vehicle_type_dist",
            "congestion_level"
        ],
        "weather": [
            "station_id", "date", "time",
            "weather_condition", "temperature",
            "visibility"
        ],
        "calendar": [
            "date", "public_holiday",
            "holiday_name", "event_flag",
            "event_name", "roadwork_flag"
        ]
    }

    datasets = {
        "traffic": traffic,
        "weather": weather,
        "calendar": calendar
    }

    for name, required in required_columns.items():

        missing = set(required) - set(datasets[name].columns)

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {sorted(missing)}"
            )

    print("All required dataset columns are present.")

    return traffic, weather, calendar


# ==================================================
# DUPLICATE HANDLING
# ==================================================

def remove_duplicates(df, key_columns, name):

    duplicates = df[
        df.duplicated(
            subset=key_columns,
            keep="first"
        )
    ].copy()

    clean_df = df.drop_duplicates(
        subset=key_columns,
        keep="first"
    ).copy()

    duplicates.to_csv(
        QUARANTINE_DIR / f"{name}_duplicates.csv",
        index=False
    )

    print(
        f"{name}: {len(duplicates)} duplicate rows found"
    )

    return clean_df, len(duplicates)


# ==================================================
# STATISTICAL OUTLIER DETECTION
# ==================================================

def detect_statistical_outliers(traffic):

    print("\nRunning statistical outlier detection...")

    flags = pd.DataFrame(index=traffic.index)

    results = {
        "iqr": {},
        "zscore": {},
        "unique_iqr_rows": 0,
        "unique_zscore_rows": 0
    }

    for column in TRAFFIC_COLUMNS:

        values = traffic[column]

        iqr_flag = pd.Series(
            False,
            index=traffic.index
        )

        zscore_flag = pd.Series(
            False,
            index=traffic.index
        )

        # Calculate separately for each road
        for road_id, group in traffic.groupby("road_id"):

            road_values = group[column].dropna()

            if len(road_values) < 2:
                continue

            # IQR calculation
            q1 = road_values.quantile(0.25)
            q3 = road_values.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            road_iqr_flags = (
                (road_values < lower)
                | (road_values > upper)
            )

            iqr_flag.loc[road_values.index] = road_iqr_flags

            # Z-score calculation
            mean = road_values.mean()
            std = road_values.std()

            if pd.notna(std) and std > 0:

                z_scores = (
                    road_values - mean
                ) / std

                road_zscore_flags = z_scores.abs() > 3

                zscore_flag.loc[road_values.index] = (
                    road_zscore_flags
                )

        flags[f"{column}_iqr_flag"] = iqr_flag
        flags[f"{column}_zscore_flag"] = zscore_flag

        results["iqr"][column] = int(iqr_flag.sum())
        results["zscore"][column] = int(zscore_flag.sum())

        print(
            f"{column}: "
            f"IQR={iqr_flag.sum()}, "
            f"Z-score={zscore_flag.sum()}"
        )

    results["unique_iqr_rows"] = int(
        flags[
            [f"{c}_iqr_flag" for c in TRAFFIC_COLUMNS]
        ].any(axis=1).sum()
    )

    results["unique_zscore_rows"] = int(
        flags[
            [f"{c}_zscore_flag" for c in TRAFFIC_COLUMNS]
        ].any(axis=1).sum()
    )

    # Save flagged records separately for review.
    # These records are NOT quarantined or deleted.
    flagged = flags.any(axis=1)

    if flagged.any():

        flagged_records = pd.concat(
            [
                traffic.loc[flagged],
                flags.loc[flagged]
            ],
            axis=1
        )

        flagged_records.to_csv(
            REPORT_DIR / "traffic_statistical_outliers.csv",
            index=False
        )

    else:
        pd.DataFrame().to_csv(
            REPORT_DIR / "traffic_statistical_outliers.csv",
            index=False
        )

    print(
        "Unique IQR rows:",
        results["unique_iqr_rows"]
    )

    print(
        "Unique Z-score rows:",
        results["unique_zscore_rows"]
    )

    return results


# ==================================================
# TRAFFIC CLEANING
# ==================================================

def clean_traffic(traffic):

    original_count = len(traffic)

    # Convert numeric columns safely
    for column in TRAFFIC_COLUMNS:
        traffic[column] = pd.to_numeric(
            traffic[column],
            errors="coerce"
        )

    # Convert date and time
    traffic["date"] = pd.to_datetime(
        traffic["date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    traffic["time"] = pd.to_datetime(
        traffic["time"],
        format="%H:%M",
        errors="coerce"
    ).dt.time

    # Physical validity checks
    invalid = (
        (traffic["traffic_volume"] < 0)
        | (traffic["avg_speed"] < 0)
        | (traffic["avg_speed"] > 200)
        | (traffic["occupancy"] < 0)
        | (traffic["occupancy"] > 100)
        | traffic["date"].isna()
        | traffic["time"].isna()
    )

    rejected = traffic[invalid].copy()

    traffic = traffic[~invalid].copy()

    rejected.to_csv(
        QUARANTINE_DIR / "invalid_traffic.csv",
        index=False
    )

    invalid_count = len(rejected)

    print("Invalid traffic rows:", invalid_count)

    # Statistical checks run before duplicate removal
    # and missing-value imputation.
    outlier_results = detect_statistical_outliers(
        traffic
    )

    # Remove duplicates
    traffic, duplicate_count = remove_duplicates(
        traffic,
        TRAFFIC_KEYS,
        "traffic"
    )

    # Fill missing numeric readings using road medians
    for column in TRAFFIC_COLUMNS:

        traffic[column] = traffic.groupby(
            "road_id"
        )[column].transform(
            lambda values: values.fillna(values.median())
        )

    # Parse vehicle type JSON
    malformed_json_count = 0

    def parse_distribution(value):

        nonlocal malformed_json_count

        try:

            if pd.isna(value):
                return {}

            parsed = json.loads(value)

            if not isinstance(parsed, dict):
                malformed_json_count += 1
                return {}

            return parsed

        except (json.JSONDecodeError, TypeError):
            malformed_json_count += 1
            return {}

    vehicle_types = traffic[
        "vehicle_type_dist"
    ].apply(parse_distribution)

    vehicle_types = pd.json_normalize(
        vehicle_types
    ).fillna(0)

    vehicle_types.index = traffic.index

    traffic = pd.concat(
        [
            traffic.drop(
                columns=["vehicle_type_dist"]
            ),
            vehicle_types
        ],
        axis=1
    )

    # Derive missing congestion labels
    ratio = traffic["traffic_volume"] / (
        traffic["road_capacity"] / 2
    )

    labels = [
        "Free-flow",
        "Moderate",
        "Heavy",
        "Severe"
    ]

    missing_congestion = (
        traffic["congestion_level"].isna()
    )

    derived = pd.cut(
        ratio[missing_congestion],
        bins=[
            -float("inf"),
            0.50,
            0.80,
            1.00,
            float("inf")
        ],
        labels=labels,
        right=False
    ).astype("object")

    traffic.loc[
        missing_congestion,
        "congestion_level"
    ] = derived

    results = {
        "original": original_count,
        "invalid": invalid_count,
        "duplicates": duplicate_count,
        "outliers": outlier_results,
        "malformed_json": malformed_json_count
    }

    return traffic, results


# ==================================================
# WEATHER CLEANING
# ==================================================

def clean_weather(weather):

    original_count = len(weather)

    # Weather date format: DD/MM/YYYY
    weather["date"] = pd.to_datetime(
        weather["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    weather["time"] = pd.to_datetime(
        weather["time"],
        format="%H:%M",
        errors="coerce"
    ).dt.strftime("%H:%M")

    # Normalise weather labels
    weather["weather_condition"] = (
        weather["weather_condition"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    weather["weather_condition"] = (
        weather["weather_condition"].replace({
            "rainy": "rain",
            "rain": "rain",
            "clear": "clear",
            "cloudy": "cloudy",
            "fog": "fog"
        })
    )

    weather["weather_condition"] = (
        weather["weather_condition"].str.title()
    )

    # Invalid date/time records
    invalid = (
        weather["date"].isna()
        | weather["time"].isna()
    )

    rejected = weather[invalid].copy()

    rejected.to_csv(
        QUARANTINE_DIR / "invalid_weather.csv",
        index=False
    )

    weather = weather[~invalid].copy()

    invalid_count = len(rejected)

    # Remove duplicates
    weather, duplicate_count = remove_duplicates(
        weather,
        WEATHER_KEYS,
        "weather"
    )

    # Fill missing measurements by station median
    for column in ["temperature", "visibility"]:

        weather[column] = weather.groupby(
            "station_id"
        )[column].transform(
            lambda values: values.fillna(values.median())
        )

    results = {
        "original": original_count,
        "invalid": invalid_count,
        "duplicates": duplicate_count
    }

    print("Invalid weather rows:", invalid_count)

    return weather, results


# ==================================================
# CALENDAR CLEANING
# ==================================================

def clean_calendar(calendar):

    original_count = len(calendar)

    calendar["date"] = pd.to_datetime(
        calendar["date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    invalid = calendar["date"].isna()

    rejected = calendar[invalid].copy()

    rejected.to_csv(
        QUARANTINE_DIR / "invalid_calendar.csv",
        index=False
    )

    calendar = calendar[~invalid].copy()

    invalid_count = len(rejected)

    calendar, duplicate_count = remove_duplicates(
        calendar,
        ["date"],
        "calendar"
    )

    results = {
        "original": original_count,
        "invalid": invalid_count,
        "duplicates": duplicate_count
    }

    print("Invalid calendar rows:", invalid_count)

    return calendar, results


# ==================================================
# DATASET MERGING AND VALIDATION
# ==================================================

def merge_datasets(traffic, weather, calendar):

    # Convert half-hour traffic time to weather hour
    traffic["weather_time"] = (
        traffic["time"]
        .astype(str)
        .str.slice(0, 2)
        + ":00"
    )

    weather = weather.rename(
        columns={"time": "weather_time"}
    )

    # Traffic-weather merge
    merged = pd.merge(
        traffic,
        weather,
        left_on=[
            "weather_station_id",
            "date",
            "weather_time"
        ],
        right_on=[
            "station_id",
            "date",
            "weather_time"
        ],
        how="left",
        validate="many_to_one",
        indicator="_weather_merge"
    )

    weather_matched = int(
        (merged["_weather_merge"] == "both").sum()
    )

    weather_unmatched = int(
        (merged["_weather_merge"] == "left_only").sum()
    )

    print("\nWeather merge validation:")
    print("Matched:", weather_matched)
    print("Unmatched:", weather_unmatched)

    merged = merged.drop(
        columns=["station_id", "_weather_merge"],
        errors="ignore"
    )

    # Calendar merge
    merged = pd.merge(
        merged,
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
        indicator="_calendar_merge"
    )

    calendar_matched = int(
        (merged["_calendar_merge"] == "both").sum()
    )

    calendar_unmatched = int(
        (merged["_calendar_merge"] == "left_only").sum()
    )

    print("\nCalendar merge validation:")
    print("Matched:", calendar_matched)
    print("Unmatched:", calendar_unmatched)

    merged = merged.drop(
        columns=[
            "weather_time",
            "_calendar_merge"
        ],
        errors="ignore"
    )

    results = {
        "weather_matched": weather_matched,
        "weather_unmatched": weather_unmatched,
        "calendar_matched": calendar_matched,
        "calendar_unmatched": calendar_unmatched
    }

    return merged, results


# ==================================================
# FINAL DATA VALIDATION
# ==================================================

def validate_final_dataset(
    merged,
    traffic_results,
    weather_results,
    calendar_results,
    merge_results
):

    print("\n========== FINAL DATA VALIDATION ==========")

    # Row accounting
    expected_rows = (
        traffic_results["original"]
        - traffic_results["invalid"]
        - traffic_results["duplicates"]
    )

    actual_rows = len(merged)

    accounting_passed = (
        expected_rows == actual_rows
    )

    print("Expected rows:", expected_rows)
    print("Actual rows:", actual_rows)
    print("Row accounting passed:", accounting_passed)

    # Merge validation
    weather_passed = (
        merge_results["weather_unmatched"] == 0
    )

    calendar_passed = (
        merge_results["calendar_unmatched"] == 0
    )

    print("Weather merge passed:", weather_passed)
    print("Calendar merge passed:", calendar_passed)

    # Missing values
    missing_values = merged.isnull().sum()

    print("\nMissing values:")
    print(missing_values)

    # Holiday/event names are expected to be missing
    # on ordinary dates without those events.
    unexpected_missing = missing_values.drop(
        labels=["holiday_name", "event_name"],
        errors="ignore"
    )

    missing_data_passed = (
        unexpected_missing.sum() == 0
    )

    print(
        "\nUnexpected missing values:",
        int(unexpected_missing.sum())
    )

    # Save processed dataset
    merged.to_csv(
        PROCESSED_DIR / "flowcast_cleaned.csv",
        index=False
    )

    results = {
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "accounting_passed": accounting_passed,
        "weather_passed": weather_passed,
        "calendar_passed": calendar_passed,
        "missing_data_passed": missing_data_passed,
        "missing_values": missing_values
    }

    return results


# ==================================================
# DATA QUALITY REPORT
# ==================================================

def generate_quality_report(
    traffic_results,
    weather_results,
    calendar_results,
    merge_results,
    final_results
):

    outliers = traffic_results["outliers"]

    report = f"""# FlowCast — Data Quality Report

## 1. Dataset Overview

- Project: FlowCast
- Dataset: Northline Corridor
- Traffic aggregation: 30 minutes
- Road segments: 25
- Data coverage: January–May 2025

## 2. Traffic Cleaning

| Operation | Records |
|---|---:|
| Original traffic records | {traffic_results['original']} |
| Invalid records quarantined | {traffic_results['invalid']} |
| Duplicate records quarantined | {traffic_results['duplicates']} |
| Final processed records | {final_results['actual_rows']} |

## 3. Weather Cleaning

| Operation | Records |
|---|---:|
| Original weather records | {weather_results['original']} |
| Invalid records quarantined | {weather_results['invalid']} |
| Duplicate records quarantined | {weather_results['duplicates']} |

## 4. Calendar Cleaning

| Operation | Records |
|---|---:|
| Original calendar records | {calendar_results['original']} |
| Invalid records quarantined | {calendar_results['invalid']} |
| Duplicate records quarantined | {calendar_results['duplicates']} |

## 5. Statistical Outlier Detection

IQR and Z-score analysis was performed separately
for each road segment, using physically valid
traffic readings before duplicate removal and imputation.

| Column | IQR flags | Z-score flags |
|---|---:|---:|
| traffic_volume | {outliers['iqr']['traffic_volume']} | {outliers['zscore']['traffic_volume']} |
| avg_speed | {outliers['iqr']['avg_speed']} | {outliers['zscore']['avg_speed']} |
| occupancy | {outliers['iqr']['occupancy']} | {outliers['zscore']['occupancy']} |

Unique rows flagged by IQR: {outliers['unique_iqr_rows']}

Unique rows flagged by Z-score: {outliers['unique_zscore_rows']}

Statistical outliers are retained for review.
They are not automatically quarantined or deleted.

## 6. Physical Validation Rules

The following readings are considered invalid:

- Negative traffic volume.
- Negative speed.
- Speed above 200 km/h.
- Occupancy below 0%.
- Occupancy above 100%.

Invalid traffic records are saved in the quarantine directory.

## 7. Merge Validation

| Validation | Matched | Unmatched |
|---|---:|---:|
| Traffic-weather | {merge_results['weather_matched']} | {merge_results['weather_unmatched']} |
| Traffic-calendar | {merge_results['calendar_matched']} | {merge_results['calendar_unmatched']} |

## 8. Row Accounting

Expected processed rows: {final_results['expected_rows']}

Actual processed rows: {final_results['actual_rows']}

Row accounting passed: {final_results['accounting_passed']}

## 9. Missing Values

{final_results['missing_values'].to_string()}

Missing holiday_name and event_name values are expected
on dates without holidays or events.

## 10. Additional Processing

- Date and time standardisation.
- Weather category normalisation.
- Missing numeric value imputation using group medians.
- Vehicle type JSON parsing.
- Missing congestion-level derivation.
- Final dataset validation.

Malformed vehicle-type JSON records encountered:
{traffic_results['malformed_json']}

## 11. Validation Status

| Check | Result |
|---|---|
| Row accounting | {final_results['accounting_passed']} |
| Weather merge | {final_results['weather_passed']} |
| Calendar merge | {final_results['calendar_passed']} |
| Unexpected missing values | {final_results['missing_data_passed']} |

## 12. Output Files

- data/processed/flowcast_cleaned.csv
- data/interim/traffic_cleaned.csv
- data/interim/weather_cleaned.csv
- data/interim/calendar_cleaned.csv
- data/quarantine/
- reports/traffic_statistical_outliers.csv
"""

    report_path = REPORT_DIR / "data_quality.md"

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    print("\nData quality report generated:")
    print(report_path)


# ==================================================
# MAIN PIPELINE
# ==================================================

def main():

    print("=" * 55)
    print("FLOWCAST DATA CLEANING PIPELINE")
    print("=" * 55)

    print("\n1. Loading datasets...")

    traffic, weather, calendar = load_data()

    # Traffic
    print("\n2. Cleaning traffic...")

    traffic, traffic_results = clean_traffic(
        traffic
    )

    # Weather
    print("\n3. Cleaning weather...")

    weather, weather_results = clean_weather(
        weather
    )

    # Calendar
    print("\n4. Cleaning calendar...")

    calendar, calendar_results = clean_calendar(
        calendar
    )

    # Save interim datasets
    traffic.to_csv(
        INTERIM_DIR / "traffic_cleaned.csv",
        index=False
    )

    weather.to_csv(
        INTERIM_DIR / "weather_cleaned.csv",
        index=False
    )

    calendar.to_csv(
        INTERIM_DIR / "calendar_cleaned.csv",
        index=False
    )

    # Merge
    print("\n5. Merging datasets...")

    merged, merge_results = merge_datasets(
        traffic,
        weather,
        calendar
    )

    # Final validation
    print("\n6. Validating final dataset...")

    final_results = validate_final_dataset(
        merged,
        traffic_results,
        weather_results,
        calendar_results,
        merge_results
    )

    # Report
    print("\n7. Generating quality report...")

    generate_quality_report(
        traffic_results,
        weather_results,
        calendar_results,
        merge_results,
        final_results
    )

    print("\n" + "=" * 55)
    print("FLOWCAST CLEANING PIPELINE COMPLETED")
    print("=" * 55)

    print("Final dataset shape:", merged.shape)

    if not all([
        final_results["accounting_passed"],
        final_results["weather_passed"],
        final_results["calendar_passed"],
        final_results["missing_data_passed"]
    ]):

        print("\nWARNING: One or more validation checks failed.")
        print("Review the data quality report.")

    else:
        print("\nAll final validation checks passed.")


if __name__ == "__main__":
    main()
