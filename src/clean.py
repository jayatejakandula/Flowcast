
import calendar
import json
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
QUARANTINE_DIR = Path("data/quarantine")

for folder in [INTERIM_DIR, PROCESSED_DIR, QUARANTINE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    traffic = pd.read_csv(RAW_DIR / "traffic_sensor_log.csv")
    weather = pd.read_csv(RAW_DIR / "weather_observations.csv")
    calendar = pd.read_csv(RAW_DIR / "calendar_events.csv")

    return traffic, weather, calendar


# --------------------------------------------------
# DUPLICATE HANDLING
# --------------------------------------------------

def remove_duplicates(df, key_columns, name):

    duplicates = df[df.duplicated(
        subset=key_columns,
        keep="first"
    )].copy()

    clean_df = df.drop_duplicates(
        subset=key_columns,
        keep="first"
    ).copy()

    duplicates.to_csv(
        QUARANTINE_DIR / f"{name}_duplicates.csv",
        index=False
    )

    print(f"{name}: {len(duplicates)} duplicate rows found")

    return clean_df


# --------------------------------------------------
# TRAFFIC VALIDATION
# --------------------------------------------------

def clean_traffic(traffic):

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

    # Identify invalid readings
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

    print("Invalid traffic rows:", len(rejected))

    # Remove duplicates
    traffic = remove_duplicates(
        traffic,
        ["road_id", "date", "time"],
        "traffic"
    )

    # Fill missing numeric readings using each road's median
    for column in ["traffic_volume", "avg_speed", "occupancy"]:
        traffic[column] = traffic.groupby("road_id")[column].transform(
            lambda values: values.fillna(values.median())
        )

    # Parse vehicle type JSON
    def parse_distribution(value):
        try:
            if pd.isna(value):
                return {}
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}

    vehicle_types = traffic["vehicle_type_dist"].apply(
        parse_distribution
    )

    vehicle_types = pd.json_normalize(vehicle_types).fillna(0)
    vehicle_types.index = traffic.index

    traffic = pd.concat(
        [traffic.drop(columns=["vehicle_type_dist"]), vehicle_types],
        axis=1
    )

    # Derive missing congestion labels
    ratio = traffic["traffic_volume"] / (
        traffic["road_capacity"] / 2
    )

    conditions = [
        ratio < 0.50,
        (ratio >= 0.50) & (ratio < 0.80),
        (ratio >= 0.80) & (ratio < 1.00),
        ratio >= 1.00
    ]

    labels = [
        "Free-flow",
        "Moderate",
        "Heavy",
        "Severe"
    ]

    derived = pd.Series(
        pd.NA,
        index=traffic.index,
        dtype="object"
    )

    missing_congestion = traffic["congestion_level"].isna()

    derived.loc[missing_congestion] = pd.cut(
        ratio[missing_congestion],
        bins=[-float("inf"), 0.50, 0.80, 1.00, float("inf")],
        labels=labels,
        right=False
    ).astype("object")

    traffic.loc[missing_congestion, "congestion_level"] = (
        derived.loc[missing_congestion]
    )

    return traffic


# --------------------------------------------------
# WEATHER CLEANING
# --------------------------------------------------

def clean_weather(weather):

    # Weather date format is DD/MM/YYYY
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

    # Standardise weather labels
    weather["weather_condition"] = (
        weather["weather_condition"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    weather["weather_condition"] = weather[
        "weather_condition"
    ].replace({
        "rainy": "rain",
        "rain": "rain",
        "clear": "clear",
        "cloudy": "cloudy",
        "fog": "fog"
    })

    weather["weather_condition"] = (
        weather["weather_condition"].str.title()
    )

    # Remove invalid date/time records
    invalid = weather["date"].isna() | weather["time"].isna()

    rejected = weather[invalid].copy()

    rejected.to_csv(
        QUARANTINE_DIR / "invalid_weather.csv",
        index=False
    )

    weather = weather[~invalid].copy()

    # Remove duplicates
    weather = remove_duplicates(
        weather,
        ["station_id", "date", "time"],
        "weather"
    )

    # Fill missing weather measurements using station medians
    for column in ["temperature", "visibility"]:
        weather[column] = weather.groupby("station_id")[column].transform(
            lambda values: values.fillna(values.median())
        )

    return weather


# --------------------------------------------------
# CALENDAR CLEANING
# --------------------------------------------------

def clean_calendar(calendar):

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

    calendar = remove_duplicates(
        calendar,
        ["date"],
        "calendar"
    )

    return calendar


# --------------------------------------------------
# MERGE DATASETS
# --------------------------------------------------

def merge_datasets(traffic, weather, calendar):

    # Match each half-hour traffic window to its weather hour
    traffic["weather_time"] = (
        traffic["time"]
        .astype(str)
        .str.slice(0, 2)
        + ":00"
    )

    weather = weather.rename(
        columns={"time": "weather_time"}
    )

    # Merge traffic with hourly weather observations
    merged = pd.merge(
        traffic,
        weather,
        left_on=["weather_station_id", "date", "weather_time"],
        right_on=["station_id", "date", "weather_time"],
        how="left",
        validate="many_to_one"
    )

    # Merge calendar information
    merged = pd.merge(
        merged,
        calendar,
        on="date",
        how="left",
        validate="many_to_one"
    )

    # Remove redundant join columns
    merged = merged.drop(
        columns=["station_id", "weather_time"],
        errors="ignore"
    )

    return merged

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------


def main():

    print("Loading datasets...")

    traffic, weather, calendar = load_data()

    print("\nCleaning traffic...")
    traffic = clean_traffic(traffic)

    print("\nCleaning weather...")
    weather = clean_weather(weather)

    print("\nCleaning calendar...")
    calendar = clean_calendar(calendar)

    # Inspect calendar data
    print("\nCalendar sample:")
    print(calendar.head(10).to_string(index=False))

    print("\nHoliday flag distribution:")
    print(calendar["public_holiday"].value_counts(dropna=False))

    print("\nEvent flag distribution:")
    print(calendar["event_flag"].value_counts(dropna=False))

    print("\nRows with no holiday name:")
    print(calendar[calendar["holiday_name"].isna()].head(5))
    print("\nHolidays missing names:")
    print(
    calendar[
        (calendar["public_holiday"] == 1)
        & (calendar["holiday_name"].isna())
    ]
)

    print("\nEvents missing names:")
    print(
    calendar[
        (calendar["event_flag"] == 1)
        & (calendar["event_name"].isna())
    ]
)

    # Save cleaned individual datasets
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

    print("\nMerging datasets...")

    merged = merge_datasets(
        traffic,
        weather,
        calendar
    )
    

    merged.to_csv(
        PROCESSED_DIR / "flowcast_cleaned.csv",
        index=False
    )

    print("\nCleaning completed!")
    print("Final dataset shape:", merged.shape)

    print("\nMissing values:")
    print(merged.isnull().sum())


if __name__ == "__main__":
    main()



