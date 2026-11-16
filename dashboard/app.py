
import pandas as pd
from pathlib import Path

# Load datasets
traffic = pd.read_csv("data/raw/traffic_sensor_log.csv")
calendar = pd.read_csv("data/raw/calendar_events.csv")
processed = pd.read_csv("data/processed/flowcast_cleaned.csv")

quarantine = Path("data/quarantine")

print("========== ROW ACCOUNTING ==========")

invalid_file = quarantine / "invalid_traffic.csv"
duplicate_file = quarantine / "traffic_duplicates.csv"

invalid_count = len(pd.read_csv(invalid_file))
duplicate_count = len(pd.read_csv(duplicate_file))

print("Original traffic rows:", len(traffic))
print("Invalid rows quarantined:", invalid_count)
print("Duplicate rows quarantined:", duplicate_count)
print("Expected final rows:",
      len(traffic) - invalid_count - duplicate_count)
print("Actual final rows:", len(processed))

print("\nRow accounting matches:",
      len(traffic) - invalid_count - duplicate_count
      == len(processed))

# Calendar alignment
print("\n========== CALENDAR VALIDATION ==========")

traffic_dates = set(
    pd.to_datetime(
        traffic["date"], format="%Y-%m-%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d").dropna()
)

calendar_dates = set(
    pd.to_datetime(
        calendar["date"], format="%Y-%m-%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d").dropna()
)

unmatched_dates = traffic_dates - calendar_dates

print("Traffic dates:", len(traffic_dates))
print("Calendar dates:", len(calendar_dates))
print("Unmatched traffic dates:", len(unmatched_dates))

if unmatched_dates:
    print("Examples:", sorted(unmatched_dates)[:10])

# Processed dataset checks
print("\n========== PROCESSED DATASET ==========")

print("Processed rows:", len(processed))
print("Processed columns:", len(processed.columns))

print("Missing weather conditions:",
      processed["weather_condition"].isna().sum())

print("Missing calendar flags:",
      processed[
          ["public_holiday", "event_flag", "roadwork_flag"]
      ].isna().sum().sum())

print("\nValidation completed.")
