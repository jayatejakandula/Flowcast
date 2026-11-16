
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


traffic = pd.read_csv(DATA_DIR / "traffic_sensor_log.csv")

weather = pd.read_csv(DATA_DIR / "weather_observations.csv")

calendar = pd.read_csv(DATA_DIR / "calendar_events.csv")


def inspect_dataset(name, df):
    print(f"\n{'=' * 50}")
    print(f"{name} DATASET")
    print(f"{'=' * 50}")

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDescriptive statistics:")
    print(df.describe(include="all"))


inspect_dataset("TRAFFIC", traffic)
inspect_dataset("WEATHER", weather)
inspect_dataset("CALENDAR", calendar)


invalid_speed = traffic[
    (traffic["avg_speed"] < 0) |
    (traffic["avg_speed"] > 200)
]

invalid_volume = traffic[
    traffic["traffic_volume"] < 0
]

invalid_occupancy = traffic[
    (traffic["occupancy"] < 0) |
    (traffic["occupancy"] > 100)
]



print("\nINVALID SPEED RECORDS:")
print(invalid_speed)

print("\nINVALID TRAFFIC VOLUME RECORDS:")
print(invalid_volume)

print("\nINVALID OCCUPANCY RECORDS:")
print(invalid_occupancy)

print("\nINGESTION COMPLETED.")
