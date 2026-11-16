
# FlowCast — Data Quality Report

## 1. Dataset Overview

- Project: FlowCast
- Dataset: Northline Corridor
- Traffic aggregation: 30 minutes
- Road segments: 25
- Traffic data coverage: January–May 2025

## 2. Cleaning Summary

| Operation | Result |
|---|---:|
| Invalid traffic records quarantined | 712 |
| Duplicate traffic records identified | 1,759 |
| Weather duplicates identified | 0 |
| Calendar duplicates identified | 0 |
| Final processed rows | 175,997 |
| Final processed columns | 29 |

## 3. Statistical Outlier Detection

IQR and Z-score methods were applied separately
to each road segment.

| Column | IQR flags | Z-score flags |
|---|---:|---:|
| traffic_volume | 814 | 620 |
| avg_speed | 4,954 | 1,189 |
| occupancy | 1,137 | 821 |

Statistical outliers are flagged for review and retained.
They are not automatically treated as invalid records.

## 4. Physical Validation Rules

The following readings are treated as invalid:

- Negative traffic volume.
- Negative speed.
- Speed above 200 km/h.
- Occupancy below 0%.
- Occupancy above 100%.

Invalid traffic records are saved in the quarantine folder.

## 5. Missing Values

The processed dataset contains no missing values
in the numerical and operational fields.

Missing holiday_name and event_name values are
expected on dates without holidays or events.

## 6. Data Processing

The pipeline performs:

1. Schema and type handling.
2. Physical validity checks.
3. Duplicate detection.
4. Missing-value handling.
5. Statistical outlier detection.
6. Category normalisation.
7. Date and time harmonisation.
8. Traffic, weather and calendar merging.
9. Congestion-level derivation.
10. Final dataset validation.

## 7. Outlier Handling Policy

Physical invalidity rules determine which records
are quarantined.

IQR and Z-score methods identify statistically
unusual observations. These observations are
retained unless further validation establishes
that they are erroneous.

## 8. Output

Processed dataset:

data/processed/flowcast_cleaned.csv

Quarantined records:

data/quarantine/

Intermediate datasets:

data/interim/
