import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Load original stress dataset
data_path = Path("data/current_stress.csv")
df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)

# Split into two equal batches while preserving churn distribution
event_data, scheduled_data = train_test_split(
    df,
    test_size=0.5,
    random_state=42,
    stratify=df["churn"]
)

# Save both datasets
event_data.to_csv("data/stress_event.csv", index=False)
scheduled_data.to_csv("data/stress_scheduled.csv", index=False)

print("\nDataset splitting completed!")
print("Event batch shape:", event_data.shape)
print("Scheduled batch shape:", scheduled_data.shape)

print("\nFiles saved:")
print("data/stress_event.csv")
print("data/stress_scheduled.csv")