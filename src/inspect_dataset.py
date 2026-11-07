
import json
from pathlib import Path

# Location of your downloaded GSM-IC dataset
dataset_folder = Path("GSM-IC")

files = [
    "GSM-IC_2step.json",
    "GSM-IC_mstep.json"
]

for filename in files:

    file_path = dataset_folder / filename

    print("\n" + "=" * 50)
    print("FILE:", filename)
    print("=" * 50)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Data type:", type(data).__name__)

    if isinstance(data, list):
        print("Total records:", len(data))

        if len(data) > 0:
            print("\nFirst record:")
            print(json.dumps(data[0], indent=4))

    elif isinstance(data, dict):
        print("Available keys:", list(data.keys()))

        print("\nFirst record:")
        print(json.dumps(data, indent=4)[:3000])
