import os

folders = [
    "gait_project/data/normal",
    "gait_project/data/abnormal",
    "gait_project/csv_files"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Folder created: {folder}")
