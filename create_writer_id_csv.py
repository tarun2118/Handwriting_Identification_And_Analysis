import os
import csv
from pathlib import Path

def create_csv():
    base = Path("Dataset")
    writers = ["WRITER_A", "WRITER_B", "WRITER_C", "WRITER_D"]

    rows = []

    for w in writers:
        line_root = base / f"{w}_LINES"
        if not line_root.exists():
            print(f"Folder not found: {line_root}")
            continue
        
        for page_folder in line_root.iterdir():
            if page_folder.is_dir():
                for img in page_folder.glob("*.png"):
                    rows.append([str(img), w])

    with open("writer_id.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "writer_id"])
        writer.writerows(rows)

    print(f"CSV CREATED: {len(rows)} samples")

create_csv()
