import csv

INPUT = "vo_data.csv"
OUTPUT = "vo_data_scaled.csv"
SCALE = 2.0

with open(INPUT, "r", newline="") as fin, open(OUTPUT, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        row["dy"] = str(float(row["dy"]) * SCALE)
        writer.writerow(row)

print(f"Saved {OUTPUT}")