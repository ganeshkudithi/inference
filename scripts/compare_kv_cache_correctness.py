
import csv
import json
from pathlib import Path

kv_on_path = Path(
    "results/processed/correctness_details_20261106T120143Z.csv"
)

kv_off_path = Path(
    "results/processed/correctness_details_20261106T121031Z.csv"
)


def load_batch_one(path):
    records = {}

    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["batch_size"]) == 1:
                index = int(row["request_index"])
                records[index] = json.loads(
                    row["candidate_token_ids"]
                )

    return records


kv_on = load_batch_one(kv_on_path)
kv_off = load_batch_one(kv_off_path)

if set(kv_on) != set(kv_off):
    raise ValueError("Request indices do not match.")

matches = 0
total = len(kv_on)

print("\nDirect KV Cache Comparison")
print("-" * 45)

for index in sorted(kv_on):
    exact = kv_on[index] == kv_off[index]
    matches += int(exact)

    print(
        f"Request {index}: "
        f"{'MATCH' if exact else 'MISMATCH'} "
        f"(ON={len(kv_on[index])}, OFF={len(kv_off[index])})"
    )

print("-" * 45)
print(f"Exact matches: {matches}/{total}")
print(f"Exact match rate: {matches / total * 100:.2f}%")

