#!/usr/bin/env python

"""Fetch random OEIS sequences for the experiment dataset."""

import json
import random
import time
from pathlib import Path

import requests

TARGET = 10000


def fetch_oeis_sequence(oeis_id: str) -> dict | None:
    """Fetch a single OEIS sequence from the API."""
    url = f"https://oeis.org/{oeis_id}?fmt=json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data["data"]:
            return None

        return {
            "number": data["number"],
            "data": [int(num) for num in data["data"].split(",")],
            "name": data["name"],
            "comment": data.get("comment", []),
        }

    except requests.RequestException:
        return None


def main() -> None:
    f"""Fetch {TARGET} random OEIS sequences and save to JSONL file."""
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "oeis_sequences.jsonl"
    backup_file = output_dir / "oeis_sequences_backup.jsonl"

    if output_file.exists():
        backup_file.write_text(output_file.read_text())

        sequences = []
        with open(output_file) as f:
            for line in f:
                sequences.append(json.loads(line.strip()))

        print(f"Found {len(sequences)} existing sequences")
    else:
        sequences = []

    fetched_ids = {seq["number"] for seq in sequences}
    successes = 0
    attempts = 0

    print(f"Fetching {TARGET - len(sequences)} random OEIS sequences...")

    start_time = time.time()

    sequences_buffer = []
    while len(sequences) + len(sequences_buffer) < TARGET:
        attempts += 1
        oeis_id = f"A{random.randint(1, 999999):06d}"

        if oeis_id in fetched_ids:
            continue

        fetched_ids.add(oeis_id)

        time.sleep(0.01)

        sequence_data = fetch_oeis_sequence(oeis_id)

        if sequence_data is not None:
            successes += 1
            sequences_buffer.append(sequence_data)

        if attempts % 10 == 0:
            elapsed_time = time.time() - start_time
            rate = successes / elapsed_time if elapsed_time > 0 else -1
            eta = (TARGET - len(sequences)) / rate if rate > 0 else -1
            print(
                f"\rProgress: {len(sequences)}/{TARGET} sequences ({len(sequences) / 100:.1f}%) | "
                f"Attempts: {attempts} | Success rate: {successes / attempts:.1%} | "
                f"Rate: {rate:.1f} seq/s | ETA: {eta / 60:.1f} min      ",
                end="",
            )

        # Save every 100 newly fetched sequences
        if len(sequences_buffer) % 100 == 0:
            with open(output_file, "a") as f:
                for sequence in sequences_buffer:
                    f.write(json.dumps(sequence) + "\n")
            sequences.extend(sequences_buffer)
            sequences_buffer = []

    # Final save - sort sequences by number
    sequences.extend(sequences_buffer)
    sequences.sort(key=lambda seq: seq["number"])
    with open(output_file, "w") as f:
        for sequence in sequences:
            f.write(json.dumps(sequence) + "\n")

    print(f"\nCompleted! Fetched {len(sequences)} sequences in {attempts} attempts")
    print(f"Data saved to: {output_file}")


if __name__ == "__main__":
    main()
