#!/usr/bin/env python


"""Run LLM experiments on OEIS integer sequence prediction."""

import itertools
import json
import secrets
import subprocess
import tempfile
import textwrap
from pathlib import Path

import litellm
import pandas as pd

# =============================================================================
# PARAMETERS
# =============================================================================


VERBOSE: int = 2

TIMEOUT: dict = {
    "execution": 30,
    "call": 10,
}

TEMPERATURE: float = 0.0
TOP_K: int = 1

# Number of sequences to use
NUM_SEQUENCES: int = 10

PARAMETER_SPACE: dict[str, list] = {
    "model_name": [
        "gpt-4o-mini",
        "o4-mini",
        "gpt-4.1",
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
        # "claude-opus-4-20250514", # expensive
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-2.5-flash",
        "xai/grok-4",
    ],
    "capability": [
        "single_guess",  # Only output guess
        "compute",  # Output Python script, run it, then guess
    ],
    "include_description": [True, False],
    "sequence_start": [0],
    "sequence_length": [8],
    "experiment_id": [pd.NA],
    "timestamp": [pd.NaT],
}

PARAMETER_SPACE_DTYPE: dict = {
    "model_name": pd.CategoricalDtype(categories=PARAMETER_SPACE["model_name"], ordered=False),
    "capability": pd.CategoricalDtype(categories=PARAMETER_SPACE["capability"], ordered=False),
    "include_description": "boolean",
    "sequence_start": "Int64",
    "sequence_length": "Int64",
    "experiment_id": "Int64",
    "timestamp": "datetime64[ns]",
}

# File paths
OEIS_DATA_PATH: Path = Path("data/oeis_sequences.jsonl")
EXPERIMENTS_DF_PATH: Path = Path("data/experiments.parquet")
LOGS_DIR: Path = Path("data/experiment_logs")

OUTCOME_SPACE: dict[str, list] = {
    "guess": [pd.NA],
    "expected": [pd.NA],
    "is_correct": [pd.NA],
    "absolute_error": [pd.NA],
    "cost": [pd.NA],
}

OUTCOME_SPACE_DTYPE: dict[str, str] = {
    "guess": "Int64",
    "expected": "Int64",
    "is_correct": "boolean",
    "absolute_error": "Int64",
    "cost": "Float64",
}

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DEFINITIONS
# =============================================================================


def execute_code(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_path = Path(tmp_file.name)

    try:
        result = subprocess.run(
            ["python", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT["execution"],
        )
        if result.returncode != 0:
            raise RuntimeError(f"Code execution failed with error:\n{result.stderr}")
        return result.stdout
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Code execution timed out after {TIMEOUT['execution']} seconds.")
    finally:
        if tmp_path.is_file():
            tmp_path.unlink()


def estimate_experiment_costs(existing_df: pd.DataFrame, new_df: pd.DataFrame):
    """Estimate and print the total cost of running new experiments based on existing data."""

    # Calculate average cost per (model_name, capability) combination
    avg_costs = existing_df.groupby(["model_name", "capability"], observed=False)["cost"].mean()

    # Create pivot table with models as rows and capabilities as columns
    cost_table = avg_costs.unstack(fill_value=0)

    print("\nAverage costs in cents per 100 experiments:")
    print(cost_table.to_string(float_format=lambda x: f"{100*100*x:.2f}"))

    # Calculate estimated total cost for new experiments
    total_estimated_cost = 0.0
    experiments_with_estimates = 0

    for _, exp in new_df.iterrows():
        key = (exp["model_name"], exp["capability"])
        if key in avg_costs and not pd.isna(avg_costs.at[key]):
            total_estimated_cost += avg_costs.at[key]
            experiments_with_estimates += 1

    experiments_without_estimates = len(new_df) - experiments_with_estimates

    # scale total_estimated_cost by fraction of experiments with estimate

    if experiments_with_estimates > 0:
        total_estimated_cost /= experiments_with_estimates / len(new_df)

    print("\nCost estimation summary:")
    print(f"  Experiments with cost estimates: {experiments_with_estimates}")
    print(f"  Experiments without cost estimates: {experiments_without_estimates}")
    print(f"  Estimated total cost: ${total_estimated_cost:.4f}")


def guess_next_integer(
    config: pd.Series, sequence_info: dict
) -> tuple[tuple[int, int] | None, dict]:
    """Ask an LLM for the next integer in the sequence."""

    messages = []

    cost = 0
    log_data = {
        "messages": [],
        "error": "",
    }

    def query_llm(prompt: str) -> str | None:
        nonlocal cost
        messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=config["model_name"],
                messages=messages,
                # temperature=TEMPERATURE,
                # top_k=TOP_K,
                timeout=TIMEOUT["call"],
            )
        except Exception as e:
            log_data["error"] = str(e)
            return None

        content = response.choices[0].message.content
        cost += response._hidden_params["response_cost"]
        messages.append({"role": "assistant", "content": content})
        if VERBOSE >= 2:
            print("  Prompt:")
            print(textwrap.indent(prompt, prefix="  | "))
            print("  Response:")
            print(textwrap.indent(content, prefix="  | "))
        return content

    sequence_str = ", ".join(str(s) for s in sequence_info["partial"])
    prompt = f"Given the integer sequence: {sequence_str}\n"
    if bool(config["include_description"]):
        prompt += f"Described/Defined as: {sequence_info['name']}\n"
    prompt += "What is the next integer in the sequence?\n"

    if config["capability"] == "compute":
        prompt += (
            "First, write a Python script. Output only the Python code, nothing else. "
            "Your next message will be directly executed as python code with exec(). "
            "DO NOT wrap it with markdown backticks (```python ```). NEVER. "
            "DO NOT add any comments (# a comment). NEVER. "
            f"Your script should execute in under {TIMEOUT['execution']} seconds or it will fail. "
            "I will then share the stdout in my next message."
        )

        code_response = query_llm(prompt)
        if code_response is None:
            return None, log_data

        log_data["messages"].extend([prompt, code_response])

        try:
            prompt = execute_code(code_response)
        except Exception as e:
            log_data["error"] = str(e)
            return None, log_data

    prompt += (
        "Output only the integer, nothing else. "
        "Your next message will be directly converted in python with int()."
    )

    response = query_llm(prompt)
    if response is None:
        return None, log_data

    log_data["messages"] = [prompt, response]

    try:
        guess = int(response)
    except Exception as e:
        error = f"Error parsing model guess: {str(e)}"
        log_data["error"] = error
        return None, log_data

    log_data["cost"] = cost

    return (guess, cost), log_data


def main():
    """Run the experiment suite."""

    # Load existing experiments file
    if EXPERIMENTS_DF_PATH.exists():
        existing_df = pd.read_parquet(EXPERIMENTS_DF_PATH)

        # Extend model name category type
        PARAMETER_SPACE_DTYPE["model_name"] = pd.CategoricalDtype(
            categories=existing_df["model_name"].cat.categories.union(
                PARAMETER_SPACE_DTYPE["model_name"].categories
            )
        )
        existing_df["model_name"] = existing_df["model_name"].cat.set_categories(
            PARAMETER_SPACE_DTYPE["model_name"].categories, ordered=False
        )
    else:
        existing_df = None

    print("Loading OEIS data into parameters...")
    with open(OEIS_DATA_PATH, "r") as f:
        oeis_data = [json.loads(line) for line in f]
        oeis_data = {seq["number"]: seq for seq in oeis_data}

    PARAMETER_SPACE["oeis_id"] = list(oeis_data.keys())[:NUM_SEQUENCES]
    PARAMETER_SPACE_DTYPE["oeis_id"] = "int64"

    # Generate experiment configurations
    full_space = {**PARAMETER_SPACE, **OUTCOME_SPACE}
    full_space_dtype = {**PARAMETER_SPACE_DTYPE, **OUTCOME_SPACE_DTYPE}

    col_names, values = zip(*full_space.items())
    new_df = pd.DataFrame(
        {
            col_names[i]: pd.Series(col, dtype=full_space_dtype[col_names[i]])
            for i, col in enumerate(zip(*itertools.product(*values)))
        }
    )

    if existing_df is not None:
        # Join the existing df with the new df
        param_names = list(PARAMETER_SPACE.keys())
        new_df = pd.merge(
            new_df, existing_df[param_names], on=param_names, how="left", indicator=True
        )
        # Keep only the experiments that have not been run before
        new_df = new_df[new_df["_merge"] == "left_only"].drop("_merge", axis=1)

    new_df = new_df.sort_values(
        by=["oeis_id", "include_description", "capability", "model_name"]
    ).reset_index(drop=True)

    print(f"Total experiments to run: {len(new_df)}")

    # Estimate costs if we have existing data
    if existing_df is not None:
        estimate_experiment_costs(existing_df, new_df)

    if input("Proceed? (y/N): ").lower() != "y":
        print("Aborting.")
        return

    completed = []
    for i, exp in new_df.iterrows():
        if VERBOSE >= 1:
            print(f"\nRunning experiment {i}/{len(new_df)}")
            print(f"  OEIS: {exp['oeis_id']}")
            print(f"  Model: {exp['model_name']}")
            print(f"  Capability: {exp['capability']}")
            print(f"  Description: {exp['include_description']}")

        sequence_info = oeis_data[exp["oeis_id"]]

        sequence_full = sequence_info["data"]

        # Extract the partial sequence to guess from
        start = exp["sequence_start"]
        length = min(exp.at["sequence_length"], len(sequence_full) - 1)
        sequence_info["partial"] = sequence_full[start : start + length]
        sequence_info["expected"] = sequence_full[start + length]

        out, log_data = guess_next_integer(exp, sequence_info)

        timestamp = pd.Timestamp.now()
        exp_id = secrets.randbits(63)

        if out is None:
            print(f"  {log_data['error']}")
            log_name = f"{timestamp}-ERROR.json"
            with open(LOGS_DIR / log_name, "w") as f:
                json.dump(log_data, f, indent=2)
            continue

        guess, cost = out

        log_name = f"{timestamp}-{exp_id:019d}.json"
        with open(LOGS_DIR / log_name, "w") as f:
            json.dump(log_data, f, indent=2)

        new_df.loc[
            i,
            [
                "guess",
                "expected",
                "is_correct",
                "absolute_error",
                "experiment_id",
                "timestamp",
                "cost",
            ],
        ] = [
            guess,
            sequence_info["expected"],
            guess == sequence_info["expected"],
            abs(guess - sequence_info["expected"]),
            exp_id,
            timestamp,
            cost,
        ]

        if VERBOSE >= 1:
            print(f"  Guess: {guess}")
            print(f"  Expected: {sequence_info['expected']}")
            print(f"  IsCorrect: {new_df.at[i, 'is_correct']}")
            print(f"  AbsoluteDifference: {new_df.at[i, 'absolute_error']}")

        completed.append(i)

    new_df = new_df.loc[completed]

    if existing_df is None:
        combined_df = new_df
    else:
        existing_df.to_parquet(
            EXPERIMENTS_DF_PATH.with_suffix(EXPERIMENTS_DF_PATH.suffix + ".backup"), index=False
        )
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    combined_df.to_parquet(EXPERIMENTS_DF_PATH, index=False)

    print(f"\nAll experiments completed. Results saved to {EXPERIMENTS_DF_PATH}")
    print(f"Logs saved to {LOGS_DIR}")


if __name__ == "__main__":
    main()
