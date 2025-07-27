# LLMs Guessing Integer Sequences

How well can LLMs guess the next integer in sequence puzzles? We can easily test this on annotated sequences from the [On-Line Encyclopedia of Integer Sequences](OEIS.org) (OEIS).

The quality of the guess would depend on model and whether we allow it to perform an analysis or if we force it to generate a single guess.


## Setup

We test different models, setups, and parameters:

- Sequence starting point.
- Sequence length.
- Adding to the prompt a hint or description of the sequence (or not).
- Different LLMs of varying sizes.
- Progressively more capable settings:
  - **Single guess:** The model only outputs a single number.
  - **Reasoning:** The model is allowed to reason until it reaches an answer.
  - **Compute:** The model can reason as before, and will write a Python script, then guess after seeing its output.

We evaluate and analyse two measures of error:

- Guessed correctly.
- Integer distance to correct guess.


## OEIS puzzles

OEIS catalogues puzzles with IDs starting with "A" and followed by 6 digits. We can retrieve all information on a sequence in json format from:

```
https://oeis.org/A******?fmt=json
```

We save the fields:

```python
number: int         # The integer in A______
data: list[int]     # The integer sequence
name: str           # Usually includes a definition/description of the sequence
comment: list[str]  # Different interpretations of the sequence plus sources
```

## Expectations

We expect that the sequences may be bunched and categorized given the results. For example:

- **In-memory sequences:** Some sequences are well-known or are generated from very simple recursion relations, so even small models on "Single guess" can generate good guesses. Model size/capabilities will have little effect.
  - **[A000040](https://oeis.org/A000040):** The prime numbers.
- **Reasoning sequences:** These sequences may be unknown or not remarkable, but the recursion relation is simple enough to be guessed after some reasoning.
  - **[A001020](https://oeis.org/A001020):** Powers of 11: a(n) = 11^n.
- **Hard problem solutions:** Some sequences do not follow a simple recursion relation but are instead the solution(s) to a more complex math question. Proper reasoning and computation are needed to generate a good guess, and most likely the description of the sequence too.
  - **[A007510](https://oeis.org/A007510):** Primes p such that neither p-2 nor p+2 is prime.
  - **[A001006](https://oeis.org/A001006):** Number of ways of drawing any number of nonintersecting chords joining n (labeled) points on a circle.
  - **[A001050](https://oeis.org/A001050):** Number of letters in n (in Finnish).

Almost all puzzles belong to the last category.

### Results Structure

```python
class Result:
    # Sequence metadata
    oeis_id: str                    # e.g., "A000040"
    sequence_data: list[int]        # The actual sequence values
    sequence_name: str              # OEIS name/description

    # Experimental conditions
    model_name: str                 # e.g., "gpt-4", "claude-3-sonnet"
    capability_level: str           # "single_guess", "reasoning", "compute"
    sequence_length: int            # How many terms provided
    starting_point: int             # Which term we started from
    include_description: bool       # Whether OEIS description was included

    # Results
    correct_answer: int             # The true next term
    model_guess: int                # What the model guessed
    is_correct: bool                # Whether guess was exactly right
    absolute_error: int             # |guess - correct|

    # Additional data
    model_reasoning: str            # Full reasoning/explanation (if applicable)
    generated_code: str | None      # Python code (if compute mode)
    response_time: float            # How long the model took
    timestamp: datetime             # When experiment was run
```

