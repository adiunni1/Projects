# Projects
A compilation of projects that I've done. 



# Appointment Scheduler

## What This Is

This is a a Python implementation of an appointment scheduling system that tracks a list of appointments and automatically detects any **scheduling conflicts**, particularly where cases where two appointments overlap in time.

I made the project is built around two classes: `Appt` (a single appointment) and `Agenda` (a collection of appointments), and demonstrates object-oriented design using Python's special methods to make the objects work naturally with comparison operators and built-in functions.

---

## How It Works

### `Appt` — A Single Appointment

Each appointment has a start time, an end time, and a description. The class defines comparison operators so appointments can be compared and sorted intuitively:

| Operator | Meaning |
|----------|---------|
| `appt1 < appt2` | `appt1` ends before or exactly when `appt2` starts (no overlap) |
| `appt1 > appt2` | `appt1` starts after or exactly when `appt2` ends (no overlap) |
| `appt1 == appt2` | Both appointments cover the exact same time period |

Two helper methods build on these:

- **`overlaps(other)`** — returns `True` if there is any time overlap between the two appointments
- **`intersect(other)`** — returns a new `Appt` representing the overlapping time window, or `None` if there is no overlap

### `Agenda` — A Collection of Appointments

An `Agenda` holds a list of `Appt` objects and provides two key operations:

- **`sort()`** — sorts all appointments by start time
- **`conflicts()`** — returns a new `Agenda` containing only the overlapping portions of any conflicting appointments

The conflict detection algorithm sorts appointments first, then uses a nested loop with an early exit: once it finds an appointment that starts after the current one ends, it can skip the rest (since all subsequent appointments will start even later).

---

## Example

```python
from datetime import datetime
from appointments import Appt, Agenda

appt1 = Appt(datetime(2024, 3, 15, 13, 30), datetime(2024, 3, 15, 15, 30), "Early afternoon nap")
appt2 = Appt(datetime(2024, 3, 15, 15, 0),  datetime(2024, 3, 15, 16, 0),  "Coffee break")

agenda = Agenda()
agenda.append(appt1)
agenda.append(appt2)

conflicts = agenda.conflicts()
print(conflicts)
# 2024-03-15 15:00 15:30 | Overlap
```

---

## Key Design Decisions

- **Comparison operators** (`__lt__`, `__gt__`, `__eq__`) are defined in terms of the *time period*, not the description — two appointments at the same time are "equal" regardless of what they're called
- **`overlaps` is derived from `__lt__` and `__gt__`** rather than implementing its own logic, keeping the definition clean and consistent
- **Conflict detection uses an early exit** — after sorting, once a non-overlapping later appointment is found, the inner loop breaks, making the algorithm more efficient than a naive double loop

---

## Dependencies

- Python 3.10+
- Standard library only (`datetime`)

---

## Files

- **`appointments.py`** — contains both the `Appt` and `Agenda` classes, plus a short demo in the `__main__` block


# Movie Genre Classifier

## What This Is

This project builds a machine learning classifier that predicts whether a movie is a **comedy or thriller** based purely on the words used in its screenplay. Given a movie's word-frequency profile, the classifier finds the most similar movies in the training set and uses their genres to make a prediction.

The dataset contains around 5,000 stemmed word features extracted from movie scripts, along with metadata like title, year, and rating.

---

## How It Works

### The Core Idea: K-Nearest Neighbors (k-NN)

Rather than learning explicit rules about what makes a movie a comedy or thriller, k-NN works by similarity: given a new movie, find the *k* most similar movies in the training set and take a majority vote on their genres.

"Similarity" here means **Euclidean distance** across all word-frequency features. Two movies that use words like "laugh" and "marri" (stemmed form of "marry") frequently will be close together in feature space; movies heavy on "dead" and "cop" will cluster differently.

### Pipeline

1. **Data Loading** — load `movies.csv` (one row per movie, columns for each stemmed word) and `stem.csv` (mapping stems back to their original words)
2. **Exploratory Analysis** — visualize word correlations and genre distributions across the dataset
3. **Train/Test Split** — 85% training, 15% test (no shuffling; sequential split)
4. **Feature Selection** — experiment with different word feature subsets to find the most predictive ones
5. **Classification** — for each test movie, compute distances to all training movies and take a majority vote among the *k* nearest neighbors
6. **Evaluation** — measure proportion of correct predictions on the held-out test set

---

## Key Functions

### `distance(features_array1, features_array2)`
Computes Euclidean distance between two movies represented as arrays of word-frequency values. Works across any number of features.

### `classify(test_row, train_rows, train_labels, k)`
Core classifier. Given a test movie's feature vector, finds the *k* nearest neighbors in the training set and returns the most common genre among them.

```python
classify(test_features, train_features, train_labels, k=13)
```

### `classify_feature_row(row)`
Wrapper around `classify` for use with `DataFrame.apply()` — runs the classifier on every row of the test set in one call.

### `most_common(label, table)`
Returns the most frequent value in a column of a DataFrame. Used to tally genre votes among nearest neighbors.

### `su(array)`
Standardizes an array to zero mean and unit standard deviation. Used when computing correlations between word features.

---

## Feature Sets Explored

Two feature sets were tested and compared:

| Feature Set | Words |
|-------------|-------|
| Common words | `i`, `the`, `to`, `a`, `it`, `and`, `that`, `of`, `your`, `what` |
| Genre-specific words | `laugh`, `marri`, `dead`, `heart`, `cop` |

The genre-specific word set captures more meaningful signal for distinguishing comedies from thrillers.

---

## Results

- The k-NN classifier achieved **85% accuracy** on the held-out test set
- Best results used k=13 neighbors and the genre-specific feature set
- Accuracy was measured as the proportion of test movies whose predicted genre matched the actual genre

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install numpy pandas matplotlib seaborn
```

---

## Files

- **`mov_gen_classifier.py`** — main script with all analysis and classification code
- **`movies.csv`** — dataset of movies with stemmed word frequencies and genre labels (not included in repo; required to run)
- **`stem.csv`** — mapping from stemmed words back to their original forms (not included in repo; required to run)

**Duck Machine Assembler, Phase 1**

What This Is

Writing code directly in machine language (raw numbers the computer understands) is painful and error-prone. Assembly language is a human-readable alternative: instead of calculating numeric instruction values by hand, you write things like ADD or STORE, and an assembler translates that into the binary the machine actually runs.

This project is Phase 1 of a two-phase assembler for the Duck Machine, a simulated CPU architecture used in CIS 211 at the University of Oregon. In my mind, Phase 1's job is to take "shorthand" assembly code and resolve it into a fully explicit form that Phase 2 can then convert into machine code.

Specifically, Phase 1 handles two things that raw Duck Machine assembly doesn't support:


Symbolic labels. Instead of calculating memory addresses by hand (e.g., "jump 3 instructions back"), you label a line and refer to it by name
JUMP pseudo-instructions — a cleaner way to write conditional and unconditional branches, which get translated into the actual ADD r15,... instructions the CPU understands


Example

Instead of writing:

     ADD    r1,r1,r0[2]
     STORE  r1,r0,r0[511]
     SUB    r0,r1,r0[10]
     ADD/P  r15,r0,r15[-3]

You can write:

again: ADD    r1,r1,r0[2]
       STORE  r1,r0,r0[511]
       SUB    r0,r1,r0[10]
       JUMP/P again

Phase 1 resolves again to its actual address and rewrites JUMP/P again as ADD/P r15,r0,r15[-3], which Phase 2 can then encode into machine code.


How It Works

Two-Pass Algorithm

Because a label might be used before it's defined (e.g., jumping forward to a label that appears later in the file), the assembler makes two passes through the source code:


Pass 1 (resolve) — reads every line and builds a dictionary mapping each label name to its memory address
Pass 2 (transform) — goes through the lines again and rewrites any instruction that references a label, replacing it with a PC-relative address


PC-relative means the offset is calculated as target_address - current_address, so the code works regardless of where in memory it's loaded.

Line Types

Each line of assembly is matched against one of four patterns:

KindExampleWhat happensCOMMENT# this is a comment or a blank linePassed through unchanged; does not count as a memory addressDATAx: DATA 42Passed through unchanged; counts as one memory wordFULLADD r1,r1,r0[2]Passed through unchanged; already fully resolvedJUMPJUMP/P againRewritten as ADD/P r15,r0,r15[offset] #again

Labels on any line type are recorded during the first pass.


Files


assembler_phase1.py — this file; Phase 1 assembler
assembler_phase2.py (provided) — takes fully resolved assembly and produces object code (machine instructions as integers)
run/asmgo.py (provided) — convenience script that chains Phase 1, Phase 2, and the Duck Machine simulator together in one command



How to Run

Basic usage

bashpython3 assembler_phase1.py input.asm output.asm


input.asm — your assembly source file (can use labels and JUMP)
output.asm — the resolved output, ready for Phase 2


If no files are specified, it reads from stdin and writes to stdout.

Full pipeline (Phase 1 + Phase 2 + run)

bashpython3 run/asmgo.py programs/asm/your_program.asm


Key Functions

resolve(lines) → dict[str, int]

First pass. Scans all lines and maps each label to its memory address. Comment lines don't count toward addresses; all other line types take up one memory word.

transform(lines) → list[str]

Second pass. Calls resolve first to get the label table, then rewrites each line. JUMP instructions become ADD r15,... instructions with a computed PC-relative offset. Lines that don't need changes are passed through as-is.

parse_line(line) → dict

Tries each regex pattern against a line and returns a dictionary of named fields (label, opcode, predicate, target, offset, comment, etc.) plus a kind field indicating which pattern matched. Raises SyntaxError if nothing matches.

resolve_labels(fields, labels, address)

Given parsed fields, the label table, and the current address, computes the PC-relative offset for a label reference.


Error Handling


Syntax errors, unknown labels, and unexpected exceptions are printed to stderr with the offending line number
The assembler stops after 5 errors to avoid flooding output



Dependencies


Python 3.10+
Standard library only (re, argparse, sys, logging)

