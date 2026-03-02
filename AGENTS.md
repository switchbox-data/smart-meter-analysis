# Agent guide: smart-meter-analysis

This file orients AI agents so they can work effectively in this repo — building data pipelines, running analysis, and managing regulatory-grade datasets — without reading the entire codebase.

## What this repo is

**smart-meter-analysis** is [Switchbox's](https://switch.box/) smart meter data pipeline and analysis repo. Switchbox is a nonprofit think tank that produces rigorous, accessible data on U.S. state climate policy for advocates, policymakers, and the public.

This repo processes ComEd smart meter data for the [Citizens Utility Board](https://www.citizensutilityboard.org/) (CUB) of Illinois, supporting regulatory proceedings that examine utility rate equity and energy affordability. It combines a **data engineering pipeline** (CSV-to-Parquet compaction) with **statistical analysis** (rate simulations, clustering, regression) to produce regulatory-grade datasets and publication-ready figures.

The main inputs are ComEd interval meter data (CSV and Parquet), Census demographic data, and geographic shapefiles. The main outputs are compacted Parquet datasets, statistical analyses, GeoJSON maps, and figures for regulatory testimony.

The companion repo [reports2](https://github.com/switchbox-data/reports2) produces the final published reports from these analysis outputs. See its AGENTS.md for report-writing conventions.

## Layout

| Path                              | Purpose                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------ |
| `scripts/csv_to_parquet/`         | CSV-to-Parquet migration pipeline: ingestion, compaction, validation.                      |
| `scripts/analysis/`              | Analysis scripts: rate comparisons, clustering, regression.                                |
| `scripts/bench/`                 | Benchmarking scripts for pipeline performance.                                             |
| `smart_meter_analysis/`          | Installable Python package (shared utilities).                                             |
| `analysis/`                      | Exploratory analysis notebooks and one-off investigations.                                 |
| `tests/`                         | Pytest test suite.                                                                         |
| `config/`                        | Configuration files for pipeline runs.                                                     |
| `data/`                          | Local data cache (gitignored — real data lives on S3).                                     |
| `results/`                       | Analysis outputs: tables, summary statistics.                                              |
| `figures/`                       | Generated plots and maps.                                                                  |
| `docs/`                          | MkDocs documentation.                                                                      |
| `infra/`                         | Terraform and EC2 infrastructure (see `infra/README.md` for VM setup).                     |
| `logs/`                          | Pipeline run logs.                                                                         |
| `archive/`                       | Archived scripts and old approaches (reference only).                                      |
| `.devcontainer/`                 | Dev container configuration (Dockerfile, devcontainer.json).                               |
| `Justfile`                       | Root task runner: `install`, `check`, `test`, `dev-setup`, `dev-login`, `dev-teardown`.    |
| `pyproject.toml`                 | Python dependencies (managed by uv).                                                       |

## Pipeline architecture

The primary data engineering work is compacting raw ComEd CSV exports (~30,000 files per month) into sorted, validated Parquet files suitable for regulatory testimony. This pipeline is the core of the repo — understand it before touching any pipeline code.

### Data flow

```text
Raw CSVs (S3/local) → Ingestion → Monthly Parquet files → Compaction → Validation → Validated output
```

1. **Ingestion** (`scripts/csv_to_parquet/migrate_month_runner.py`): Reads raw CSVs, converts to Parquet with consistent schema.
2. **Compaction** (`scripts/csv_to_parquet/compact_month_output.py`): Merges monthly Parquet files into fewer, larger files with correct sort order. Uses a two-pass k-way merge-sort to handle ~60 input files and up to 500M rows per month.
3. **Validation**: Checks schema consistency, sort order, null thresholds, duplicate detection, row count expectations.

### Critical constraints

These constraints are non-negotiable. They exist because pipeline outputs support regulatory testimony subject to cross-examination.

- **Memory**: Pipeline runs on EC2 (m7i.2xlarge: 8 vCPUs, 32 GB RAM). All operations must stay within ~28 GB working memory. Use PyArrow `iter_batches()` for streaming reads. Never `collect()` or `to_pandas()` on full monthly datasets.
- **Sort order**: All compacted output must be sorted by `(account_id, date)`. Downstream analysis and regulatory reproducibility depend on it. Verify sort order explicitly after every compaction operation — do not trust it implicitly.
- **Data quality**: Every transformation must be auditable and reproducible. No silent data loss. No silent duplicate creation.
- **Naming conventions**: Output files follow Spark conventions with `_SUCCESS.json` metadata markers.

### Reading large data

**Polars (preferred for analysis):**

```python
import polars as pl

# Lazy scan — stays out of memory until .collect()
lf = pl.scan_parquet("/data.sb/comed/interval_data/2023/*.parquet")
result = lf.filter(pl.col("account_id") == "12345").collect()
```

**PyArrow (preferred for pipeline I/O):**

```python
import pyarrow.parquet as pq

# Streaming read for large files — never loads full file into memory
pf = pq.ParquetFile("/data.sb/comed/interval_data/2023/07.parquet")
for batch in pf.iter_batches(batch_size=100_000):
    process(batch)
```

Stay in lazy execution as long as possible. Only `collect()` / `compute()` when you need the data in memory and have filtered first.

### What NOT to do in pipeline code

- Do not load full months into memory with `pl.read_parquet()` or `pd.read_parquet()`. Use `pl.scan_parquet()` or PyArrow's `pq.ParquetFile` with `iter_batches()`.
- Do not use pandas in production pipeline code. Use PyArrow for I/O and Polars for transforms.
- Do not hardcode file paths. Use config files or CLI arguments.
- Do not skip validation after compaction. Every output file must pass schema and sort-order checks.
- Do not assume sort order is preserved through joins or concatenations. Verify explicitly.

## Analysis conventions

Analysis scripts examine how alternative electricity rate structures (DTOU, Rate BEST) affect different customer segments. The analysis feeds into reports published via the reports2 repo.

### Methods

- **Rate simulation**: Computing hypothetical bills under alternative tariff structures for ~328,000 Chicago households.
- **DTW clustering**: Identifying distinct electricity usage patterns from interval data (19.8M household-day observations).
- **Multinomial logistic regression**: Quantifying how demographics explain usage pattern membership.
- **Geographic analysis**: GeoJSON maps showing rate impact by census block group.
- **Income regression**: Scatterplots examining equity (regressivity/progressivity) across income levels.

### Output standards

Analysis outputs may be used in regulatory testimony. Every number must be traceable to source data through documented transformations.

- Figures go to `figures/` directory.
- Summary tables go to `results/` directory.
- All statistics must be reproducible from source data.
- No hardcoded statistics — compute from data.
- Document assumptions in code comments explaining _why_ a threshold, filter, or parameter was chosen.

## Working with data

All data lives on S3, mounted at `/data.sb/` on the EC2 VM. Never store data files in git.

### Storage paths

| Location    | Path        | Size      | Persistent? | Use                              |
| ----------- | ----------- | --------- | ----------- | -------------------------------- |
| S3 mount    | `/data.sb/` | Unlimited | Yes         | Source data, shared datasets     |
| EBS volume  | `/ebs/`     | 500 GB    | Yes         | Home directories, persistent work |
| Local cache | `data/`     | —         | No (git)    | Temporary local data (gitignored) |

### S3 naming conventions

```text
s3://data.sb/<org>/<dataset>/<filename_YYYYMMDD.parquet>
```

- Lowercase with underscores. Date suffix reflects when data was downloaded.
- Always use a dataset directory, even for single files.
- Prefer Parquet format.

### Local caching

`data/` is gitignored. Use it for caching downloads and intermediate results, but the analysis must be reproducible from S3 alone. Never reference local-only files in committed code without a clear download/generation step.

## Code quality

Before considering any change done:

- **`just check`**: Runs pre-commit hooks (ruff-check, ruff-format, trailing whitespace, end-of-file newline, YAML/JSON/TOML validation, no large files, no merge conflict markers).
- **`just test`**: Runs pytest suite. Add or extend tests for new or changed behavior.

Python formatting: Ruff for formatting and linting. Type checking: mypy or ty.

## How to work in this repo

### Tasks

Use `just` as the main interface. The root `Justfile` handles dev tasks and VM management.

### Dependencies

- **Python**: `uv add <package>` (updates `pyproject.toml` + `uv.lock`). Never use `pip install`.

### Computing contexts

- Data scientists' laptops (Mac with Apple Silicon)
- EC2 VM via `just dev-login` (`m7i.2xlarge`: 8 vCPUs, 32 GB RAM, 500 GB EBS)
- Be aware of which context you're in (affects available memory, S3 latency, and data access patterns).

### AWS

Data is on S3 in `us-west-2`. The EC2 VM mounts S3 at `/data.sb/`. See `infra/README.md` for full VM setup, login, and teardown instructions. Always run `just dev-teardown` when done to avoid unnecessary AWS costs.

## Commits, branches, and PRs

### Commits

- **Atomic**: One logical change per commit.
- **Message format**: Imperative verb, <50 char summary (e.g., "Fix compaction sort-key overlap").
- **WIP commits**: Prefix with `WIP:` for work-in-progress snapshots.

### Branches and PRs

- **PR title** MUST start with `[area]` (e.g., `[pipeline] Fix memory overflow in compaction stage`) — this becomes the squash-merge commit message on `main`.
- **Create PRs early** (draft is fine). This gives the team visibility into in-flight work.
- PRs should **merge within the sprint**; break large work into smaller PRs if needed.
- **Delete branches** after merging.
- **Description**: Don't duplicate the issue. Write: high-level overview, reviewer focus, non-obvious implementation details.
- **Close the GitHub issue**: Include `Closes #<github_issue_number>` (not the Linear identifier).
- Do not add "Made with Cursor" or LLM attribution.

## Issue conventions

All work is tracked via Linear issues (which sync to GitHub Issues). When creating or updating tickets, use the Linear MCP tools. Every new issue MUST satisfy the following before it is created:

### Issue fields

- **Type**: One of **Code** (delivered via commits/PRs), **Research** (starts with a question, findings documented in issue comments), or **Other** (proposals, graphics, coordination — deliverables vary).
- **Title**: `[area] Brief description` starting with a verb (e.g., `[pipeline] Add sort-order validation to compaction stage`).
- **What**: High-level description. Anyone can understand scope at a glance.
- **Why**: Context, importance, value.
- **How** (skip only when the What is self-explanatory and implementation is trivial):
  - For Code issues: numbered implementation steps, trade-offs, dependencies.
  - For Research issues: background context, options to consider, evaluation criteria.
- **Deliverables**: Concrete, verifiable outputs that define "done":
  - Code: "PR that adds ...", "Tests for ...", "Updated `data/` directory with ..."
  - Research: "Comment in this issue documenting ... with rationale and sources"
  - Other: "Google Doc at ...", "Slide deck for ...", link to external deliverable
  - Never vague ("Finish the analysis") or unmeasurable ("Make it better").
- **Project**: Must be set.
- **Status**: Default to Backlog. Options: Backlog, To Do, In Progress, Under Review, Done.
- **Milestone**: Set when applicable (strongly encouraged).
- **Assignee**: Set if known.
- **Priority**: Set when urgency/importance is clear.

### Status transitions

Keep status updated as work progresses — this is critical for team visibility:

- **Backlog** -> **To Do**: Picked for the current sprint
- **To Do** -> **In Progress**: Work has started (branch created for code issues)
- **In Progress** -> **Under Review**: PR ready for review, or findings documented
- **Under Review** -> **Done**: PR merged (auto-closes), or reviewer approves and closes

## Conventions agents should follow

1. **Memory first.** Always consider RAM constraints. Use streaming/lazy patterns for any dataset over 1 GB.
2. **Sort order is sacred.** Compacted output must be sorted by `(account_id, date)`. Verify after every compaction operation.
3. **No pandas in pipeline code.** PyArrow for I/O, Polars for transforms.
4. **Validate everything.** After compaction, after joins, after any transformation that could silently drop or duplicate rows.
5. **Use Context7.** Always look up current library docs before writing PyArrow or Polars code. These APIs change frequently. Do not rely on training data for API signatures.
6. **Run `just check`** before considering a change done.
7. **Config over hardcoding.** File paths, thresholds, and parameters belong in config files or CLI arguments, not inline.
8. **Data never goes in git.** S3 and `/data.sb/` for real data, `data/` (gitignored) for local caches.
9. **Tests for pipeline changes.** Any change to ingestion, compaction, or validation must have corresponding tests.
10. **Document assumptions.** Pipeline code is regulatory evidence. Comments should explain _why_ a threshold was chosen, why a filter is applied, what the expected data shape is.

## MCP Tools

### Context7

When writing or modifying code that uses a library, use the Context7 MCP server to fetch up-to-date documentation. Do not rely on training data for API signatures or usage patterns.

### Linear

When a task involves creating, updating, or referencing issues, use the Linear MCP server to interact with the workspace directly. Follow the issue conventions above.

## Quick reference

| Command                 | Where | What it does                              |
| ----------------------- | ----- | ----------------------------------------- |
| `just install`          | Root  | Set up dev environment                    |
| `just check`            | Root  | Lint, format, pre-commit hooks            |
| `just test`             | Root  | Run pytest suite                          |
| `just dev-setup`        | Root  | Spin up EC2 VM (one-time admin)           |
| `just dev-login`        | Root  | Log in to EC2 VM                          |
| `just dev-teardown`     | Root  | Stop VM, preserve data volume             |
| `just dev-teardown-all` | Root  | Destroy VM and all data (permanent)       |
