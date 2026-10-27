# github-advi-usage

> **Note to self:** when this repo lives in iCloud-synced Documents, the
> installed commands (`github-advi-usage ...`) fail with "No module named
> github_advi_usage" because iCloud hides the editable install's `.pth` file.
> Run `python -m github_advi_usage.cli ...` instead (e.g.
> `python -m github_advi_usage.cli --db data/github-vi-software.sqlite analyze`).

Companion package to simpleVI. It searches GitHub for code using
ADVI interfaces in PyMC, Pyro, NumPyro, TensorFlow
Probability, and Stan, then classifies every matched call site by whether
the optimizer and MC/sample-count were left at the package's default or
explicitly adjusted.

It has two independent capabilities, each its own CLI subcommand:

1. **`fetch`** — search GitHub and scan every
   matched file into a local SQLite database. Needs network access and a
   GitHub token.
2. **`analyze`** — classify what's in the fetched database and write CSV reports.
   No network access required.

Run either alone, or both together with `all`.

## Install

```bash
python -m pip install -e .
```

The only dependency is `requests`; `analyze` itself is pure standard library.

## GitHub API access

`fetch` calls the GitHub REST and Code Search APIs, which require
authentication even for public repositories. Create a token first:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens) and
   generate a token — either a classic personal access token with no scopes
   selected (code search and public repo contents don't need any), or a
   fine-grained token with "Public repositories" read-only access.
2. Make it available to this package, either:
   - `export GITHUB_TOKEN=ghp_...` in your shell, or
   - a `.env` file at the repository root containing `GITHUB_TOKEN=ghp_...`
     (loaded automatically; never commit this file — it's gitignored).

GitHub's Code Search API is rate-limited; `fetch` already paces its requests
(`--search-interval`, `--scan-interval`) to stay under those limits, so a
full run can take a while. `analyze` needs no token and no network access.

## Usage

```bash
export GITHUB_TOKEN=...

# Capability (1): populate/refresh the database from GitHub.
github-advi-usage --db data/github-vi-software.sqlite fetch

# Capability (2): classify the database and write reports/*.csv.
github-advi-usage --db data/github-vi-software.sqlite analyze

# Do both in sequence.
github-advi-usage --db data/github-vi-software.sqlite all
```

`--db` defaults to `data/github-vi-software.sqlite`. Reports are written
next to the database: passing that exact default path writes to
`data/reports/`; any other `--db` path writes to `<db's folder>/<db
stem>/reports/` instead (e.g. `--db /tmp/foo.sqlite` → `/tmp/foo/reports/`).
Override directly with `--reports-dir`.

### Choosing what `analyze` writes

By default (no flags), `analyze` writes five files: `call_site_evidence.csv`,
`default_advi_repos.csv`, `adjusted_advi_repos.csv`, `repo_level_summary.csv`,
and `overview.csv` — see "Reading the output" below for what each contains.
Flags narrow or expand that:

| Flag | Writes |
|---|---|
| `--default-only` | only `default_advi_repos.csv` |
| `--adjusted-only` | only `adjusted_advi_repos.csv` |
| `--overview` | only `overview.csv` |
| `--call-sites-only` | only `call_site_evidence.csv` |
| `--repo-summary-only` | only `repo_level_summary.csv` |
| `--include-review-queue` | adds the three review-queue files (below) to whatever else was requested |
| `--comprehensive` | everything, including the review-queue and audit files |

These are additive — `--default-only --overview` writes exactly those two.
By default, `analyze` also deletes any report in `--reports-dir` that the
current run didn't ask for, so the directory always matches exactly what you
requested; pass `--keep-existing` to disable that and accumulate instead.

Other `analyze` flags: `--limit N` caps how many new call sites get
classified in one run; `--refresh` reclassifies call sites that already have
a classification (useful after a rule change); `--skip-verify` skips the
internal consistency check that otherwise runs after writing.

## Reading the output

Every output lives under `reports/`, one row per call site unless noted.

**Core (written by default):**

- **`call_site_evidence.csv`** — one row per matched call site: its
  `final_classification`, the exact snippet (`actual_call_snippet`), a
  permalink to the line (`github_permalink`), and `classification_reason`.
  `final_classification` is one of `default`, `adjusted_optimizer`,
  `adjusted_mc`, `adjusted_both`, `needs_review`, or
  `no_vi_fit` (a GitHub search hit that turned out not to be a VI fit call —
  `why_in_query_results` says why: `import_only`, `comment_only`,
  `method_repo_code_itself`, or `no_vi_fit`).
- **`default_advi_repos.csv`** — the `repo_link`/`filename`/`line_number` of
  every call site classified `default`: both the optimizer and the
  MC/sample-count were left at the package's built-in default.
- **`adjusted_advi_repos.csv`** — same three columns plus
  `adjusted_setting`, for every call site that changed something:
  `optimizer` (only the optimizer was changed), `mc` (only MC/sample-count
  was changed), or `both`.
- **`repo_level_summary.csv`** — one row per repository: which VI packages
  it uses, whether it adjusts the optimizer and/or MC/sample-count anywhere
  (`adjusts_optimizer`/`adjusts_mc`), and `supporting_call_site_ids` linking back to the
  specific call sites behind that row.
- **`overview.csv`** — `metric,value` pairs: total call sites, the
  default/adjusted/adjusted-optimizer-only/adjusted-mc-only/adjusted-both
  breakdown, how many repositories fall into each bucket, and so on. Start
  here before reading the row-level files.

**Review queue (`--include-review-queue` or `--comprehensive`):** working
files for the interactive review tools below, not final results.

- **`needs_review_queue.csv`** — call sites whose classification is
  uncertain (e.g. it's in a test or tutorial path) and wants a human look.
- **`unresolved_parameter_queue.csv`** — call sites where the optimizer or
  MC/sample-count value couldn't be statically resolved (e.g. it's passed in
  through a variable the static analysis couldn't trace).
- **`parameter_resolution_evidence.csv`** — the audit trail of every
  parameter value that *was* resolved (by the automated resolver, by manual
  review, or by `deep-resolve`), plus the still-unresolved ones.

**Audit (`--comprehensive` only):**

- **`duplicate_repo_clusters.csv`** — groups of repositories whose
  `needs_review` call sites substantially overlap (independent re-uploads of
  the same content), with one canonical repository picked per group.
- **`random_repo_review_actions.csv`** — the saved decisions from spot-check
  review sessions (below); empty until you run one.

## Interactive review tools

```bash
# Resolve needs_review/unresolved entries one at a time.
github-advi-usage-manual-review --db data/github-vi-software.sqlite --reports-dir data/reports

# Spot-check a random sample of already-classified repos for quality control.
github-advi-usage-random-review --db data/github-vi-software.sqlite --reports-dir data/reports

# Re-fetch more files from a specific repo to resolve a value that couldn't
# be traced statically (network access; reads unresolved_parameter_queue.csv).
github-advi-usage-deep-resolve --db data/github-vi-software.sqlite \
    --unknown-csv data/reports/unresolved_parameter_queue.csv
```

`manual-review` shows one entry at a time — the call site, its link, snippet,
and current classification — and asks you to pick a label (`default`,
`adjusted_mc`, `adjusted_optimizer`, `adjusted_both`, or one of several
false-positive labels), or skip/stop. Decisions are saved
to the database immediately; `--no-rebuild-each` defers regenerating the
queue files until you stop, instead of after every decision.

`random-review` samples from `repo_level_summary.csv`, shows a repo's
classification plus a few of its supporting call sites, and asks whether
that classification looks correct. It's a QC audit trail — it doesn't change
any classification itself; use `manual-review` for that.

## Search queries

`fetch` runs every query in `github_advi_usage/github_query_manifest.csv`,
which records each query with when it was run and how many results it
returned. Queries GitHub caps at 1,000 results are split into `size:`
windows automatically; the manifest's `size:` rows are the windows an
earlier run split off, so `fetch` runs their base query once instead. Rows
Q100–Q106 have not been run for the saved database yet.

Use `--manifest path/to/other.csv` to search with a different manifest, or
`--query "..."` (repeatable) to run specific queries instead.

The classification rules themselves (what counts as "default" per package,
what's ignored) are documented in `github_advi_usage/classification_rules.md`.

## Other commands

`analyze` runs three steps: classify new call sites, write the reports, and
verify them. Two commands run the last two steps on their own:

```bash
# Rewrite reports from the database's existing classifications, without
# classifying new call sites or verifying. Takes the same output-selection
# flags as `analyze` (--default-only, --comprehensive, --keep-existing, ...).
github-advi-usage-build --db data/github-vi-software.sqlite --reports-dir data/reports

# Check the consistency of whichever reports are present in a directory,
# e.g. after editing or copying them. Exits non-zero on failure.
github-advi-usage-verify --reports-dir data/reports
```

To see how a piece of code would be classified without touching a database,
use `classify_source` from `github_advi_usage/vi_github_classifier.py`, which
runs the same rules as `analyze` on an in-memory copy;
`github_advi_usage/example_query_and_classification.ipynb` shows it in use.

## Data is gitignored

`.gitignore` excludes `/data/` (the SQLite database and every `reports/*.csv`
output) and any stray `*.sqlite*` file — the database contains verbatim
scraped third-party GitHub code, and the reports are regeneratable from it,
so neither belongs in version control. Only code, docs, and the query
manifest are tracked.

## License

MIT — see `LICENSE`.
