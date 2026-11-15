# Releasing trace-digitiser

Releases are built and uploaded to PyPI by GitHub Actions
(`.github/workflows/publish.yml`). Nothing is uploaded from a laptop and no
PyPI password or API token is stored anywhere: PyPI is told to trust this
repository's workflow ("trusted publishing").

## One-time setup on PyPI

This has to be done once by an owner of the `trace-digitiser` project on PyPI.

1. Sign in at <https://pypi.org> and open
   <https://pypi.org/manage/project/trace-digitiser/settings/publishing/>
   (*Your projects* -> *Manage* -> *Publishing*).
2. Under *Add a new publisher*, choose **GitHub** and enter:

   | Field             | Value              |
   |-------------------|--------------------|
   | Owner             | `comparativechrono` |
   | Repository name   | `trace-digitiser`  |
   | Workflow name     | `publish.yml`      |
   | Environment name  | *(leave blank)*    |

3. Click **Add**.

That is all.

### Optional hardening

PyPI recommends tying the publisher to a GitHub *environment*, which lets you
require a manual approval before each upload. The workflow does not use one
yet (when the repository was private, GitHub offered environments only on paid
plans). Now that the repository is public, one can be added:

1. Create an environment called `pypi` under *Settings -> Environments*
   (optionally with required reviewers).
2. Add `environment: pypi` to the `publish` job in `publish.yml`.
3. Edit the publisher on PyPI and set *Environment name* to `pypi`.

## Making a release

1. Set `version` in `pyproject.toml`. PyPI never accepts the same version
   twice; the versions already published are listed at
   <https://pypi.org/project/trace-digitiser/#history>.
2. In `CHANGELOG.md`, give the section for the version its date in place of
   "unreleased", and end its link at the new tag (`...vX.Y.Z` in place of
   `...HEAD`).
3. Run the published-figure test set
   ([`validation/published-figures`](validation/published-figures/README.md)),
   which reads figures from papers whose data are published, and compare the
   results with the last release's:

   ```bash
   python validation/published-figures/run.py --digitised \
       --json validation/published-figures/results-X.Y.Z.json \
       --compare validation/published-figures/results-<last release>.json
   ```

   Look into any case marked WORSE before going on. Commit the new results
   and what each case read (`digitised/`), and update the table in that
   folder's README (`--markdown` prints it).

   Then run the activity-trace set
   ([`validation/activity-traces`](validation/activity-traces/README.md)),
   which needs pandas, openpyxl, SciPy and matplotlib besides:

   ```bash
   python validation/activity-traces/run.py
   git diff --stat validation/activity-traces
   ```

   Its readings and results are kept in git (the script records the
   release in `digitised/VERSION`), so the diff shows what the release
   changed. Look into any change in `results/agreement_*.csv` or in the
   verdicts of `results/claims_checked.csv`, and update the numbers in that
   folder's README (`tables.py` prints its tables).
4. Merge to `main` and wait for the *Tests* workflow to pass.
5. On GitHub: *Releases -> Draft a new release*, create the tag `vX.Y.Z`
   (matching the version exactly, with a leading `v`) on `main`, and publish.

Publishing the release starts the workflow. It checks that the tag matches the
version, builds the sdist and wheel, validates them, runs the tests, and
uploads to PyPI. If the tag and version disagree, or a test fails, the run
stops before anything is uploaded.

The README is shown on PyPI as it is in the release. Its links to files in
this repository are absolute, so they work there too.

## Checking a build without releasing

*Actions -> Publish to PyPI -> Run workflow* builds and validates the
distributions and runs the tests, but never uploads anything. Locally:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```
