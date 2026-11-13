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

That is all. The link works while the repository is private.

### Optional hardening

PyPI recommends tying the publisher to a GitHub *environment*, which lets you
require a manual approval before each upload. GitHub only offers environments
on private repositories for paid plans, so the workflow does not use one. Once
the repository is public (or on a paid plan):

1. Create an environment called `pypi` under *Settings -> Environments*
   (optionally with required reviewers).
2. Add `environment: pypi` to the `publish` job in `publish.yml`.
3. Edit the publisher on PyPI and set *Environment name* to `pypi`.

## Making a release

1. Set `version` in `pyproject.toml` (PyPI never accepts the same version
   twice; the last published version is 1.2.0).
2. In `CHANGELOG.md`, give the section for the version its date in place of
   "not yet released".
3. Merge to `main` and wait for the *Tests* workflow to pass.
4. On GitHub: *Releases -> Draft a new release*, create the tag `vX.Y.Z`
   (matching the version exactly, with a leading `v`) on `main`, and publish.

Publishing the release starts the workflow. It checks that the tag matches the
version, builds the sdist and wheel, validates them, runs the tests, and
uploads to PyPI. If the tag and version disagree, or a test fails, the run
stops before anything is uploaded.

The README is shown on PyPI as it is in the release. Its links to files in
this repository are absolute, so they work there only once the repository is
public.

## Checking a build without releasing

*Actions -> Publish to PyPI -> Run workflow* builds and validates the
distributions and runs the tests, but never uploads anything. Locally:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```
