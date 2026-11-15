# The search behind the activity-trace set

How the papers in [`../papers.csv`](../papers.csv) were found, on 15 November 2026. These scripts query public
services (Europe PMC's REST API and PMC's open-access bucket on Amazon S3), so a later run will find more papers.

| File | What it does |
|---|---|
| `search.py` | searches Europe PMC: activity named in a figure legend, with days or light–dark conditions, mice or rats, open access, CC BY (the query is at the top of the script). 3,950 hits, 3,921 in PMC. |
| `harvest.py` | for each paper, its licence and files from PMC's open-access bucket, and from its XML the figure legends, data-availability statement, supplementary files and the rhythm-analysis methods its text names. 3,886 papers had a CC BY or CC0 licence. Needs lxml. |
| `score.py` | scores each legend for activity drawn as a line over consecutive days (activity named +3, a number of days +2, a trace or time course +1, bins +1, light–dark +1; behavioural tests −2; actograms −1). |
| `screened.csv` | the 781 papers with a legend scoring 6 or more: their best-scoring figures, data files, repository, rhythm methods named, and whether the paper was shortlisted. |
| `text.py` | prints the paragraphs of a paper that cite a figure; used to find what each paper says about its traces. |

Figures of the screened papers were then looked at by eye, about 170 of them; [`../papers.csv`](../papers.csv) lists
the 26 shortlisted and why 8 of those were dropped.
