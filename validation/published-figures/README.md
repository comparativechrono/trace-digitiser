# Published figures with known numbers

trace-digitiser's test suite draws figures from numbers it knows and checks that it reads them back. This folder
holds the other kind of test: figures as they were published in papers whose authors also published the numbers
behind them. `run.py` reads each figure as someone using the package would — first with nothing given, then with
the help the [workbench](https://comparativechrono.github.io/digitiser-workbench/) lets a user give (drawing the plot
area, clicking two ticks on an axis, clicking a colour) — and compares what it reads with the authors' numbers.

It is not part of the test suite (it takes about five minutes, and its results depend a little on the version of
Tesseract), and it is not in the package. Run it before a release, as [RELEASING.md](../../RELEASING.md) says, to
see that the release reads real figures no worse than the last one, and to record how it does.

## Running it

From the repository, with the version to test installed (`pip install -e .`):

```bash
python validation/published-figures/run.py                   # every case
python validation/published-figures/run.py foley andreani-1E # the cases whose names start so
python validation/published-figures/run.py --list            # the cases, and what each is given
python validation/published-figures/run.py --json new.json --compare validation/published-figures/results-2.3.0.json
python validation/published-figures/run.py --digitised                # and keep what each case read
```

`--compare` shows each case's result beside one saved before with `--json`, and marks the cases read better or
worse. `--markdown` prints the table below, and `--results FILE` shows saved results without reading anything
again. `--out DIR` keeps each run's files, with an overlay of what was found drawn on the figure, for looking at.
`--digitised` keeps what each case read in the plots it compares, in [`digitised/`](digitised): a CSV file of the
values and a JSON file saying which panel was compared with which known values, and whether its axes were read.
The paper's S2 and S3 Fig are drawn from them (`paper/figures.py`).
`run.py` needs nothing beyond trace-digitiser (and Tesseract, as trace-digitiser does). `make_known.py`, which
builds `known.json` from the papers' data in `sources/`, also needs openpyxl.

## What counts as read

Errors are in percent of the range an axis is labelled over (0–24 h is 24 h), so that panels in different units
can be compared.

* A **point** is found when it lies within 3 % (of each axis) of a known one. Points are paired with the known ones
  by the Hungarian method. *+N extra* counts the marks read as points that pair with none of them: legend entries,
  letters, a line's corners.
* A **bar** is read when its height is within 1 %. Bars are paired with the known ones from left to right.
* A **curve** is read when the trace that follows it covers at least half of it with a median error of 1 % or less.
* A **dot** of a dot plot is found when it is within 3 % (of the y axis) of one of its column's values. The coloured
  marks found are split into the plot's six columns at the widest gaps between them, then paired by value.
* An **error bar**'s half-length is compared with the standard deviation or error the paper gives: for a point the
  mean of its two arms, for a bar the longer one, as a bar often hides the other.

The totals add up the points, bars, curves and dots read. The figures of Andreani et al. were **held out** until
2.2.0: they were first looked at after the changes in 2.2.0 were made, so their results with 2.2.0 show how those
changes carry over to figures they were not made for. The changes in 2.3.0 were made while looking at them, so their
results with 2.3.0 are not those of figures held out (the totals still count them apart). A figure that has guided a
change is no longer held out; add another in its place.

In Andreani et al.'s Fig 3D three of the six columns are not drawn from the values deposited for them (the means
drawn over their dots are 69.5, 92.3 and 88.4 min; the deposited sheet's are 84.6, 124.7 and 123.7): they are kept
as columns, so that the dots are split into columns as drawn, but not compared.

## Results

| Case | Given | 2.2.0 | 2.3.0 |
|---|---|---|---|
| `leypunskiy-1C-nothing` | nothing | 12/12 points (+2 extra), y error 0.22% (worst 0.63%), error bars 8/12 (0.87%) | 12/12 points, y error 0.31% (worst 0.70%), error bars 6/12 (0.87%) |
| `leypunskiy-1C-colour` | the colour | 12/12 points (+2 extra), y error 0.22% (worst 0.63%), error bars 8/12 (0.87%) | 12/12 points, y error 0.31% (worst 0.70%), error bars 6/12 (0.87%) |
| `leypunskiy-2B-nothing` | nothing | 18:6: y axis not read; 12:12: y axis not read; 6:18: y axis not read | 18:6: 0.58% (worst 1.1%); 12:12: 0.37% (worst 1.4%); 6:18: 0.19% (worst 0.9%) |
| `leypunskiy-2B-areas` | the plot areas | 18:6: 0.72% (worst 1.2%); 12:12: 2.77% (worst 4.7%); 6:18: y axis not read | 18:6: 0.60% (worst 1.1%); 12:12: 0.29% (worst 1.5%); 6:18: 0.24% (worst 0.9%) |
| `leypunskiy-2B-areas-colour` | the plot areas, the line's colour | 18:6: 0.74% (worst 1.2%); 12:12: 2.72% (worst 4.5%); 6:18: y axis not read | 18:6: 0.65% (worst 1.2%); 12:12: 0.35% (worst 1.3%); 6:18: 0.21% (worst 0.9%) |
| `leypunskiy-2B-points` | the plot areas, the markers' colour | 50/102 points (+21 extra), y error 0.70% (worst 2.89%), error bars on 10 points that have none; y axis not read | 99/102 points, y error 0.60% (worst 1.11%) |
| `leypunskiy-2C-nothing` | nothing | 29/29 points (+1 extra), y error 0.07% (worst 0.46%), error bars 24/29 (0.33%) | 29/29 points (+1 extra), y error 0.07% (worst 0.46%), error bars 22/29 (0.33%) |
| `leypunskiy-2C-colours` | the colours | 29/29 points, y error 0.07% (worst 0.64%), error bars 20/29 (0.35%) | 29/29 points (+1 extra), y error 0.07% (worst 0.64%), error bars 18/29 (0.35%) |
| `leypunskiy-2C-colours-size` | the colours, the marker size | 29/29 points (+4 extra), y error 0.08% (worst 0.64%), error bars 23/29 (0.35%) | 29/29 points (+3 extra), y error 0.08% (worst 0.64%), error bars 21/29 (0.36%) |
| `leypunskiy-5A-nothing` | nothing | x axis not read | 9/9 points (+3 extra), y error 0.04% (worst 0.25%), error bars 8/9 (0.02%) |
| `leypunskiy-5A-ticks` | the plot area, two ticks on each axis | 9/9 points (+3 extra), y error 0.04% (worst 0.18%), error bars 9/9 (0.02%) | 9/9 points (+3 extra), y error 0.04% (worst 0.25%), error bars 8/9 (0.02%) |
| `leypunskiy-5A-ticks-colour-size` | the plot area, ticks, the colour, the marker size | 9/9 points (+6 extra), y error 0.03% (worst 0.15%), error bars 8/9 (0.25%) | 9/9 points (+5 extra), y error 0.03% (worst 0.25%), error bars 8/9 (0.26%) |
| `leypunskiy-5B-nothing` | nothing | 5/10 points (+1 extra), y error 0.01% (worst 0.14%), error bars 5/5 (0.02%) | 5/10 points, y error 0.01% (worst 0.14%), error bars 5/5 (0.02%) |
| `leypunskiy-5B-colour-size` | the colour, the marker size | 10/10 points (+10 extra), y error 0.03% (worst 0.11%), error bars 7/10 (0.02%) | 10/10 points (+19 extra), y error 0.09% (worst 0.20%), error bars 8/10 (0.20%) |
| `leypunskiy-5C-nothing` | nothing | 10/19 points (+11 extra), y error 0.04% (worst 0.21%), error bars 9/10 (0.33%) | 15/19 points (+1 extra), y error 0.03% (worst 0.40%), error bars 14/15 (0.34%) |
| `leypunskiy-5C-colours` | the colours | 17/19 points (+14 extra), y error 0.03% (worst 0.28%), error bars 15/17 (0.41%) | 17/19 points (+14 extra), y error 0.03% (worst 0.60%), error bars 13/17 (0.34%) |
| `leypunskiy-5C-colours-size` | the colours, the marker size | 17/19 points (+18 extra), y error 0.03% (worst 0.26%), error bars 12/17 (0.65%) | 17/19 points (+17 extra), y error 0.05% (worst 0.26%), error bars 15/17 (0.34%) |
| `foley-2C-25C-nothing` | nothing | 6/6 bars within 1%, error 0.18% (worst 0.28%), error bars 3/6 (0.08%) | 6/6 bars within 1%, error 0.13% (worst 0.22%), error bars 3/6 (0.08%) |
| `foley-2C-30C-nothing` | nothing | 5/5 bars within 1%, error 0.15% (worst 0.29%), error bars 4/5 (0.07%) | 5/5 bars within 1%, error 0.15% (worst 0.29%), error bars 4/5 (0.07%) |
| `foley-2D-nothing` | nothing | 3/3 bars within 1%, error 0.16% (worst 0.18%), error bars 3/3 (0.06%) | 3/3 bars within 1%, error 0.10% (worst 0.10%), error bars 3/3 (0.06%) |
| `foley-2E-nothing` | nothing | 3/3 bars within 1%, error 0.10% (worst 0.22%), error bars 2/3 (0.42%) | 3/3 bars within 1%, error 0.15% (worst 0.26%), error bars 2/3 (0.42%) |
| `foley-2F-period-nothing` | nothing | 3/3 bars within 1%, error 0.06% (worst 0.12%), error bars 3/3 (0.02%) | 3/3 bars within 1%, error 0.06% (worst 0.14%), error bars 3/3 (0.02%) |
| `foley-2F-rhythmic-nothing` | nothing | 3/3 bars within 1%, error 0.24% (worst 0.25%) | 3/3 bars within 1%, error 0.23% (worst 0.25%) |
| `foley-2G-period-nothing` | nothing | 3/3 bars within 1%, error 0.09% (worst 0.11%), error bars 3/3 (0.08%) | 3/3 bars within 1%, error 0.09% (worst 0.09%), error bars 3/3 (0.08%) |
| `foley-2G-rhythmic-nothing` | nothing | 3/3 bars within 1%, error 0.07% (worst 0.27%) | 3/3 bars within 1%, error 0.10% (worst 0.24%) |
| `foley-2H-period-nothing` | nothing | 3/3 bars within 1%, error 0.13% (worst 0.19%), error bars 3/3 (0.17%) | 3/3 bars within 1%, error 0.13% (worst 0.19%), error bars 3/3 (0.17%) |
| `foley-2H-rhythmic-nothing` | nothing | 3/3 bars within 1%, error 0.21% (worst 0.24%) | 3/3 bars within 1%, error 0.20% (worst 0.23%) |
| `foley-2C-25C-area` | the plot area | 6/6 bars within 1%, error 0.18% (worst 0.28%), error bars 5/6 (0.08%) | 6/6 bars within 1%, error 0.18% (worst 0.28%), error bars 5/6 (0.08%) |
| `pearl-reed-3-curve-nothing` | nothing | logistic: x and y axes not read | logistic: x and y axes not read |
| `pearl-reed-3-dots-nothing` | nothing | x and y axes not read | x and y axes not read |
| `pearl-reed-3-curve-ticks` | two ticks on each axis | logistic: 0.38% (worst 66.9%) | logistic: 0.38% (worst 66.9%) |
| `pearl-reed-3-curve-ticks-exclude` | ticks, the equation excluded | logistic: 0.17% (worst 40.0%) | logistic: 0.17% (worst 40.0%) |
| `pearl-reed-3-dots-ticks` | two ticks on each axis | 0/13 points | 0/13 points |
| `pearl-reed-3-dots-ticks-colour` | ticks, the colour | 13/13 points (+1 extra), y error 0.10% (worst 0.88%), error bars on 13 points that have none | 13/13 points (+5 extra), y error 0.08% (worst 0.88%), error bars on 10 points that have none |
| `andreani-1B-area` | the plot area | x axis not read | x axis not read |
| `andreani-1B-area-colours` | the plot area, the colours | x axis not read | x axis not read |
| `andreani-1B-area-ticks` | the plot area, two ticks on the x axis | 0/37 points (+130 extra) | 0/37 points (+3 extra) |
| `andreani-1B-area-ticks-colours` | the plot area, x ticks, the colours | 33/37 points (+3 extra), y error 0.28% (worst 1.31%), error bars on 24 points that have none | 34/37 points, y error 0.27% (worst 1.31%), error bars on 5 points that have none |
| `andreani-1C-nothing` | nothing | control: 0.60% (worst 2.7%); FD: 4.20% (worst 34.6%) | control: 0.60% (worst 2.7%); FD: 0.99% (worst 19.5%) |
| `andreani-1E-area` | the plot area | y axis not read | 24/24 bars within 1%, error 0.17% (worst 0.39%), error bars 22/24 (0.15%) |
| `andreani-1E-area-ticks` | the plot area, two ticks | 24/24 bars within 1%, error 0.16% (worst 0.39%), error bars 22/24 (0.15%) | 24/24 bars within 1%, error 0.16% (worst 0.39%), error bars 22/24 (0.15%) |
| `andreani-3A-nothing` | nothing | 10/165 dots, y error 0.37% (worst 0.75%) | 120/165 dots (+29 extra), y error 0.14% (worst 2.91%) |
| `andreani-3B-nothing` | nothing | 41/108 dots (+4 extra), y error 0.12% (worst 1.08%) | 77/108 dots, y error 0.09% (worst 2.14%) |
| `andreani-3C-nothing` | nothing | 56/180 dots (+3 extra), y error 0.13% (worst 1.91%) | 114/180 dots (+27 extra), y error 0.15% (worst 2.92%) |
| `andreani-3D-nothing` | nothing | 18/87 dots (+15 extra), y error 1.51% (worst 2.46%) | 37/44 dots (+11 extra), y error 0.14% (worst 2.94%) |
| `andreani-3A-colours` | the colours | 10/165 dots, y error 0.37% (worst 0.75%) | 129/165 dots (+24 extra), y error 0.22% (worst 2.82%) |
| `andreani-3B-colours` | the colours | 48/108 dots (+3 extra), y error 0.12% (worst 2.36%) | 78/108 dots (+8 extra), y error 0.10% (worst 2.14%) |
| `andreani-3C-colours` | the colours | 54/180 dots (+3 extra), y error 0.12% (worst 1.91%) | 141/180 dots (+34 extra), y error 0.18% (worst 2.77%) |
| `andreani-3D-colours` | the colours | 44/87 dots (+32 extra), y error 0.32% (worst 2.94%) | 38/44 dots (+4 extra), y error 0.15% (worst 2.94%) |

Summed over the cases, 2.3.0 reads 366 of 409 points, bars and curves in the development figures (2.2.0: 296) and
818 of 1,192 in those of Andreani et al. (2.2.0: 318 of the same 1,192; 339 of 1,278, with the three columns of 3D
that are no longer compared). The 2.1.0 results are in `results-2.1.0.json`.

What 2.3.0 reads that 2.2.0 did not comes from the changes listed in the [changelog](../../CHANGELOG.md): markers
that overlap, found by the edges of each that can be seen (the dot plots of Andreani et al. 3A-D: 386 of 497 dots
with their colours given, 2.2.0 143); plots stacked with no gap between them (Leypunskiy et al. 2B, now read with
nothing given); tick labels set close together (Leypunskiy et al. 5A with nothing given; Andreani et al. 1E with its
plot area); legends and words inside plots, left out; a line inside a pale band of its own colour (Andreani et al.
1C, FD); and error bars no longer found in shaded bands or broken gridlines.

What is still hard:

* Tick labels that are slanted (Andreani et al. 1B) or printed in an old typeface (Pearl and Reed's scan) are not
  read: two ticks clicked on the axis are needed.
* Two series of one colour told apart only by the shape of their markers (Leypunskiy et al. 5B) are found as one
  colour; with nothing given, only the markers of one series are read, and with the colour and marker size given
  every marker is found, with the other series' markers among the extra marks.
* In the dot plots, dots that lie almost exactly on top of each other, or wholly under a mean's bar, cannot be told
  apart: about a quarter of them are missed. The workbench's *Edit points* tool adds them by hand.
* Error bars of markers are found less often than in 2.2.0 in Leypunskiy et al. 1C and 2C (6 and 22 of 12 and 29,
  against 8 and 24): a bar is no longer accepted where the line through the marker runs on aslant or beyond a gap.
* The fitted curve of Pearl and Reed's scan is followed to within 0.2 % (median), but where it crosses a printed
  gridline the trace is followed up the gridline for a few columns (worst 40 %).

## The figures

### Leypunskiy et al. (2017)

> Leypunskiy E, Lin J, Yoo H, Lee U, Dinner AR, Rust MJ (2017). The cyanobacterial circadian clock follows midday
> in vivo and in vitro. *eLife* 6:e23539. <https://doi.org/10.7554/eLife.23539>

`leypunskiy-2017-fig1.jpg`, `leypunskiy-2017-fig2.jpg` and `leypunskiy-2017-fig5.jpg` are its figures 1, 2 and 5.
Panels used: 1C (peak time against day length: red squares with error bars, a fitted line and three lines for
comparison), 2B (three stacked plots of %P-KaiC over three days: a black line with square markers over shaded days
and nights), 2C (peak time against day length: four series with error bars and fitted lines), 5A (phase shift
against the time of a dark pulse), 5B (two series: squares and triangles) and 5C (five series in five colours), all
with error bars. The numbers are the paper's source data: figure 1—source data 2, figure 2—source data 1 and 2, and
figure 5—source data 1 (sheets *PRC*, *wedge* and *entrainment*). The axis of pulse times in 5A runs 24 h behind the
sheet's: 36 h in the figure is 60 h in the sheet.

### Foley et al. (2019)

> Foley LE, Ling J, Joshi R, Evantal N, Kadener S, Emery P (2019). *Drosophila* PSI controls circadian period and
> the phase of circadian behavior under temperature cycle via *tim* splicing. *eLife* 8:e50063.
> <https://doi.org/10.7554/eLife.50063>

`foley-2019-fig2.jpg` is its figure 2. Panels 2C–2H are ten bar charts, 35 bars in all, filled, hatched or
outlined in black, grey, blue or magenta: the mean circadian period of each genotype, with error bars (SEM) and
brackets marking significance, and the percentage of flies that were rhythmic. The numbers are worked out from
figure 2—source data 1, which gives each fly's period and whether it was rhythmic: the mean and SEM of the periods
of the rhythmic flies, and the percentage rhythmic. For one bar (2H, TG4/+; PdfGAL80/UAS-Psi) that sheet gives a
mean of 23.95 h where the figure and the paper's summary of the periods (figure 2—source data 2) have 24.88 h,
presumably because one fly's period was mistyped; the summary's value is used.

### Pearl and Reed (1920)

> Pearl R, Reed LJ (1920). On the rate of growth of the population of the United States since 1790 and its
> mathematical representation. *Proceedings of the National Academy of Sciences* 6(6):275–288.
> <https://doi.org/10.1073/pnas.6.6.275>

`pearl-reed-1920-fig3.png` is its figure 3 (page 285, "Showing result of fitting equation (xviii) to population
data"): the census populations from 1790 to 1910 as dots, and the logistic curve fitted to them, on a printed grid.
It comes from the Internet Archive's scan of that issue
(<https://archive.org/details/sim_proceedings-of-the-national-academy-of-sciences-usa_1920-06-15_6_6>): page 11 of
its PDF rendered at 300 dpi with `pdftoppm` and cropped to the figure (columns 330–1520, rows 900–1860). The numbers
are the census populations of the paper's table 1 and the curve of its equation (xviii),
y = 2930.3009 / (e<sup>−0.0313395 x</sup> + 0.014854) thousand people, x in years from 1780, which gives the
paper's table 3 of calculated populations to within its rounding. `sources/pearl-reed-1920-tables.csv` holds both
tables, typed from the scan and checked against it.

### Andreani et al. (2022), held out

> Andreani T, Rosensweig C, Sisobhan S, Ogunlana E, Kath W, Allada R (2022). Circadian programming of the
> ellipsoid body sleep homeostat in *Drosophila*. *eLife* 11:e74327. <https://doi.org/10.7554/eLife.74327>

`andreani-2022-fig1.jpg` and `andreani-2022-fig3.jpg` are its figures 1 and 3. Panels used: 1B (sleep through the
day: three series of markers on lines, over shaded areas, with slanted x labels), 1C (activity through the day: two
lines with translucent bands), 1E (24 bars with error bars, a dashed fitted line and an equation in a box) and 3A–D
(dot plots: each fly's sleep lost, baseline sleep and sleep gain at a morning and an evening time, in blue and red,
with means and error bars drawn over the dots). The numbers are from figure 1—source data 1 and figure 3—source
data 1. In the sheet for 1E the two rows under the means are labelled the wrong way round: the row headed "standard
deviation" holds the standard errors the figure draws. Panel 1D is not used, as some of its values lie beyond its
axis.

### Files, licences and checksums

The figures of the eLife articles are as eLife's image server gives them at full size (a JPEG of the published TIFF,
e.g. `https://iiif.elifesciences.org/lax/23539%2Felife-23539-fig1-v2.tif/full/full/0/default.jpg`), unaltered; the
source data are the files published with the articles, unaltered. The articles, their figures and their source data
are © their authors and distributed under the Creative Commons Attribution 4.0 International licence
(<https://creativecommons.org/licenses/by/4.0/>), which allows them to be copied here with this attribution. Pearl
and Reed's paper was published in the United States in 1920 and is in the public domain there.

The MIT licence of this repository covers the code in this folder (`run.py`, `make_known.py`), not the figures or the
data.

| File | Pixels | SHA-256 |
|------|--------|---------|
| `figures/leypunskiy-2017-fig1.jpg` | 2591 × 1778 | `934deab7b369e64cf86d405365cfd651a37e66c8b02c8e22e9e98e1f14778aac` |
| `figures/leypunskiy-2017-fig2.jpg` | 2571 × 1841 | `ea0173487d1dfee9dbd29595b11a7209cf105f89cf3f40c2aeb427abb259a477` |
| `figures/leypunskiy-2017-fig5.jpg` | 2625 × 3287 | `d6dfda55829160ce2b7b2807c66ff1959865c4c0089ded3cfe72479d9f400ad2` |
| `figures/foley-2019-fig2.jpg` | 4421 × 4975 | `fd4425e055045ad6bedeb721d8e4db9dff9b2a51d05a4663d743692a337d3867` |
| `figures/pearl-reed-1920-fig3.png` | 1190 × 960 | `b9a038fb135542c1fd2ad9b0b671a2859ae5a5345039e9fd3aecdceff282ec7c` |
| `figures/andreani-2022-fig1.jpg` | 1950 × 1228 | `3ce15d6a7ec5d914af3141336c9f7cbd71e9e6030e2c2191c4553ef692ccc5fa` |
| `figures/andreani-2022-fig3.jpg` | 2400 × 2068 | `ee4b0af85468ab02630808a4e2b9cbaf1d8b228fd6dd16d0df98d05f361717d5` |
| `sources/elife-23539-fig1-data2-v2.csv` | | `83cc8aa371ae315018798190e7db9045e1b7052e019b2dd3dcd00da21576913b` |
| `sources/elife-23539-fig2-data1-v2.csv` | | `52944edd03b10735e2e449aca727c593adce28bb8bb1257f72fa3c6b4f7fc578` |
| `sources/elife-23539-fig2-data2-v2.xlsx` | | `33d11f6311d4565cf9944d5fd330c7a01db2402e6b0a33b85c89f4ee03ed8f01` |
| `sources/elife-23539-fig5-data1-v2.xlsx` | | `b92783f7d80c00cd35121a0aeaceb741c5f067604a6acafad12e12a23eec319e` |
| `sources/elife-50063-fig2-data1-v2.xlsx` | | `8da568eb81870f3ad4c147102a812c2a6975ebff44d01e435f034053073e2e91` |
| `sources/elife-50063-fig2-data2-v2.xlsx` | | `32b573faa64a3cc0b49b4151c6fe156f1a0a0bce5bea7139f78c3c3bdb44a48f` |
| `sources/elife-74327-fig1-data1-v2.xlsx` | | `3ab520039fb04a2ef5626906b6db9e1eb29d147097f8e907e6511d1ab6718ad6` |
| `sources/elife-74327-fig3-data1-v2.xlsx` | | `ddb5520d723da852e903e88f2db78fc2546ddd07d8499320c0c1106c2b1f8891` |

`results-2.1.0.json` and `results-2.2.0.json` are what `run.py` gave for those releases (Tesseract 5.3.4).

## Adding a figure

1. Put the figure in `figures/` and the data behind it in `sources/` — only if their licence allows them to be
   copied here — and record where they come from above.
2. Add to `make_known.py` the few lines that turn the data into the values the figure shows, and run it.
3. In `run.py`, give the panel's plot area (`BOX`) and the range of its axes (`SPAN`), and add its cases to `CASES`:
   one with nothing given, and one with the help a user would give.
4. Run it with the last release and with the current code, and add both to the results.
