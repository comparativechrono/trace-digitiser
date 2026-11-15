"""Score each figure legend of the harvested papers for how likely it is to show activity drawn as a line over
consecutive days, and summarise each paper: licence, data, tools.

    python3 -I score.py PAPERS_DIR OUT.json
"""
import glob
import json
import re
import sys

ACT = re.compile(r"locomotor activit|home[- ]cage activit|wheel[- ]running|running[- ]wheel|wheel revolutions|"
                 r"activity counts|beam[- ]?breaks?|beam crossings|infrared|telemetr|spontaneous (locomotor )?activit|"
                 r"general activit|gross motor|movement counts|activity level|activity rhythm|locomotion", re.I)
DAYS = re.compile(r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|fourteen)[- ]?(consecutive |successive |)"
                  r"(days?|d\b|nights?|cycles?)|consecutive days|successive days|over (several|multiple) days|"
                  r"days? \d+ ?(-|to|–) ?\d+|day-by-day|continuous(ly)? (recorded|monitored|recording)", re.I)
TRACE = re.compile(r"\btraces?\b|\bline\b|profiles?|time[- ]course|time series|plotted|over time|across (the )?days|"
                   r"recorded (for|over|during)|continuous", re.I)
BINS = re.compile(r"\b\d+[- ]?(min|minute|h|hr|hour)[- ]?(bins?|intervals?|epochs?)|per (hour|30 ?min|10 ?min)|"
                  r"hourly|/h\b|/ ?30 ?min|bins? of", re.I)
LDDD = re.compile(r"\bLD\b|\bDD\b|\bLL\b|light[-/ ]dark|dark[-/ ]light|constant dark|constant light|12 ?: ?12|\bZT\b|"
                  r"zeitgeber|lights? (on|off)|dark phase|light phase|circadian|diurnal", re.I)
NEG = re.compile(r"open[- ]field|elevated plus|rotarod|y[- ]maze|forced swim|novel object|tail suspension|"
                 r"morris water|fear condition|social interaction|light[- ]dark box|light/dark box|marble", re.I)
ACTOGRAM = re.compile(r"actogram|double[- ]plotted|raster", re.I)


def score(caption):
    s = 0.0
    a = bool(ACT.search(caption))
    s += 3 * a
    s += 2 * bool(DAYS.search(caption))
    s += 1 * bool(TRACE.search(caption))
    s += 1 * bool(BINS.search(caption))
    s += 1 * bool(LDDD.search(caption))
    s -= 2 * bool(NEG.search(caption))
    s -= 1 * bool(ACTOGRAM.search(caption)) * (not TRACE.search(caption))
    return s if a else 0.0


DATA_EXT = re.compile(r"\.(xlsx?|csv|tsv|txt|zip|mat|json|ods|dat|awd|h5|rds|rdata|sav|prism|pzfx)$", re.I)
REPO = re.compile(r"dryad|figshare|zenodo|osf\.io|github|mendeley data|gin\.g-node|dandi|openneuro|"
                  r"10\.5061|10\.6084|10\.5281|10\.17632|data\.mendeley|harvard dataverse|10\.7910", re.I)
ON_REQUEST = re.compile(r"(up)?on (reasonable )?request|from the corresponding author|available from the authors", re.I)


def main(d, out):
    papers = []
    for f in sorted(glob.glob(d + "/PMC*.json")):
        r = json.load(open(f))
        if r.get("license") not in ("CC BY", "CC0") or "figures" not in r:
            continue
        figs = []
        for fig in r["figures"]:
            sc = score(fig["caption"])
            if sc >= 6:
                figs.append({"label": fig["label"], "score": sc, "graphics": fig["graphics"], "id": fig["id"],
                             "caption": fig["caption"]})
        if not figs:
            continue
        data_files = [m for m in r.get("media", []) if DATA_EXT.search(m)]
        da = r.get("data_availability") or ""
        papers.append({
            "pmcid": r["pmcid"], "doi": r.get("doi"), "title": r.get("title"), "journal": r.get("journalTitle"),
            "year": r.get("pubYear"), "authors": r.get("authorString"), "license": r["license"],
            "version": r.get("version"), "figures": figs, "best": max(x["score"] for x in figs),
            "data_files": data_files, "repository": bool(REPO.search(da)), "on_request": bool(ON_REQUEST.search(da)),
            "data_availability": da[:600], "tools": sorted(r.get("tools", {})),
            "supplementary": [{"label": s["label"], "caption": s["caption"][:200], "files": s["files"]}
                              for s in r.get("supplementary", [])],
        })
    papers.sort(key=lambda p: -p["best"])
    json.dump(papers, open(out, "w"), indent=1)
    print(len(papers), "papers with a figure scoring 6 or more", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
