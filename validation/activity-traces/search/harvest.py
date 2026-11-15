"""For each paper found by search.py: its licence and files (PMC open-access bucket), and from its JATS XML the figure
legends, the data-availability statement, the supplementary files, and the rhythm-analysis tools its text names.

    python3 -I harvest.py SEARCH.json OUT_DIR [N_THREADS]

Writes OUT_DIR/PMCxxxx.json for each paper (skipping those already done).
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from lxml import etree

S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
TOOLS = {
    "ClockLab": r"clock\s?lab",
    "chi-square periodogram": r"chi[- ]?squared?\s+periodogram|χ\s?2\s*periodogram|sokolove|qp\s+statistic",
    "Lomb-Scargle": r"lomb[- ]?scargle",
    "periodogram": r"periodogram",
    "cosinor": r"cosinor",
    "JTK_CYCLE": r"jtk[_ ]?cycle",
    "RAIN": r"\bRAIN\b(?! forest)",
    "MetaCycle": r"metacycle",
    "ActogramJ": r"actogram\s?j",
    "CircWave": r"circwave",
    "El Temps": r"el\s+temps",
    "BioDare": r"biodare",
    "CircaCompare": r"circacompare",
    "Actiware/VitalView": r"actiware|vital\s?view",
    "FFT": r"\bFFT\b|fast fourier",
    "wavelet": r"wavelet",
    "autocorrelation": r"autocorrelation",
    "free-running period": r"free[- ]running period|\btau\b|τ",
    "acrophase": r"acrophase",
    "M10/L5/IS/IV": r"\bM10\b|\bL5\b|interdaily stability|intradaily variability",
}
TOOL_RE = {k: re.compile(v, re.I) for k, v in TOOLS.items()}
PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True, recover=True)


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 407):
                raise
        except Exception:
            pass
        time.sleep(1 + 2 * i)
    return None


def text(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def version_of(pmcid):
    meta = get(f"{S3}/{pmcid}.1/{pmcid}.1.json")
    if meta:
        return 1, json.loads(meta)
    listing = get(f"{S3}/?list-type=2&prefix={pmcid}.&max-keys=50")
    if not listing:
        return None, None
    vers = sorted({int(v) for v in re.findall(rf"<Key>{pmcid}\.(\d+)/", listing.decode())})
    if not vers:
        return None, None
    v = vers[-1]
    meta = get(f"{S3}/{pmcid}.{v}/{pmcid}.{v}.json")
    return (v, json.loads(meta)) if meta else (None, None)


def harvest(hit, out_dir):
    pmcid = hit["pmcid"]
    path = os.path.join(out_dir, pmcid + ".json")
    if os.path.exists(path):
        return "done"
    v, meta = version_of(pmcid)
    rec = {**hit, "version": v}
    if not meta:
        rec["error"] = "not in the open-access bucket"
        json.dump(rec, open(path, "w"))
        return "missing"
    rec["license"] = meta.get("license_code")
    rec["media"] = [u.split("/")[-1].split("?")[0] for u in meta.get("media_urls", [])]
    if rec["license"] not in ("CC BY", "CC0"):
        json.dump(rec, open(path, "w"))
        return "licence"
    xml = get(f"{S3}/{pmcid}.{v}/{pmcid}.{v}.xml")
    if not xml:
        rec["error"] = "no XML"
        json.dump(rec, open(path, "w"))
        return "noxml"
    root = etree.fromstring(xml, PARSER)
    if root is None:
        rec["error"] = "XML did not parse"
        json.dump(rec, open(path, "w"))
        return "noxml"
    figs = []
    for f in root.iter("{*}fig", "fig"):
        hrefs = [g.get("{http://www.w3.org/1999/xlink}href") for g in f.iter("{*}graphic", "graphic")]
        figs.append({"id": f.get("id"), "label": text(f.find("{*}label") if f.find("{*}label") is not None
                                                          else f.find("label")),
                     "caption": text(f.find("{*}caption") if f.find("{*}caption") is not None
                                     else f.find("caption"))[:4000],
                     "graphics": [h for h in hrefs if h]})
    rec["figures"] = figs
    da = []
    for el in root.iter():
        t = local(el.tag)
        st = (el.get("sec-type") or el.get("notes-type") or "").lower()
        if t in ("sec", "notes") and ("availab" in st or "data" in st):
            da.append(text(el))
        elif t == "sec":
            title = el.find("{*}title") if el.find("{*}title") is not None else el.find("title")
            if title is not None and re.search(r"data (and code )?availab|availability of data|data access",
                                               text(title), re.I):
                da.append(text(el))
        elif t == "custom-meta":
            if re.search(r"availab", text(el.find("{*}meta-name") if el.find("{*}meta-name") is not None
                                          else el.find("meta-name")), re.I):
                da.append(text(el))
    rec["data_availability"] = " | ".join(dict.fromkeys(da))[:3000]
    supp = []
    for s in root.iter("{*}supplementary-material", "supplementary-material"):
        hrefs = [m.get("{http://www.w3.org/1999/xlink}href") for m in s.iter("{*}media", "media")]
        supp.append({"label": text(s.find("{*}label") if s.find("{*}label") is not None else s.find("label")),
                     "caption": text(s.find("{*}caption") if s.find("{*}caption") is not None
                                     else s.find("caption"))[:600],
                     "files": [h for h in hrefs if h]})
    rec["supplementary"] = supp
    body = root.find(".//{*}body")
    if body is None:
        body = root.find(".//body")
    btext = text(body)
    rec["tools"] = {}
    for k, rx in TOOL_RE.items():
        m = list(rx.finditer(btext))
        if m:
            rec["tools"][k] = [btext[max(0, x.start() - 120): x.end() + 120] for x in m[:2]]
    rec["n_body_chars"] = len(btext)
    json.dump(rec, open(path, "w"))
    return "ok"


def main(search, out_dir, n=8):
    os.makedirs(out_dir, exist_ok=True)
    hits = json.load(open(search))["hits"]
    counts = {}
    t = time.time()
    with ThreadPoolExecutor(n) as ex:
        for i, r in enumerate(ex.map(lambda h: harvest(h, out_dir), hits)):
            counts[r] = counts.get(r, 0) + 1
            if i % 200 == 0:
                print(i, counts, f"{time.time() - t:.0f} s", file=sys.stderr, flush=True)
    print("finished", counts, f"{time.time() - t:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
