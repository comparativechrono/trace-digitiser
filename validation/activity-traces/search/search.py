"""Search Europe PMC for open-access mouse/rat papers whose figure legends describe activity recorded over days.

    python3 -I search.py OUT.json
"""
import json
import sys
import time
import urllib.parse
import urllib.request

ACT = ('(FIG:"locomotor activity" OR FIG:"home cage activity" OR FIG:"home-cage activity" OR FIG:"wheel-running" '
       'OR FIG:"wheel running" OR FIG:"running wheel" OR FIG:"running-wheel" OR FIG:"activity counts" '
       'OR FIG:"beam breaks" OR FIG:"beam-break" OR FIG:"spontaneous activity" OR FIG:"general activity" '
       'OR FIG:"gross motor activity" OR FIG:"locomotor rhythm" OR FIG:"activity rhythm")')
TIME = ('(FIG:"days" OR FIG:"LD" OR FIG:"DD" OR FIG:"light-dark" OR FIG:"light/dark" OR FIG:"constant darkness" '
        'OR FIG:"12:12" OR FIG:"circadian" OR FIG:"diurnal")')
SP = ('(FIG:mice OR FIG:mouse OR FIG:rats OR FIG:rat OR ABSTRACT:mice OR ABSTRACT:mouse OR ABSTRACT:rats '
      'OR ABSTRACT:rat)')
QUERY = f'{ACT} AND {TIME} AND {SP} AND OPEN_ACCESS:y AND LICENSE:"cc by"'


def main(out):
    hits, cursor = [], "*"
    while True:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
            {"query": QUERY, "format": "json", "pageSize": 1000, "resultType": "lite", "cursorMark": cursor})
        with urllib.request.urlopen(url, timeout=120) as r:
            d = json.load(r)
        for x in d.get("resultList", {}).get("result", []):
            if x.get("pmcid"):
                hits.append({k: x.get(k) for k in ("pmcid", "pmid", "doi", "title", "authorString", "journalTitle",
                                                   "pubYear")})
        nxt = d.get("nextCursorMark")
        print(len(hits), "of", d.get("hitCount"), file=sys.stderr)
        if not nxt or nxt == cursor or not d.get("resultList", {}).get("result"):
            break
        cursor = nxt
        time.sleep(0.5)
    json.dump({"query": QUERY, "date": time.strftime("%Y-%m-%d"), "hits": hits}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
