"""Print the paragraphs of a paper (PMC XML) that cite a figure, for reading what the authors say about it.

    python text.py PAPER.xml "Fig 1" [more labels ...] [--words rhythm,activity]
"""
import re
import sys
import xml.etree.ElementTree as ET


def paragraphs(path):
    root = ET.parse(path).getroot()
    for sec in root.iter():
        if sec.tag in ("p", "caption"):
            text = "".join(sec.itertext())
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                yield sec.tag, text


def main(path, *args):
    words = []
    labels = []
    it = iter(args)
    for a in it:
        if a == "--words":
            words = next(it).split(",")
        else:
            labels.append(a)
    pats = [re.compile(r"\b" + re.escape(lab).replace(r"\ ", r"\.?\s*") + r"(?![0-9])", re.I) for lab in labels]
    for tag, t in paragraphs(path):
        if any(p.search(t) for p in pats) or (words and all(w.lower() in t.lower() for w in words)):
            print(f"[{tag}] {t}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
