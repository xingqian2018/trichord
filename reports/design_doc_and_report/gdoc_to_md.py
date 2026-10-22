import json
import sys


HEADING = {
    "TITLE": "#",
    "HEADING_1": "##",
    "HEADING_2": "###",
    "HEADING_3": "####",
    "HEADING_4": "#####",
}


def load(path):
    data = json.load(open(path))
    if isinstance(data, list):
        data = json.loads(data[0]["text"])
    if "content" in data and isinstance(data["content"], (dict, str)):
        data = data["content"]
        if isinstance(data, str):
            data = json.loads(data)
    return data


def runtext(elements):
    out = ""
    for e in elements:
        if "textRun" in e:
            t = e["textRun"]["content"]
            style = e["textRun"].get("textStyle", {})
            core = t.rstrip("\n")
            tail = t[len(core):]
            if style.get("bold") and core.strip():
                lead = core[: len(core) - len(core.lstrip())]
                trail = core[len(core.rstrip()):]
                core = f"{lead}**{core.strip()}**{trail}"
            if style.get("link", {}).get("url") and core.strip():
                core = f"[{core}]({style['link']['url']})"
            out += core + tail
        elif "person" in e:
            props = e["person"].get("personProperties", {})
            out += "@" + (props.get("name") or props.get("email", "")) + " "
        elif "dateElement" in e:
            out += e["dateElement"].get("dateElementProperties", {}).get("displayText", "")
        elif "richLink" in e:
            props = e["richLink"].get("richLinkProperties", {})
            out += f"[{props.get('title', '')}]({props.get('uri', '')})"
    return out.replace("\ue907", "").rstrip("\n")


def celltext(cell):
    parts = [runtext(c["paragraph"]["elements"]) for c in cell["content"] if "paragraph" in c]
    return " ".join(p.strip() for p in parts if p.strip()).replace("|", "\\|")


def render(doc):
    tab = doc["tabs"][0]["documentTab"] if "tabs" in doc else doc
    lists = tab.get("lists", {})
    counters = {}
    lines = []
    inlist = False
    for el in tab["body"]["content"]:
        if "paragraph" in el:
            p = el["paragraph"]
            text = runtext(p["elements"])
            kind = p["paragraphStyle"].get("namedStyleType", "NORMAL_TEXT")
            if kind in HEADING:
                inlist = False
                if text.strip():
                    lines += ["", f"{HEADING[kind]} {text.strip().strip('*')}", ""]
                continue
            if "bullet" in p:
                if not inlist and lines and lines[-1].strip():
                    lines.append("")
                inlist = True
                lid = p["bullet"]["listId"]
                level = p["bullet"].get("nestingLevel", 0)
                glyph = lists.get(lid, {}).get("listProperties", {}).get("nestingLevels", [{}])[level]
                indent = "  " * level
                if "glyphType" in glyph and glyph["glyphType"] not in ("GLYPH_TYPE_UNSPECIFIED", "NONE"):
                    key = (lid, level)
                    counters[key] = counters.get(key, 0) + 1
                    lines.append(f"{indent}{counters[key]}. {text.strip()}")
                else:
                    lines.append(f"{indent}- {text.strip()}")
                continue
            if inlist and text.strip():
                lines.append("")
            inlist = False
            lines.append(text)
        elif "table" in el:
            inlist = False
            rows = [[celltext(c) for c in r["tableCells"]] for r in el["table"]["tableRows"]]
            lines.append("")
            lines.append("| " + " | ".join(rows[0]) + " |")
            lines.append("|" + "---|" * len(rows[0]))
            for r in rows[1:]:
                lines.append("| " + " | ".join(r) + " |")
            lines.append("")
    out = []
    for line in lines:
        if line.strip() == "" and out and out[-1].strip() == "":
            continue
        out.append(line.rstrip())
    return "\n".join(out).strip() + "\n"


def main():
    src, docid, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    doc = load(src)
    header = f"<!-- synced from https://docs.google.com/document/d/{docid}/edit revision {doc.get('revisionId', '')} -->\n\n"
    open(dst, "w").write(header + render(doc))


if __name__ == "__main__":
    main()
