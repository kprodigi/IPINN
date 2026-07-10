"""Build Manuscript_v2.docx from paper/manuscript_v2.md with the PRODIGI
Elsevier reference template.

Implements the four known pandoc/PRODIGI pitfalls:
(a) template Heading1-4 auto-number via numbering.xml -> content carries no
    explicit section numbers, so auto-numbering is the single source;
(b) the raw template lacks centred Title/Author styles -> added here;
(c) a custom --reference-doc must contain EVERY style pandoc emits, or
    LibreOffice floats the tables -> merge pandoc's default styles in;
(d) pandoc sizes table columns equally -> post-process to content-aware
    widths, and strip the first-line indent from the cell style.

Usage:  python paper/build/build_docx.py   (from anywhere; paths are absolute)
"""
import copy
import os
import re
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.path.join(REPO, "paper")
BUILD = os.path.join(PAPER, "build")
MD = os.path.join(PAPER, "manuscript_v2.md")
SUPP_MD = os.path.join(PAPER, "supplementary.md")
TEMPLATE_SRC = r"C:\Users\101198337\Dropbox\Ph.D\Template\PRODIGI PAPER TEMPLATE.docx"
TEMPLATE_LOCAL = os.path.join(BUILD, "PRODIGI_TEMPLATE_LOCAL.docx")  # repo-local copy
REF = os.path.join(BUILD, "prodigi_reference.docx")                  # patched reference
OUT = os.path.join(REPO, "Manuscript_v2.docx")
SUPP_OUT = os.path.join(REPO, "Supplementary_Material.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)
def w(tag):
    return f"{{{W}}}{tag}"


def find_pandoc():
    try:
        import pypandoc
        return pypandoc.get_pandoc_path()
    except Exception:
        return "pandoc"


def read_zip_xml(path, member):
    with zipfile.ZipFile(path) as z:
        return z.read(member)


def replace_zip_member(path, member, data):
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == member:
                zout.writestr(item, data)
            else:
                zout.writestr(item, zin.read(item.filename))
    os.replace(tmp, path)


def merge_pandoc_styles(pandoc):
    """Gotcha (c): copy every pandoc-default style the template lacks."""
    default_ref = os.path.join(BUILD, "_pandoc_default_ref.docx")
    with open(default_ref, "wb") as f:
        f.write(subprocess.run([pandoc, "--print-default-data-file", "reference.docx"],
                               capture_output=True, check=True).stdout)
    t_root = ET.fromstring(read_zip_xml(REF, "word/styles.xml"))
    p_root = ET.fromstring(read_zip_xml(default_ref, "word/styles.xml"))
    have = {s.get(w("styleId")) for s in t_root.findall(w("style"))}
    added = []
    for s in p_root.findall(w("style")):
        sid = s.get(w("styleId"))
        if sid not in have:
            t_root.append(copy.deepcopy(s))
            added.append(sid)
    replace_zip_member(REF, "word/styles.xml",
                       ET.tostring(t_root, xml_declaration=True, encoding="UTF-8"))
    print(f"  merged {len(added)} pandoc styles: {', '.join(sorted(added))}")


def _set(el_parent, tag, **attrs):
    el = el_parent.find(w(tag))
    if el is None:
        el = ET.SubElement(el_parent, w(tag))
    for k, v in attrs.items():
        el.set(w(k), v)
    return el


def style_overrides():
    """Gotcha (b): centred TNR Title/Author; gotcha (d) part 2: Compact style
    10 pt with no first-line indent."""
    root = ET.fromstring(read_zip_xml(REF, "word/styles.xml"))
    styles = {s.get(w("styleId")): s for s in root.findall(w("style"))}

    def centre_style(sid, size_halfpt, bold):
        s = styles.get(sid)
        if s is None:
            return
        ppr = s.find(w("pPr")) or ET.SubElement(s, w("pPr"))
        _set(ppr, "jc", val="center")
        ind = _set(ppr, "ind")
        ind.set(w("firstLine"), "0")
        rpr = s.find(w("rPr")) or ET.SubElement(s, w("rPr"))
        fonts = _set(rpr, "rFonts")
        for a in ("ascii", "hAnsi", "cs"):
            fonts.set(w(a), "Times New Roman")
        _set(rpr, "sz", val=str(size_halfpt))
        _set(rpr, "szCs", val=str(size_halfpt))
        b = rpr.find(w("b"))
        if bold and b is None:
            ET.SubElement(rpr, w("b"))
        if not bold and b is not None:
            rpr.remove(b)

    centre_style("Title", 34, True)     # 17 pt bold centred
    centre_style("Author", 24, False)   # 12 pt centred

    comp = styles.get("Compact")
    if comp is not None:
        ppr = comp.find(w("pPr")) or ET.SubElement(comp, w("pPr"))
        ind = _set(ppr, "ind")
        ind.set(w("firstLine"), "0")
        rpr = comp.find(w("rPr")) or ET.SubElement(comp, w("rPr"))
        _set(rpr, "sz", val="20")
        _set(rpr, "szCs", val="20")

    # FirstParagraph must match the body text (the template copy carries a
    # smaller inherited size), with no first-line indent per Elsevier usage.
    fp = styles.get("FirstParagraph")
    if fp is not None:
        ppr = fp.find(w("pPr")) or ET.SubElement(fp, w("pPr"))
        ind = _set(ppr, "ind")
        ind.set(w("firstLine"), "0")
        rpr = fp.find(w("rPr")) or ET.SubElement(fp, w("rPr"))
        _set(rpr, "sz", val="24")
        _set(rpr, "szCs", val="24")

    replace_zip_member(REF, "word/styles.xml",
                       ET.tostring(root, xml_declaration=True, encoding="UTF-8"))
    print("  Title/Author centred; Compact -> 10 pt, no first-line indent")


def autofit_tables(path=None):
    """Gotcha (d) part 1: content-aware column widths in the built docx."""
    path = path or OUT
    root = ET.fromstring(read_zip_xml(path, "word/document.xml"))
    n = 0
    for tbl in root.iter(w("tbl")):
        grid = tbl.find(w("tblGrid"))
        if grid is None:
            continue
        cols = grid.findall(w("gridCol"))
        ncol = len(cols)
        total = sum(int(c.get(w("w"), "0")) for c in cols) or 9360
        # score columns by longest cell text (skip gridSpan cells) and track
        # the longest unbreakable word per column
        score = [3.0] * ncol
        longword = [3] * ncol
        for tr in tbl.findall(w("tr")):
            ci = 0
            for tc in tr.findall(w("tc")):
                tcpr = tc.find(w("tcPr"))
                span = 1
                if tcpr is not None:
                    gs = tcpr.find(w("gridSpan"))
                    if gs is not None:
                        span = int(gs.get(w("val"), "1"))
                if span == 1 and ci < ncol:
                    text = "".join(t.text or "" for t in tc.iter(w("t")))
                    longest = max((len(x) for x in text.split()), default=0)
                    score[ci] = max(score[ci], min(len(text), 60), longest)
                    longword[ci] = max(longword[ci], longest)
                ci += span
        ssum = sum(score)
        # Per-column floor: the longest unbreakable word must fit on one line
        # (~115 twips/char at 10 pt TNR + cell padding), never below 1000
        # twips so short labels ("LC1", "MAPE") don't wrap.  Excess created by
        # the floors is reclaimed proportionally from columns above floor.
        cap = max(1200, (total * 2) // max(ncol, 1))
        floors = [min(cap, max(1000, lw * 115 + 280)) for lw in longword]
        widths = [max(f, total * s / ssum) for f, s in zip(floors, score)]
        excess = sum(widths) - total
        pool = sum(max(0.0, x - f) for x, f in zip(widths, floors))
        if excess > 0:
            if pool > excess:
                widths = [f + max(0.0, x - f) * (1 - excess / pool)
                          for x, f in zip(widths, floors)]
            else:
                # even the floors exceed the table width: scale the floors
                # themselves (some wrapping is then unavoidable but widths
                # stay positive and proportional)
                fsum = sum(floors)
                widths = [f * total / fsum for f in floors]
        widths = [int(round(x)) for x in widths]
        widths[-1] += total - sum(widths)
        for c, wd in zip(cols, widths):
            c.set(w("w"), str(wd))
        for tr in tbl.findall(w("tr")):
            ci = 0
            for tc in tr.findall(w("tc")):
                tcpr = tc.find(w("tcPr"))
                if tcpr is None:
                    tcpr = ET.SubElement(tc, w("tcPr"))
                span = 1
                gs = tcpr.find(w("gridSpan"))
                if gs is not None:
                    span = int(gs.get(w("val"), "1"))
                tcw = tcpr.find(w("tcW"))
                if tcw is None:
                    tcw = ET.SubElement(tcpr, w("tcW"))
                tcw.set(w("type"), "dxa")
                tcw.set(w("w"), str(sum(widths[ci:ci + span]) if ci + span <= ncol else widths[-1]))
                ci += span
        n += 1
    replace_zip_member(path, "word/document.xml",
                       ET.tostring(root, xml_declaration=True, encoding="UTF-8"))
    print(f"  content-aware widths applied to {n} tables")


EMBED_SOURCES = {
    "Fig_framework_schematic.png": "results_paper_v2",
    "Fig_architecture_schematic.png": "results_paper_v2",
    "Fig_dataset_overview.png": "results_paper_v2",
    "Fig_mode_signatures.png": "results_mechanics",
    "Fig_master_curve_collapse.png": "results_mechanics",
    "Fig_unseen_load_curves.png": "results_paper_v2",
    "Fig_parity_unseen.png": "results_paper_v2",
    "Fig_physics_verification.png": "results_paper_v2",
    "Fig_reliability_diagram.png": "results_paper_v2",
    "Fig_design_space.png": "results_paper_v2",
    "Fig_solution_landscape.png": "results_paper_v2",
    "Fig_inverse_parity_uncertainty.png": "results_paper_v2",
    "Fig_inverse_vs_nearest_experimental_curve.png": "results_paper_v2",
    "Fig_pareto_tradeoff.png": "results_paper_v2",
}


def refresh_embed_figs():
    """Regenerate downscaled embed copies whenever the source PNG is newer,
    so re-running this script after a figure replot updates the manuscript."""
    from PIL import Image
    figdir = os.path.join(BUILD, "figs")
    os.makedirs(figdir, exist_ok=True)
    n = 0
    for name, srcdir in EMBED_SOURCES.items():
        src = os.path.join(REPO, srcdir, name)
        dst = os.path.join(figdir, name)
        if not os.path.exists(src):
            print(f"  WARNING: missing source figure {src}")
            continue
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            continue
        img = Image.open(src)
        wpx, hpx = img.size
        if wpx > 2400:
            img = img.resize((2400, round(hpx * 2400 / wpx)), Image.LANCZOS)
        img.save(dst, optimize=True)
        n += 1
    print(f"  embed figures refreshed: {n}")


def main():
    pandoc = find_pandoc()
    os.makedirs(BUILD, exist_ok=True)
    refresh_embed_figs()
    if not os.path.exists(TEMPLATE_LOCAL):
        shutil.copy2(TEMPLATE_SRC, TEMPLATE_LOCAL)
        print(f"  cached template copy -> {TEMPLATE_LOCAL}")
    shutil.copy2(TEMPLATE_LOCAL, REF)

    print("patching reference doc")
    merge_pandoc_styles(pandoc)
    style_overrides()

    print("running pandoc")
    for src, dst in ((MD, OUT), (SUPP_MD, SUPP_OUT)):
        if not os.path.exists(src):
            continue
        subprocess.run([pandoc, src, "--reference-doc", REF, "-o", dst,
                        "--wrap=none"], cwd=PAPER, check=True)
        autofit_tables(dst)
        print(f"  wrote {dst}")
    print("done")


if __name__ == "__main__":
    main()
