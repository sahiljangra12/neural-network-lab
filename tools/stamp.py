"""Cache-busting: stamp every CSS/JS reference (and every ES-module import) with ?v=<version>,
so a browser can never mix files from two different deployments. Run before each commit."""
import re, sys, time, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
v = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d%H%M")
n = 0
for f in ["index.html", "learn.html", "advanced.html"]:
    p = root / f; s = p.read_text()
    s2, k = re.subn(r'((?:href|src)="(?:css/[\w-]+\.css|js/[\w-]+\.js))(?:\?v=\w+)?"', rf'\1?v={v}"', s)
    p.write_text(s2); n += k
for p in (root / "js").glob("*.js"):
    s = p.read_text()
    s2, k = re.subn(r"(from '\./[\w-]+\.js)(?:\?v=\w+)?'", rf"\1?v={v}'", s)
    p.write_text(s2); n += k
print(f"stamped {n} references with v={v}")
