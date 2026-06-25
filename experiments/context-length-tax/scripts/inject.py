"""Inject results/data.json into site/index.html (replaces the __DATA__ token)."""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.join(ROOT, "results", "data.json"), encoding="utf-8-sig"))
tpl = os.path.join(ROOT, "site", "index.html")
html = open(tpl, encoding="utf-8").read()
repl = "const DATA = " + json.dumps(data, indent=2, ensure_ascii=False) + ";"
if "const DATA = __DATA__;" in html:
    html = html.replace("const DATA = __DATA__;", repl)
else:
    html = re.sub(r"const DATA = \{.*?\n\};", lambda m: repl, html, count=1, flags=re.S)
open(tpl, "w", encoding="utf-8", newline="\n").write(html)
print("Injected DATA into", tpl)
