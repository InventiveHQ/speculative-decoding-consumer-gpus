"""Inject results/data.json into site/index.html (replaces the __DATA__ token)."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.join(ROOT, "results", "data.json"), encoding="utf-8"))
tpl_path = os.path.join(ROOT, "site", "index.html")
html = open(tpl_path, encoding="utf-8").read()
import re
blob = json.dumps(data, indent=2, ensure_ascii=False)
repl = "const DATA = " + blob + ";"
if "const DATA = __DATA__;" in html:
    html = html.replace("const DATA = __DATA__;", repl)
else:
    # already injected once: replace the existing const DATA = {...}; (lambda repl avoids \-escape parsing)
    html = re.sub(r"const DATA = \{.*?\n\};", lambda m: repl, html, count=1, flags=re.S)
open(tpl_path, "w", encoding="utf-8").write(html)
print("Injected DATA into", tpl_path)
print(blob[:600])
