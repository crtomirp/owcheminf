import json
from pathlib import Path

src = Path("smartspains.json")
rules = json.loads(src.read_text(encoding="utf-8"))

out = Path("chem_inf_widgets/chemcore/data/smartspains_data.py")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    f.write("# Auto-generated from smartspains.json. Do not edit by hand.\n")
    f.write("SMARTSPAINS_RULES = ")
    f.write(repr(rules))
    f.write("\n")
print("Wrote", out)
