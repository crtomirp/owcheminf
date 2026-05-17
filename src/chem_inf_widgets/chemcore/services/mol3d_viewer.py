from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class Viewer3DConfig:
    width: int = 800
    height: int = 600
    style: str = "stick"  # stick | sphere | line
    background: str = "white"

    surface: bool = False
    surface_opacity: float = 0.6

    add_hs: bool = True
    optimize: bool = True
    random_seed: int = 0xC0FFEE

    highlight_atoms: Optional[List[int]] = None
    highlight_color: str = "orange"
    highlight_radius: float = 0.35

    label: Optional[str] = None


def ensure_3d(
    mol: Chem.Mol,
    add_hs: bool = True,
    optimize: bool = True,
    random_seed: int = 0xC0FFEE,
) -> Chem.Mol:
    if mol is None:
        raise ValueError("mol is None")

    m = Chem.Mol(mol)

    if m.GetNumConformers() > 0:
        conf = m.GetConformer()
        if conf.Is3D():
            return m

    if add_hs:
        m = Chem.AddHs(m, addCoords=True)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = True
    params.maxAttempts = 50

    res = AllChem.EmbedMolecule(m, params)
    if res != 0:
        raise ValueError("RDKit failed to generate 3D conformer (EmbedMolecule).")

    if optimize:
        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass

    return m


def mol_to_molblock_3d(mol: Chem.Mol, cfg: Viewer3DConfig) -> str:
    m3 = ensure_3d(mol, add_hs=cfg.add_hs, optimize=cfg.optimize, random_seed=cfg.random_seed)
    return Chem.MolToMolBlock(m3)


def make_3dmol_html_local_js(molblock: str, cfg: Viewer3DConfig) -> str:
    """
    HTML that expects 3Dmol-min.js in same folder as baseUrl (loaded as ./3Dmol-min.js).
    """
    mb = html.escape(molblock)

    hl = cfg.highlight_atoms or []
    hl_js = "[" + ",".join(str(i) for i in hl) + "]"

    # style mapping
    if cfg.style == "sphere":
        style_js = "{sphere:{scale:0.35}}"
    elif cfg.style == "line":
        style_js = "{line:{}}"
    else:
        style_js = "{stick:{radius:0.18}}"

    surface_js = ""
    if cfg.surface:
        surface_js = f"""
            try {{
              viewer.addSurface($3Dmol.VDW, {{opacity:{float(cfg.surface_opacity)}}});
            }} catch(e) {{}}
        """

    label_js = ""
    if cfg.label:
        lbl = html.escape(str(cfg.label))
        label_js = f"""
            try {{
              viewer.addLabel("{lbl}", {{
                backgroundColor:"rgba(255,255,255,0.75)",
                fontColor:"#222",
                fontSize:12,
                inFront:true
              }});
            }} catch(e) {{}}
        """

    highlight_js = ""
    if hl:
        highlight_js = f"""
            try {{
              viewer.setStyle({{index:{hl_js}}}, {{stick:{{radius:0.25,color:"{cfg.highlight_color}"}}}});
              viewer.addStyle({{index:{hl_js}}}, {{sphere:{{radius:{float(cfg.highlight_radius)},color:"{cfg.highlight_color}"}}}});
            }} catch(e) {{}}
        """

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="./3Dmol-min.js"></script>
  <style>
    html, body {{ margin:0; padding:0; background:{cfg.background}; }}
    #viewer {{ width:{cfg.width}px; height:{cfg.height}px; }}
    .err {{ padding:10px; font-family: sans-serif; color:#a40000; }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <script>
    if (typeof $3Dmol === "undefined") {{
      document.body.innerHTML = '<div class="err"><b>3Dmol.js not loaded.</b><br/>Check that 3Dmol-min.js is available locally (assets/3dmol).</div>';
    }} else {{
      var element = document.getElementById("viewer");
      var viewer = $3Dmol.createViewer(element, {{ backgroundColor: "{cfg.background}" }});
      viewer.addModel(`{mb}`, "sdf");
      viewer.setStyle({{}}, {style_js});
      {highlight_js}
      {surface_js}
      viewer.zoomTo();
      {label_js}
      viewer.render();
    }}
  </script>
</body>
</html>
"""
