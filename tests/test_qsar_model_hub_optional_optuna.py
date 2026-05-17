from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def test_qsar_model_hub_imports_without_optuna(tmp_path):
    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text(
        textwrap.dedent(
            """
            import builtins

            _real_import = builtins.__import__

            def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "optuna" or name.startswith("optuna."):
                    raise ImportError("optuna intentionally blocked for lean-env smoke test")
                return _real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = _blocked_import
            """
        ),
        encoding="utf-8",
    )

    code = textwrap.dedent(
        """
        from chem_inf_widgets.chemcore.services.qsar_model_hub_service import available_model_keys, hpo_available
        from chem_inf_widgets.widgets.ow_qsar_model_hub import OWQSARModelHub

        assert not hpo_available()
        assert "auto" not in available_model_keys()
        assert "auto" not in OWQSARModelHub._MODEL_KEYS
        print("ok")
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(SRC_ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep)

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert proc.stdout.strip() == "ok"
