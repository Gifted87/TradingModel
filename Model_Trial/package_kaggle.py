"""
Package source code and configuration for Kaggle execution.
Creates kaggle_package/trading_model_src.zip containing the entire src/ tree.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

def package():
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / "src"
    out_dir = base_dir / "kaggle_package"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "trading_model_src.zip"

    files_added = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                # Skip pycache and temporary compiled dumps
                if "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
                    continue
                if p.name == "compiled_codebase.txt":
                    continue
                arcname = p.relative_to(base_dir).as_posix()
                zf.write(p, arcname=arcname)
                files_added += 1

    size_kb = out_zip.stat().st_size / 1024
    print(f"✅ Successfully packaged {files_added} source files into {out_zip} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    package()
