"""Helper script to package code and configuration files for easy Kaggle upload."""

import zipfile
from pathlib import Path

def create_code_zip():
    output_zip = Path("kaggle_code.zip")
    src_files = list(Path("src").glob("*.py"))
    config_files = list(Path("configs").glob("*.yaml"))
    
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src_files:
            zf.write(f, arcname=str(f))
            print(f"Added {f}")
        for f in config_files:
            zf.write(f, arcname=str(f))
            print(f"Added {f}")
        if Path("kaggle_train_conformer.ipynb").exists():
            zf.write("kaggle_train_conformer.ipynb", arcname="kaggle_train_conformer.ipynb")
            print("Added kaggle_train_conformer.ipynb")
            
    print(f"\nSuccessfully created {output_zip.resolve()} ({output_zip.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    create_code_zip()
