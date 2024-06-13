from pathlib import Path
import re


def rename(trg_dir):
    for old_path in Path(trg_dir).glob("**/*.pdf"):
        old_stem = old_path.stem
        
        new_stem = old_stem.lower()
        new_stem = re.sub(r"[ -]", "_", new_stem)
        new_stem = re.sub(r"[\r\n]+", "_", new_stem)
        new_stem = re.sub(r", ", "_", new_stem)
        new_stem = re.sub(r",_", "_", new_stem)
        new_stem = re.sub(r"&", "_", new_stem)
        new_stem = re.sub(r":_|: |:", "_", new_stem)
        new_stem = re.sub(r"_+", "_", new_stem)

        if new_stem != old_stem:
            print(f"'{old_stem}'\n -> '{new_stem}'")
            old_path.rename((old_path.parent/new_stem).with_suffix(".pdf"))


def main():
    rename(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
