from pathlib import Path
import re


def rename(trg_dir):
    num = 1
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
        new_path = (old_path.parent / new_stem).with_suffix(".pdf")

        if new_stem != old_stem:
            if new_path.exists():
                print(f"'{new_stem} already exists!'")
            print(f"[{num}] '{old_stem}'\n    -> '{new_stem}'")
            num += 1
            old_path.rename(
                f"{str(old_path.parent / new_stem)}.pdf"
            )


def main():
    rename(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
