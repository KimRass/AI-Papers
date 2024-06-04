from pathlib import Path
import re


def rename(trg_dir):
    for path in Path(trg_dir).glob("**/*.pdf"):
        new_name = path.name.lower()
        new_name = re.sub(r"[ -]", "_", new_name)
        # new_name = re.sub(r"[:]", "", new_name)
        new_name = re.sub(r'_+', '_', new_name)

        path.rename(path.parent/new_name)


def main():
    rename(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
