from pathlib import Path
import re


def rename(trg_dir):
    for old_path in Path(trg_dir).glob("**/*.pdf"):
        old_name = old_path.name
        new_name = old_name.lower()
        new_name = re.sub(r"[ -]", "_", new_name)
        new_name = re.sub(r"[\r\n]+", "_", new_name)
        new_name = re.sub(r"_+", "_", new_name)

        if new_name != old_name:
            print(f"'{old_name}'\n -> '{new_name}'")
            old_path.rename(old_path.parent/new_name)


def main():
    rename(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
