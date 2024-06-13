from pathlib import Path
import re


def rename(trg_dir):
    for old_path in Path(trg_dir).glob("**/*"):
        # old_path=Path("/Users/kimjongbeom/Documents/workspace/AI-Papers/vision/a_style_based_generator_architecture_for_generative_adversarial_networks_pdf")
        old_name = old_path.name
        if old_name[-3:] == "pdf":
            new_name = old_name[:-4]
            print(new_name)
            old_path.rename((old_path.parent/new_name).with_suffix(".pdf"))
    # for old_path in Path(trg_dir).glob("**/*.pdf"):
    #     old_stem = old_path.stem
        
    #     new_stem = old_stem.lower()
    #     new_stem = re.sub(r"[ -:&]", "_", new_stem)
    #     new_stem = re.sub(r",[ _]|[\r\n]+", "_", new_stem)
    #     new_stem = re.sub(r"_+", "_", new_stem)

    #     if new_stem != old_stem:
    #         print(f"'{old_stem}'\n -> '{new_stem}'")
    #         old_path.rename((old_path.parent/new_stem).with_suffix(".pdf"))


def main():
    rename(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
