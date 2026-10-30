"""Download a small, balanced chest-X-ray subset from Hugging Face.

Creates data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA,TUBERCULOSIS}.
The source dataset is CC BY 4.0; retain attribution when presenting results.
"""
from pathlib import Path

from datasets import load_dataset

OUT = Path("data/chest_xray")
CLASSES = {0: "NORMAL", 1: "PNEUMONIA"}
TARGET_PER_CLASS = 1000
SPLITS = {"train": 700, "val": 150, "test": 150}


def main():
    counts = {
        name: len(list(OUT.glob(f"*/{name}/*.jpg"))) for name in CLASSES.values()
    }
    dataset = load_dataset("hf-vision/chest-xray-pneumonia", split="train")
    for row in dataset:
        label = row["label"]
        class_name = CLASSES[int(label)]
        if counts[class_name] >= TARGET_PER_CLASS:
            if all(value >= TARGET_PER_CLASS for value in counts.values()):
                break
            continue
        index = counts[class_name]
        split = "train" if index < SPLITS["train"] else "val" if index < SPLITS["train"] + SPLITS["val"] else "test"
        destination = OUT / split / class_name
        destination.mkdir(parents=True, exist_ok=True)
        image = row["image"]
        image.save(destination / f"{class_name.lower()}_{index:04d}.jpg")
        counts[class_name] += 1
        if sum(counts.values()) % 25 == 0:
            print(counts)
    print("Finished:", counts)


if __name__ == "__main__":
    main()
