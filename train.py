import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def load_split(path: Path, class_names=None, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=class_names,
        shuffle=shuffle,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    root = Path("data") / args.module
    train = load_split(root / "train")
    class_names = train.class_names
    val = load_split(root / "val", class_names=class_names)
    class_counts = {i: 0 for i in range(len(class_names))}
    for _, labels in train:
        for label in np.argmax(labels.numpy(), axis=1):
            class_counts[int(label)] += 1
    total = sum(class_counts.values())
    class_weights = {
        index: total / (len(class_names) * count)
        for index, count in class_counts.items()
        if count
    }

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs, class_weight=class_weights)

    Path("models").mkdir(exist_ok=True)
    model.save(Path("models") / f"{args.module}.keras")
    Path("models").joinpath(f"{args.module}.json").write_text(json.dumps(class_names))
    print(f"Saved models/{args.module}.keras with classes: {class_names}")


if __name__ == "__main__":
    main()
