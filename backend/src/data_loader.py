"""Data loading and tf.data preprocessing pipeline for HAM10000."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_FILENAME_COLUMN,
    IMAGE_SIZE,
    IMAGES_PART_1_DIR,
    IMAGES_PART_2_DIR,
    METADATA_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
)


AUTOTUNE = tf.data.AUTOTUNE


def _scan_image_paths() -> Dict[str, str]:
    """Build image_id -> absolute image path mapping from both image folders."""
    image_map: Dict[str, str] = {}
    image_dirs = [IMAGES_PART_1_DIR, IMAGES_PART_2_DIR]

    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        for image_path in image_dir.glob("*.jpg"):
            image_map[image_path.stem] = str(image_path)

    return image_map


def _resolve_class_names(labels: pd.Series) -> List[str]:
    """Resolve class order from config if valid, else use sorted label names."""
    label_set = set(labels.unique().tolist())
    config_set = set(CLASS_NAMES)
    if label_set.issubset(config_set):
        return [label for label in CLASS_NAMES if label in label_set]
    return sorted(label_set)


def _encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """Encode string labels into integer indices and return label mappings."""
    class_names = _resolve_class_names(df[TARGET_COLUMN])
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    encoded_df = df.copy()
    encoded_df["label"] = encoded_df[TARGET_COLUMN].map(label_to_index).astype(int)
    return encoded_df, label_to_index, index_to_label


def _compute_class_weights(y_train: np.ndarray, num_classes: int) -> Dict[int, float]:
    """Compute sklearn-balanced class weights for training labels."""
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def _split_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified 80/10/10 split using two train_test_split calls."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df[TARGET_COLUMN],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df[TARGET_COLUMN],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _augment_image(image: tf.Tensor) -> tf.Tensor:
    """Apply comprehensive medical dermoscopy augmentations for training.
    
    Includes 4-way spatial orientation (horizontal/vertical flips and 90-degree rotations)
    plus subtle contrast, brightness, and saturation jittering for illumination invariance.
    """
    # 1. Spatial Rotations & Flips (Dermoscopy images are orientation-invariant)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random 90-degree rotations (0, 90, 180, 270 degrees)
    k_rot = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k_rot)

    # 2. Photometric & Illumination Augmentations
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)

    return tf.clip_by_value(image, 0.0, 1.0)


def get_tta_augmented_variants(image: tf.Tensor) -> List[tf.Tensor]:
    """Generate 4 Test-Time Augmentation (TTA) variants of a single preprocessed image (3D tensor HxWxC).
    
    Returns:
    - Original image
    - Horizontally flipped image
    - Vertically flipped image
    - 180-degree rotated image
    """
    img_orig = tf.clip_by_value(image, 0.0, 1.0)
    img_hflip = tf.image.flip_left_right(img_orig)
    img_vflip = tf.image.flip_up_down(img_orig)
    img_rot180 = tf.image.rot90(img_orig, k=2)
    return [img_orig, img_hflip, img_vflip, img_rot180]



def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load image from file path, decode, resize, and normalize to [0, 1]."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _get_model_preprocess_fn(model_name: str):
    """Return model-specific preprocess function, or None if not requested."""
    if model_name is None:
        return None

    model_name = model_name.lower()
    if model_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if model_name == "resnet50":
        return tf.keras.applications.resnet.preprocess_input
    if model_name == "efficientnetb0":
        return tf.keras.applications.efficientnet.preprocess_input
    raise ValueError(f"Unsupported model_name: {model_name}")


def _create_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    model_name: str = None,
    training: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """Create optimized tf.data dataset for train/val/test splits."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    if training:
        dataset = dataset.map(
            lambda img, lbl: (_augment_image(img), lbl),
            num_parallel_calls=AUTOTUNE,
        )

    preprocess_fn = _get_model_preprocess_fn(model_name)
    if preprocess_fn is not None:
        dataset = dataset.map(
            lambda img, lbl: (preprocess_fn(img * 255.0), lbl),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.map(
        lambda img, lbl: (img, tf.one_hot(tf.cast(lbl, tf.int32), depth=num_classes)),
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def load_ham10000_data(batch_size: int = BATCH_SIZE, model_name: str = None) -> Dict[str, object]:
    """Load HAM10000 metadata, split data, encode labels, and build tf.data pipelines."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    metadata_df = pd.read_csv(METADATA_PATH)
    if IMAGE_FILENAME_COLUMN not in metadata_df.columns or TARGET_COLUMN not in metadata_df.columns:
        raise ValueError(
            f"Metadata must contain '{IMAGE_FILENAME_COLUMN}' and '{TARGET_COLUMN}' columns."
        )

    metadata_df = metadata_df[[IMAGE_FILENAME_COLUMN, TARGET_COLUMN]].copy()
    metadata_df[TARGET_COLUMN] = metadata_df[TARGET_COLUMN].astype(str)

    image_map = _scan_image_paths()
    metadata_df["image_path"] = metadata_df[IMAGE_FILENAME_COLUMN].map(image_map)
    metadata_df = metadata_df.dropna(subset=["image_path"]).reset_index(drop=True)

    encoded_df, label_to_index, index_to_label = _encode_labels(metadata_df)
    train_df, val_df, test_df = _split_dataframe(encoded_df)

    train_paths = train_df["image_path"].values
    val_paths = val_df["image_path"].values
    test_paths = test_df["image_path"].values

    train_labels = train_df["label"].values.astype(np.int32)
    val_labels = val_df["label"].values.astype(np.int32)
    test_labels = test_df["label"].values.astype(np.int32)

    class_weights = _compute_class_weights(
        y_train=train_labels,
        num_classes=len(label_to_index),
    )

    train_ds = _create_dataset(
        train_paths,
        train_labels,
        num_classes=len(label_to_index),
        model_name=model_name,
        training=True,
        batch_size=batch_size,
    )
    val_ds = _create_dataset(
        val_paths,
        val_labels,
        num_classes=len(label_to_index),
        model_name=model_name,
        training=False,
        batch_size=batch_size,
    )
    test_ds = _create_dataset(
        test_paths,
        test_labels,
        num_classes=len(label_to_index),
        model_name=model_name,
        training=False,
        batch_size=batch_size,
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "class_weights": class_weights,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    data_dict = load_ham10000_data()
    train_df = data_dict["train_df"]
    val_df = data_dict["val_df"]
    test_df = data_dict["test_df"]

    print("=== HAM10000 Data Loader Check ===")
    print(f"Train size: {len(train_df)}")
    print(f"Val size  : {len(val_df)}")
    print(f"Test size : {len(test_df)}")

    print("\nTrain class distribution:")
    print(train_df[TARGET_COLUMN].value_counts().sort_index())

    print("\nVal class distribution:")
    print(val_df[TARGET_COLUMN].value_counts().sort_index())

    print("\nTest class distribution:")
    print(test_df[TARGET_COLUMN].value_counts().sort_index())

    one_batch_images, one_batch_labels = next(iter(data_dict["train_ds"]))
    print("\nOne train batch shapes:")
    print(f"Images: {one_batch_images.shape}")
    print(f"Labels: {one_batch_labels.shape}")
