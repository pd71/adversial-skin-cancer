"""RASC-Net: Residual Attention Skin Cancer Network.

An original, custom CNN architecture designed for HAM10000 dermoscopy classification.

Key Design Elements:
- Custom Residual Blocks (Skip Connections) to stabilize gradient flow and improve multi-scale feature reuse.
- Convolutional Block Attention Module (CBAM) for joint Channel & Spatial Feature Localization.
- SpatialDropout2D & Regularized Classification Head to reduce co-adaptation across feature channels.
- MixUp Data Augmentation and Label Smoothing (0.1) for smoother decision boundaries and reduced overconfidence.
- Empirical Adversarial Robustness Benchmarking against FGSM, PGD, and CW attacks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def mixup_data(x_batch: np.ndarray, y_batch: np.ndarray, alpha: float = 0.2) -> tuple:
    """Apply MixUp Data Augmentation across a batch of images and one-hot labels.
    
    Linearly interpolates pairs of training samples:
    x_mix = lambda * x_i + (1 - lambda) * x_j
    y_mix = lambda * y_i + (1 - lambda) * y_j
    """
    batch_size = len(x_batch)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    index = np.random.permutation(batch_size)
    x_mix = lam * x_batch + (1.0 - lam) * x_batch[index]
    y_mix = lam * y_batch + (1.0 - lam) * y_batch[index]

    return x_mix, y_mix


def channel_attention_block(input_feature: tf.Tensor, ratio: int = 16) -> tf.Tensor:
    """Channel Attention Module of CBAM.
    
    Uses Global Average Pooling and Global Max Pooling passed through a shared MLP
    to weight feature channels based on their informativeness.
    """
    channel = input_feature.shape[-1]
    reduction = max(8, channel // ratio)
    shared_dense_1 = layers.Dense(
        reduction,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )
    shared_dense_2 = layers.Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=False,
    )

    # Average Pooling Branch
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_out = shared_dense_2(shared_dense_1(avg_pool))

    # Max Pooling Branch
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_out = shared_dense_2(shared_dense_1(max_pool))

    cbam_feature = layers.Add()([avg_out, max_out])
    cbam_feature = layers.Activation("sigmoid")(cbam_feature)

    return layers.Multiply()([input_feature, cbam_feature])


def spatial_attention_block(input_feature: tf.Tensor, kernel_size: int = 7) -> tf.Tensor:
    """Spatial Attention Module of CBAM.
    
    Computes average and max statistics across the channel dimension and applies a 
    convolutional layer to highlight lesion spatial regions.
    """
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])

    cbam_feature = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)

    return layers.Multiply()([input_feature, cbam_feature])


def cbam_block(input_feature: tf.Tensor, ratio: int = 16, kernel_size: int = 7) -> tf.Tensor:
    """Complete CBAM Attention Block combining Channel and Spatial Attention sequentially."""
    x = channel_attention_block(input_feature, ratio=ratio)
    x = spatial_attention_block(x, kernel_size=kernel_size)
    return x


def residual_block(
    input_tensor: tf.Tensor,
    filters: int,
    strides: int = 1,
    l2_reg: float = 1e-4,
) -> tf.Tensor:
    """Custom Residual Block with Pre-Activation Batch Normalization & L2 Regularization.
    
    Skip connections improve multi-scale feature reuse and gradient flow during backpropagation.
    """
    shortcut = input_tensor

    # First Conv Layer
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,
    )(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second Conv Layer
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)

    # Shortcut Projection if dimensions/strides change
    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(l2_reg),
            use_bias=False,
        )(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add Shortcut connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_rasc_net(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 7,
    l2_reg: float = 1e-4,
) -> models.Model:
    """Build RASC-Net (Residual Attention Skin Cancer Network) Model.
    
    Parameters:
    -----------
    input_shape : tuple, default=(224, 224, 3)
        Dimensions of input dermoscopy images.
    num_classes : int, default=7
        Number of output HAM10000 skin lesion categories.
    l2_reg : float, default=1e-4
        L2 weight decay multiplier for convolutional layers.
        
    Returns:
    --------
    tf.keras.models.Model
        Production-ready RASC-Net model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 1. Stem Conv Block
    x = layers.Conv2D(
        filters=32,
        kernel_size=(7, 7),
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(l2_reg),
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same", name="stem_pool")(x)

    # 2. Stage 1: Low-Level Spatial Feature Extraction (56x56)
    x = residual_block(x, filters=32, strides=1, l2_reg=l2_reg)
    x = residual_block(x, filters=32, strides=1, l2_reg=l2_reg)
    x = layers.SpatialDropout2D(rate=0.1, name="spatial_drop_1")(x)

    # 3. Stage 2: Mid-Level Texture Extraction + Attention (28x28)
    x = residual_block(x, filters=64, strides=2, l2_reg=l2_reg)
    x = residual_block(x, filters=64, strides=1, l2_reg=l2_reg)
    x = cbam_block(x, ratio=16, kernel_size=7)
    x = layers.SpatialDropout2D(rate=0.15, name="spatial_drop_2")(x)

    # 4. Stage 3: High-Level Lesion Boundary Extraction (14x14)
    x = residual_block(x, filters=128, strides=2, l2_reg=l2_reg)
    x = residual_block(x, filters=128, strides=1, l2_reg=l2_reg)
    x = cbam_block(x, ratio=16, kernel_size=7)
    x = layers.SpatialDropout2D(rate=0.2, name="spatial_drop_3")(x)

    # 5. Stage 4: Deep Representation Aggregation (7x7)
    x = residual_block(x, filters=256, strides=2, l2_reg=l2_reg)
    x = residual_block(x, filters=256, strides=1, l2_reg=l2_reg)

    # 6. Global Average Pooling Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        256,
        kernel_initializer="he_normal",
        use_bias=False,
        name="fc1",
    )(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(rate=0.4, name="head_dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="RASC_Net")
    
    # Compile with Label Smoothing Loss for empirical robustness
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    
    return model


# Backward compatibility alias
build_robust_skin_net = build_rasc_net


if __name__ == "__main__":
    model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
    model.summary()
