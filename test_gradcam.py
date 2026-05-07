import tensorflow as tf
import numpy as np

def build_test():
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    base.trainable = False  # SIMULATE NON-TRAINABLE BASE MODEL!
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    
    inner_model = base
    conv_layer = inner_model.get_layer("out_relu")
    
    inner_grad_model = tf.keras.models.Model(
        inner_model.inputs,
        [conv_layer.output, inner_model.output]
    )
    
    new_inputs = tf.keras.Input(shape=(224, 224, 3))
    conv_outputs, x = inner_grad_model(new_inputs)
    
    idx = model.layers.index(inner_model)
    for layer in model.layers[idx+1:]:
        x = layer(x)
        
    grad_model = tf.keras.models.Model(new_inputs, [conv_outputs, x])
    
    preprocessed = tf.convert_to_tensor(np.random.rand(1, 224, 224, 3).astype(np.float32))
    
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(preprocessed)
        class_channel = preds[:, tf.argmax(preds[0])]
        
    grads = tape.gradient(class_channel, conv_outs)
    print("Grads without watch:", grads is None)
    
    with tf.GradientTape() as tape:
        tape.watch(preprocessed)
        conv_outs, preds = grad_model(preprocessed)
        class_channel = preds[:, tf.argmax(preds[0])]
        
    grads = tape.gradient(class_channel, conv_outs)
    print("Grads with watch(preprocessed):", grads is None)

build_test()
