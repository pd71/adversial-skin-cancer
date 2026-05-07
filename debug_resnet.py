import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))
import config as cfg

def debug_resnet():
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
    model = tf.keras.models.load_model(resnet_path)
    print("Model loaded.")
    
    last_conv_layer_name = "conv5_block3_3_conv"
    
    inner_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break
            
    print("Inner model found:", inner_model.name)
    conv_layer = inner_model.get_layer(last_conv_layer_name)
    
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
    print("grad_model constructed.")
    
    preprocessed = tf.convert_to_tensor(np.random.rand(1, 224, 224, 3).astype(np.float32))
    
    with tf.GradientTape() as tape:
        tape.watch(preprocessed)
        conv_outs, preds = grad_model(preprocessed)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    print("Tracing gradient...")
    grads = tape.gradient(class_channel, conv_outs)
    print("Grads is None?", grads is None)

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))
    debug_resnet()
