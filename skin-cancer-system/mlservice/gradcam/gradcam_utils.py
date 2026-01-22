import tensorflow as tf
import numpy as np
import cv2

def compute_gradcam(img_array, model, layer_name):
    """
    Compute Grad-CAM by computing gradients of model output with respect to 
    intermediate layer activations. Works with any model.
    """
    print(f"🔥 Generating Grad-CAM using layer: {layer_name}")

    # Get the target convolutional layer
    target_layer = model.get_layer(layer_name)
    
    # Convert input to tensor for gradient computation
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # Watch the activation of the target layer
        # We'll create intermediate outputs by passing through layers
        activation = img_array
        
        # Forward pass through all layers up to and including target layer
        layer_found = False
        for layer in model.layers:
            activation = layer(activation, training=False)
            
            if layer.name == layer_name:
                # Watch the activation at this layer
                tape.watch(activation)
                layer_found = True
                last_conv_activation = activation
                break
        
        if not layer_found:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Continue forward pass through remaining layers to get final prediction
        final_output = last_conv_activation
        found_target = False
        for layer in model.layers:
            if found_target:
                final_output = layer(final_output, training=False)
            elif layer.name == layer_name:
                found_target = True
        
        # Compute loss (prediction score)
        loss = final_output[:, 0]

    # Compute gradients of loss with respect to conv layer activation
    grads = tape.gradient(loss, last_conv_activation)
    
    if grads is None:
        raise ValueError("Failed to compute gradients - this may indicate an issue with the layer")

    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the activations
    activation_val = last_conv_activation[0]
    heatmap = tf.reduce_sum(activation_val * pooled_grads, axis=-1)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(image_path, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay
