import numpy as np
import cv2
import tensorflow as tf
import json
import base64
from PIL import Image
import os

# Load model
config_path = "models/skin_cancer_cnn.keras/config.json"
weights_path = "models/skin_cancer_cnn.keras/model.weights.h5"

with open(config_path, 'r') as f:
    model_config = json.load(f)

model = tf.keras.Sequential.from_config(model_config['config'])
model.load_weights(weights_path)

print("=" * 60)
print("MODEL STRUCTURE")
print("=" * 60)
model.summary()

print("\n" + "=" * 60)
print("MODEL LAYERS")
print("=" * 60)
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - {layer.__class__.__name__}")

# Find conv layers
print("\n" + "=" * 60)
print("CONVOLUTIONAL LAYERS")
print("=" * 60)
conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
if conv_layers:
    for layer in conv_layers:
        print(f"  {layer.name}: output_shape = {layer.output.shape}")
    print(f"\nLast conv layer: {conv_layers[-1].name}")
else:
    print("NO CONVOLUTIONAL LAYERS FOUND!")

# Create a test image
print("\n" + "=" * 60)
print("CREATING TEST IMAGE")
print("=" * 60)
test_img = np.random.rand(224, 224, 3) * 255
test_img = test_img.astype(np.uint8)
print(f"Test image shape: {test_img.shape}, dtype: {test_img.dtype}")

# Normalize
arr = np.array(test_img) / 255.0
arr_batch = np.expand_dims(arr, 0)
print(f"Batch shape: {arr_batch.shape}, min: {arr_batch.min()}, max: {arr_batch.max()}")

# Make prediction
print("\n" + "=" * 60)
print("MAKING PREDICTION")
print("=" * 60)
pred = model.predict(arr_batch, verbose=0)
print(f"Prediction output shape: {pred.shape}")
print(f"Prediction value: {pred}")

# Try GradCAM
print("\n" + "=" * 60)
print("COMPUTING GRADCAM")
print("=" * 60)

if not conv_layers:
    print("ERROR: No convolutional layers found!")
else:
    last_conv_layer = conv_layers[-1]
    print(f"Using layer: {last_conv_layer.name}")
    print(f"Layer output shape: {last_conv_layer.output.shape}")
    
    # Create gradient model
    grad_model = tf.keras.models.Model(
        model.inputs,
        [last_conv_layer.output, model.outputs[0]]
    )
    
    img_tensor = tf.cast(arr_batch, tf.float32)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        last_conv_output_val, pred_val = grad_model(img_tensor, training=False)
        class_channel = pred_val[0, 0] if pred_val.shape[1] > 1 else pred_val[0]
    
    grads = tape.gradient(class_channel, last_conv_output_val)
    
    print(f"\nLast conv output shape: {last_conv_output_val.shape}")
    print(f"Last conv output - min: {tf.reduce_min(last_conv_output_val).numpy()}, max: {tf.reduce_max(last_conv_output_val).numpy()}")
    
    if grads is None:
        print("ERROR: Gradients are None!")
    else:
        print(f"Gradients shape: {grads.shape}")
        print(f"Gradients - min: {tf.reduce_min(grads).numpy()}, max: {tf.reduce_max(grads).numpy()}")
        
        # Compute heatmap
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        pooled_grads_np = pooled_grads.numpy()
        print(f"Pooled grads shape: {pooled_grads_np.shape}")
        print(f"Pooled grads - min: {pooled_grads_np.min()}, max: {pooled_grads_np.max()}, mean: {pooled_grads_np.mean()}")
        
        last_conv_output_np = last_conv_output_val[0].numpy()
        print(f"Conv output shape: {last_conv_output_np.shape}")
        
        # Compute heatmap
        heatmap = np.zeros((last_conv_output_np.shape[0], last_conv_output_np.shape[1]))
        for i in range(len(pooled_grads_np)):
            heatmap += pooled_grads_np[i] * last_conv_output_np[:, :, i]
        
        heatmap = np.maximum(heatmap, 0)
        print(f"Heatmap (after ReLU) - min: {heatmap.min()}, max: {heatmap.max()}, mean: {heatmap.mean()}")
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        print(f"Heatmap (normalized) - min: {heatmap.min()}, max: {heatmap.max()}, mean: {heatmap.mean()}")
        
        # Resize
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        print(f"Heatmap (resized) - shape: {heatmap_resized.shape}, min: {heatmap_resized.min()}, max: {heatmap_resized.max()}, mean: {heatmap_resized.mean()}")
        
        # Create overlay
        img_bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
        
        # Save overlay
        cv2.imwrite("test_overlay.png", overlay)
        print("Saved test_overlay.png")
        
        # Also save individual components
        cv2.imwrite("test_heatmap_uint8.png", heatmap_uint8)
        cv2.imwrite("test_heatmap_colored.png", heatmap_colored)
        print("Saved test_heatmap_uint8.png and test_heatmap_colored.png")
        
        # Save to base64
        _, buffer = cv2.imencode(".png", overlay)
        gradcam_base64 = base64.b64encode(buffer).decode("utf-8")
        print(f"Base64 length: {len(gradcam_base64)}")
        
        # Save base64 to file
        with open("test_gradcam_base64.txt", "w") as f:
            f.write(gradcam_base64[:100] + "...")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
