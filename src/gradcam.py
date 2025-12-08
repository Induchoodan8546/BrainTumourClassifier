import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

IMG_SIZE = 224

# Load class labels (same order as during training)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img

def generate_gradcam(model, img_tensor, last_conv_layer_name="Conv_1"):
    # Get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Model for getting outputs
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply gradients with feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap, class_idx.numpy()

def overlay_heatmap(heatmap, original_img, intensity=0.5):
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original = np.array(original_img)

    overlayed = cv2.addWeighted(original, 1 - intensity, heatmap_color, intensity, 0)
    return overlayed

def gradcam_for_image(model_path, img_path):
    # Load model
    model = load_model(model_path)

    # Preprocess image
    img_tensor, original_img = load_and_preprocess(img_path)

    # Generate Grad-CAM
    heatmap, class_idx = generate_gradcam(model, img_tensor)

    # Overlay heatmap
    result = overlay_heatmap(heatmap, original_img)

    # Show results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Overlay ({CLASS_NAMES[class_idx]})")
    plt.imshow(result)
    plt.axis("off")

    plt.show()


# Example usage:
#gradcam_for_image("models/best_model.h5", "data/test/glioma/Tr-gl_1267.jpg ")
