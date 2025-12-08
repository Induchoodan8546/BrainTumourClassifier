import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from data_loader import get_data_generators

# Load test data
_, _, test_gen = get_data_generators(
    "data/train",
    "data/val",
    "data/test"
)

# Load best model
model = tf.keras.models.load_model("models/best_model.h5")

# Predict
pred_probs = model.predict(test_gen)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = test_gen.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
target_names = list(test_gen.class_indices.keys())
print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=target_names))
