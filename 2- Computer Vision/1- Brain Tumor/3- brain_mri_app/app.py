import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import base64
import tempfile
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
# Initialize Flask
app = Flask(__name__)

# Load your classification model
CLASSIFICATION_MODEL_PATH = "models/best_model_brain_tumor_classification_val_0.985.keras"
# Add path for segmentation model
SEGMENTATION_MODEL_PATH = "models/segmentation_brain_tumor_89.2.keras"


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(
        y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Dice Loss


def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)


# Try to load the classification model
try:
    classification_model = tf.keras.models.load_model(
        CLASSIFICATION_MODEL_PATH)
    print("✅ Classification model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load classification model: {e}")
    classification_model = None

# Try to load the segmentation model
try:
    segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH,
                                                    custom_objects={
                                                        "dice_loss": dice_loss,
                                                        "dice_coef": dice_coef,
                                                        "iou_coef": iou_coef}
                                                    )
    print("✅ Segmentation model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load segmentation model: {e}")
    segmentation_model = None

# Helper function to convert PIL image to base64 for display


def pil_to_base64(pil_image):
    """Convert PIL Image to base64 string for HTML display"""
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def predict_brain_mri(img: Image.Image):
    if classification_model is None:
        return "❌ Model not loaded - please check model path"

    try:
        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = classification_model.predict(img_array, verbose=0)[0][0]

        if prediction > 0.5:
            label = "🧠 Tumor detected"
            prob = prediction
        else:
            label = "✅ No tumor detected"
            prob = 1 - prediction

        return f"{label} (Confidence: {prob:.2f})"

    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

# Placeholder segmentation function


def segment_brain_tumor(img: Image.Image):
    if segmentation_model is None:
        return {
            "message": "❌ Segmentation model not loaded",
            "details": "Please check the segmentation model path",
            "note": "No segmentation performed"
        }

    try:
        # Convert PIL Image to numpy array and then to BGR for OpenCV
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Store original dimensions for later resizing
        original_height, original_width = img_bgr.shape[:2]

        # Preprocess for model (resize to expected input size)
        img_resized = cv2.resize(img_bgr, (256, 256))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Predict mask
        predicted_mask = segmentation_model.predict(img_batch, verbose=0)
        mask = np.squeeze(predicted_mask) > 0.5  # Binary mask

        # Resize mask back to original image dimensions
        mask_resized = cv2.resize(mask.astype(
            np.uint8), (original_width, original_height))
        mask_resized = mask_resized.astype(bool)

        # Create glass-blue overlay
        glass_blue_bgr = np.array(
            [255, 100, 0], dtype=np.uint8)  # Blue overlay in BGR

        # Apply overlay
        overlay = img_bgr.copy()
        overlay[mask_resized] = glass_blue_bgr
        alpha = 0.4
        blended_img = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

        # Convert back to RGB for display
        result_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)

        # Convert result to base64 for rendering
        result_pil = Image.fromarray(result_rgb)
        result_base64 = pil_to_base64(result_pil)

        return {
            "message": "🎯 Segmentation analysis complete",
            "details": "Tumor regions highlighted in blue overlay",
            "segmentation_image": result_base64
        }

    except Exception as e:
        return {
            "message": "❌ Error during segmentation",
            "details": str(e),
            "note": "Check logs for more info"
        }


# Routes


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    mode = request.form.get("mode")
    file = request.files.get("file")

    if not file or file.filename == "":
        return render_template("index.html", error="⚠️ No file uploaded", tab=mode)

    try:
        # Read and validate image
        img = Image.open(file.stream).convert("RGB")

        # Convert image to base64 for display persistence
        img_base64 = pil_to_base64(img)

        if mode == "classification":
            result = predict_brain_mri(img)
            return render_template("index.html",
                                   result=result,
                                   tab="classification",
                                   uploaded_image=img_base64)

        elif mode == "segmentation":
            seg_result = segment_brain_tumor(img)
            return render_template("index.html",
                                   seg_result=seg_result,
                                   tab="segmentation",
                                   uploaded_image=img_base64)

        else:
            return render_template("index.html", error="Invalid mode selected", tab="classification")

    except Exception as e:
        return render_template("index.html", error=f"Error processing image: {str(e)}", tab=mode)


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "classification_model_loaded": classification_model is not None,
        "segmentation_model_loaded": segmentation_model is not None
    })


if __name__ == "__main__":
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    print("🚀 Starting Brain MRI Analyzer...")
    print("📁 Make sure your model is in the 'models' directory")
    print("📁 Make sure your HTML template is in the 'templates' directory")

    app.run(debug=True, host='0.0.0.0', port=5000)
