"""
============================================
BREAST CANCER PREDICTION SYSTEM
Flask Web Application
============================================
This application provides a web interface for predicting
whether a breast tumor is benign or malignant using a
trained Neural Network model.

Algorithm: Neural Networks (Deep Learning)
Model Persistence: Joblib
Features Used: 5 (radius_mean, texture_mean, perimeter_mean, 
               area_mean, smoothness_mean)

DISCLAIMER: For educational purposes only.
Not for actual medical diagnosis.
============================================
"""

import os
import numpy as np
import joblib
from flask import Flask, render_template, request

# ============================================
# FLASK APPLICATION SETUP
# ============================================
flask_app = Flask(__name__)

# ============================================
# MODEL CONFIGURATION
# ============================================

# Path to the trained model (saved using Joblib)
MODEL_PATH = "model/breast_cancer_model.pkl"

# ============================================
# LOAD MODEL AND SCALER
# ============================================

# Attempt to load the trained model
if os.path.exists(MODEL_PATH):
    # Load the model data dictionary containing model, scaler, and metadata
    model_data = joblib.load(MODEL_PATH)
    keras_model = model_data['model']
    data_scaler = model_data['scaler']
    print(f"✓ Model and scaler loaded successfully from: {MODEL_PATH}")
    print(f"  - Model Accuracy: {model_data['metrics']['accuracy']:.4f}")
    print(f"  - Model Precision: {model_data['metrics']['precision']:.4f}")
else:
    # Fallback: Try loading old format for backward compatibility
    print("⚠ Primary model not found. Checking for legacy format...")
    
    DATA_SCALER_PATH = "scaler.save"
    if os.path.exists(DATA_SCALER_PATH):
        data_scaler = joblib.load(DATA_SCALER_PATH)
        print(f"✓ Scaler loaded from: {DATA_SCALER_PATH}")
    else:
        data_scaler = None
        print("⚠ No scaler found")
    
    # Try loading old model.h5 if it exists
    OLD_MODEL_PATH = "model/model.h5"
    if os.path.exists(OLD_MODEL_PATH):
        # Import TensorFlow only if needed (conditional import for backward compatibility)
        from tensorflow.keras.models import load_model  # type: ignore
        keras_model = load_model(OLD_MODEL_PATH)
        print(f"✓ Loaded model from legacy format: {OLD_MODEL_PATH}")
    else:
        raise FileNotFoundError(
            "❌ Model file not found. Please run model_building.ipynb first to generate the model."
        )

# ============================================
# ROUTE: HOME PAGE
# ============================================

@flask_app.route("/")
def display_home_page():
    """
    Renders the home page with the prediction form.
    
    Returns:
        HTML template with empty form
    """
    return render_template("index.html", show_result=False)

# ============================================
# ROUTE: GENERATE PREDICTION
# ============================================

@flask_app.route("/generate_prediction", methods=["POST"])
def handle_prediction():
    """
    Processes user input and generates a prediction.
    
    This function:
    1. Extracts the 5 tumor features from the form
    2. Preprocesses the data (reshaping and scaling)
    3. Passes data through the neural network
    4. Interprets the prediction result
    5. Returns formatted results to the UI
    
    Returns:
        HTML template with prediction results or error message
    """
    try:
        # ============================================
        # STEP 1: EXTRACT INPUT DATA
        # ============================================
        # Retrieve form data and convert to float
        # The model expects exactly 5 features in this order
        input_data_points = [
            float(request.form["mean_radius"]),      # Feature 1
            float(request.form["mean_texture"]),     # Feature 2
            float(request.form["mean_perimeter"]),   # Feature 3
            float(request.form["mean_area"]),        # Feature 4
            float(request.form["mean_smoothness"])   # Feature 5
        ]
        
        # ============================================
        # STEP 2: PREPROCESS DATA
        # ============================================
        # Convert to numpy array and reshape for model input
        # Shape: (1, 5) - one sample with 5 features
        processed_features = np.array(input_data_points).reshape(1, -1)
        
        # Apply feature scaling using the same scaler from training
        # This ensures input data is on the same scale as training data
        if data_scaler:
            processed_features = data_scaler.transform(processed_features)
        else:
            print("⚠ Warning: No scaler applied. Predictions may be inaccurate.")
        
        # ============================================
        # STEP 3: GENERATE PREDICTION
        # ============================================
        # Pass preprocessed data through the neural network
        # Output: probability of class 1 (benign)
        model_prediction = keras_model.predict(processed_features, verbose=0)
        malignancy_probability = float(model_prediction[0][0])
        
        # ============================================
        # STEP 4: INTERPRET RESULTS
        # ============================================
        # In sklearn breast_cancer dataset:
        # - 0 = Malignant (cancerous)
        # - 1 = Benign (non-cancerous)
        # Our model outputs probability of class 1 (benign)
        
        if malignancy_probability > 0.5:
            # High probability = Benign (Non-cancerous)
            prediction_class = "Benign"
            prediction_type = "benign"
            prediction_description = "Non-cancerous"
            prediction_confidence = round(malignancy_probability * 100, 2)
        else:
            # Low probability = Malignant (Cancerous)
            prediction_class = "Malignant"
            prediction_type = "malignant"
            prediction_description = "Cancerous"
            prediction_confidence = round((1 - malignancy_probability) * 100, 2)
        
        # ============================================
        # STEP 5: RETURN RESULTS
        # ============================================
        return render_template(
            "index.html",
            show_result=True,
            prediction_class=prediction_class,
            prediction_type=prediction_type,
            prediction_description=prediction_description,
            prediction_confidence=prediction_confidence
        )
        
    except ValueError as ve:
        # Handle invalid input data (non-numeric values)
        error_message = f"Invalid input: Please enter valid numeric values. Details: {str(ve)}"
        return render_template("index.html", show_result=False, error_message=error_message)
        
    except KeyError as ke:
        # Handle missing form fields
        error_message = f"Missing required field: {str(ke)}. Please fill all fields."
        return render_template("index.html", show_result=False, error_message=error_message)
        
    except Exception as error:
        # Handle any other unexpected errors
        error_message = f"An unexpected error occurred: {str(error)}"
        print(f"❌ Error during prediction: {error}")
        return render_template("index.html", show_result=False, error_message=error_message)

# ============================================
# APPLICATION ENTRY POINT
# ============================================

if __name__ == "__main__":
    """
    Runs the Flask application in debug mode.
    
    Debug mode features:
    - Auto-reload on code changes
    - Detailed error messages
    - Interactive debugger
    
    Note: Disable debug mode in production deployment
    """
    print("\n" + "="*50)
    print("🚀 STARTING BREAST CANCER PREDICTION SYSTEM")
    print("="*50)
    print("📊 Algorithm: Neural Networks")
    print("💾 Model: Joblib (.pkl)")
    print("🔬 Features: 5 tumor characteristics")
    print("="*50 + "\n")
    
    # Start the Flask development server
    flask_app.run(debug=True)

# Expose the Flask app instance for Gunicorn
app = flask_app