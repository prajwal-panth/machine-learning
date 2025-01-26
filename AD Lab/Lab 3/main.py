from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template'))

# Get the directory of the current script (app.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'regression_model.pkl')

# Debug paths
# print("Current script directory:", script_dir)
# print("Model path:", model_path)

# Load the trained model
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form/index.html
        user_input = [float(x) for x in request.form.values()]
        
        # Ensure input is in the right shape (2D array)
        input_array = np.array(user_input).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Return result
        return render_template('index.html', predictions=f"Performance Index: {prediction[0]:.2f}")
    
    except Exception as e:
        # Log the error and display it on the page
        print(f"Error during prediction: {e}")
        return render_template('index.html', predictions=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)