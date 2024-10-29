# Zomato Restaurant Price Predictor

## Overview

A web application that predicts the cost for two people at a Zomato restaurant based on user-provided details like location, cuisine, and other features. Built using Streamlit and powered by machine learning.

## Key Components

- **Data Cleaning, Preprocessing and Model Training:**  
  The script `optimised_zomato_predictor.py` handles data cleaning, feature engineering, and preprocessing. This script also trains a machine learning model using techniques like hyperparameter tuning and ensemble methods to achieve optimal performance.

- **Web App:**  
  The `app.py` script leverages Streamlit to create a user-friendly interface for inputting restaurant details and viewing predicted costs.

## Usage

1. **Install Dependencies:**  
   Ensure you have Python 3.7+ and required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn xgboost skopt joblib streamlit
   ```

2. **Prepare Data:**  
   - Download a Zomato dataset and place it in the appropriate directory.
   - Update the data path in the `optimised_zomato_predictor.py` script.

3. **Train the Model:**  
   - Run the following command to train the model and save it as `zomato_model.pkl`:
   ```bash
   python optimised_zomato_predictor.py
   ```

4. **Run the Web App:**  
   - Execute the following command to launch the web app in your browser:
   ```bash
   streamlit run app.py
   ```

## Web App Interface

- Input restaurant details like location, cuisine, and other features.
- Click the "Predict" button to get an estimated cost for two people.

## Technical Details

- The machine learning model employs advanced techniques like Gradient Boosting, Random Forest, and XGBoost.
- The Streamlit web app provides a seamless user experience.

Feel free to contribute to this project and enhance its capabilities!
