# **🔍 Zomato Restaurant Price Predictor**
A machine learning-powered web application that predicts the approximate cost for two people at a restaurant listed on Zomato, based on inputs like location, cuisine, rating, and more. Built using Streamlit and trained using optimized ensemble models.

## **📊 Overview**
This project provides a full end-to-end pipeline from data preprocessing to model deployment via a Streamlit web interface. It allows users to input restaurant features and returns a cost prediction using a trained regression model.

## **🛠️ Key Components**
📋 Data Cleaning, Preprocessing & Model Training
The script optimised_zomato_predictor.py performs:

Missing value treatment
Encoding categorical variables
Feature engineering
Outlier handling
Model training with hyperparameter tuning (Bayesian Optimization)

Ensemble models like XGBoost, Gradient Boosting, and Random Forest are used for robust prediction.
## **💻 Web App Interface**
The app.py script uses Streamlit to:

Accept user inputs (location, cuisine, etc.)
Load the saved ML model (zomato_model.pkl)
Display the predicted cost for two people


## **🚀 Getting Started**
#### 📦 1. Install Dependencies
Make sure Python 3.7+ is installed. Then, run:
bashpip install pandas numpy scikit-learn xgboost skopt joblib streamlit
#### **📂 2. Prepare Dataset**

Download or acquire the Zomato dataset.
Place it in the appropriate folder (e.g., data/).
Update the path in optimised_zomato_predictor.py as needed.

#### **⚙️ 3. Train the Model**
Train and save the model using:
bashpython optimised_zomato_predictor.py
This will save the model as zomato_model.pkl.
#### **🌐 4. Launch Web App**
Run the Streamlit application with:
bashstreamlit run app.py

#### **🧮 Model Overview**

Algorithms Used:

XGBoost Regressor
Random Forest Regressor
Gradient Boosting Regressor


Optimization:

Hyperparameters tuned using Bayesian Optimization (skopt)
Ensemble model stacking for better generalization


Evaluation Metrics:

R² Score
RMSE (Root Mean Squared Error)




### **📱 Web App Interface**

Input features: Restaurant Name, Location, Cuisine, Online Order, Table Booking, Ratings, Votes.
Output: Predicted cost for two.


### **🌟 Real-World Applications**

Helps users estimate affordability before choosing a restaurant.
Assists businesses in adjusting pricing strategies.
Benefits food delivery platforms in personalized filtering.
Useful for market researchers to analyze pricing trends.


### **📚 Tech Stack**

Languages & Tools: Python, Streamlit
Libraries: pandas, numpy, scikit-learn, xgboost, skopt, joblib
Modeling: Supervised Regression, Ensemble Learning
Tuning: Bayesian Optimization


### **📈 Results**

Achieved R² > 0.85
Reduced RMSE through ensemble modeling and fine-tuning
Generalized well on unseen data


### **🔮 Future Improvements**

Use geolocation or zip codes for enhanced prediction.
Incorporate menu-level and user-level preferences.
Add model monitoring via MLflow or similar tools.
Deploy with Docker + CI/CD pipelines for production.


### **👥 Contributing**
Feel free to fork the repo and contribute. Suggestions, issues, and pull requests are welcome!

### **📄 License**
This project is licensed under the MIT License - see the LICENSE file for details.

### **⭐ Show your support**
If you found this project useful, consider starring the repo!
