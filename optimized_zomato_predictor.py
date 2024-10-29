# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor
# import joblib
# from tqdm import tqdm

# class OptimizedZomatoPredictor:
#     def __init__(self):
#         self.model = None
#         self.label_encoders = {}
#         self.scaler = StandardScaler()
#         self.poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        
#     def load_and_clean_data(self, filepath):
#         """Load and perform initial cleaning of the dataset"""
#         print("Loading and cleaning data...")
        
#         # Load data
#         df = pd.read_csv(filepath)
        
#         # Remove redundant columns
#         cols_to_drop = ['url', 'phone', 'dish_liked', 'reviews_list', 'menu_item']
#         df.drop(columns=cols_to_drop, inplace=True)
        
#         # Clean rate column
#         df['rate'] = df['rate'].apply(lambda x: str(x).split('/')[0] if str(x) != 'nan' else np.nan)
#         df['rate'] = df['rate'].replace({'NEW': np.nan, '-': np.nan})
#         df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

#         # Clean cost column
#         df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',', '').astype(float)
        
#         # Drop duplicates and handle missing values
#         df.drop_duplicates(inplace=True)
#         df.dropna(inplace=True)
        
#         return df
    
#     def prepare_features(self, df, training=True):
#         """Prepare features with encoding and feature engineering"""
#         df = df.copy()
        
#         # Categorical columns for encoding
#         categorical_cols = ['online_order', 'book_table', 'location', 'rest_type', 
#                           'cuisines', 'listed_in(type)', 'listed_in(city)']
        
#         # Encode categorical variables
#         for col in categorical_cols:
#             if training:
#                 self.label_encoders[col] = LabelEncoder()
#                 df[col] = self.label_encoders[col].fit_transform(df[col])
#             else:
#                 df[col] = self.label_encoders[col].transform(df[col])
        
#         # Prepare feature matrix
#         feature_cols = categorical_cols + ['votes', 'rate']
#         X = df[feature_cols]
        
#         # Scale features
#         if training:
#             X = self.scaler.fit_transform(X)
#         else:
#             X = self.scaler.transform(X)
        
#         # Add polynomial features
#         if training:
#             X = self.poly.fit_transform(X)
#         else:
#             X = self.poly.transform(X)
        
#         return X
    
#     def train(self, df):
#         """Train the model with optimized hyperparameters"""
#         print("Preparing data for training...")
        
#         # Prepare features and target
#         X = self.prepare_features(df, training=True)
#         y = df['approx_cost(for two people)']
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Define hyperparameter search space
#         param_grid = {
#             'n_estimators': [200, 300, 400],
#             'learning_rate': [0.01, 0.1, 0.2],
#             'max_depth': [5, 6, 7, 8],
#             'min_child_weight': [1, 2, 3],
#             'subsample': [0.7, 0.8, 0.9, 1.0],
#             'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#             'alpha': [0, 0.1, 1, 2],
#             'lambda': [0, 0.1, 1, 2]
#         }
        
#         # Initialize base model
#         xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        
#         # Perform randomized search
#         print("Starting hyperparameter optimization...")
#         random_search = RandomizedSearchCV(
#             estimator=xgb_model,
#             param_distributions=param_grid,
#             n_iter=50,  # Increased number of iterations
#             cv=5,
#             scoring='neg_mean_squared_error',
#             verbose=1,
#             n_jobs=-1,
#             random_state=42
#         )
        
#         # Fit with progress bar
#         with tqdm(total=random_search.n_iter * random_search.cv) as pbar:
#             random_search.fit(X_train, y_train)
#             pbar.update(random_search.n_iter * random_search.cv)
        
#         # Get best model
#         self.model = random_search.best_estimator_
        
#         # Evaluate model
#         train_pred = self.model.predict(X_train)
#         test_pred = self.model.predict(X_test)
        
#         metrics = {
#             'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
#             'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
#             'train_r2': r2_score(y_train, train_pred),
#             'test_r2': r2_score(y_test, test_pred),
#             'best_params': random_search.best_params_
#         }
        
#         # Save the model
#         joblib.dump({
#             'model': self.model,
#             'label_encoders': self.label_encoders,
#             'scaler': self.scaler,
#             'poly': self.poly
#         }, 'zomato_model.pkl')
        
#         return metrics
    
#     def predict(self, df):
#         """Make predictions for new data"""
#         X = self.prepare_features(df, training=False)
#         return self.model.predict(X)
    
#     @classmethod
#     def load_model(cls, model_path):
#         """Load a trained model"""
#         saved_objects = joblib.load(model_path)
#         predictor = cls()
#         predictor.model = saved_objects['model']
#         predictor.label_encoders = saved_objects['label_encoders']
#         predictor.scaler = saved_objects['scaler']
#         predictor.poly = saved_objects['poly']
#         return predictor
# def main():
#     # Initialize predictor
#     predictor = OptimizedZomatoPredictor()
    
#     # Load and clean data
#     df = predictor.load_and_clean_data(r"C:\Users\khush\Downloads\ZOM\khush\zomato.csv")
#     print("\nData loaded and cleaned successfully!")
    
#     # Train model and get metrics
#     metrics = predictor.train(df)
    
#     # Print performance metrics
#     print("\nModel Performance:")
#     print(f"Training RMSE: {metrics['train_rmse']:.2f}")
#     print(f"Testing RMSE: {metrics['test_rmse']:.2f}")
#     print(f"Training R2 Score: {metrics['train_r2']:.3f}")
#     print(f"Testing R2 Score: {metrics['test_r2']:.3f}")
    
#     print("\nBest Parameters:")
#     for param, value in metrics['best_params'].items():
#         print(f"{param}: {value}")
    
#     # Example prediction
#     sample = df.iloc[[0]]
#     predicted_price = predictor.predict(sample)
#     print(f"\nSample Prediction:")
#     print(f"Actual Price: {sample['approx_cost(for two people)'].values[0]:.2f}")
#     print(f"Predicted Price: {predicted_price[0]:.2f}")

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class OptimizedZomatoPredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_and_clean_data(self, filepath):
        """Load and perform initial cleaning of the dataset"""
        print("Loading and cleaning data...")

        # Load data
        df = pd.read_csv(filepath)

        # Remove redundant columns
        cols_to_drop = ['url', 'phone', 'dish_liked', 'reviews_list', 'menu_item']
        df.drop(columns=cols_to_drop, inplace=True)

        # Clean rate column
        df['rate'] = df['rate'].apply(lambda x: str(x).split('/')[0] if str(x) != 'nan' else np.nan)
        df['rate'] = df['rate'].replace({'NEW': np.nan, '-': np.nan})
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

        # Clean cost column
        df['approx_cost(for two people)'] = df['approx_cost(for two people)'].str.replace(',', '').astype(float)

        # Drop duplicates and handle missing values
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        return df

    def prepare_features(self, df, training=True):
        """Prepare features with encoding and feature engineering"""
        df = df.copy()

        # Categorical columns for encoding
        categorical_cols = ['online_order', 'book_table', 'location', 'rest_type',
                           'cuisines', 'listed_in(type)', 'listed_in(city)']

        # Encode categorical variables
        for col in categorical_cols:
            if training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])

        # Prepare feature matrix
        feature_cols = categorical_cols + ['votes', 'rate']
        X = df[feature_cols]

        # Scale features
        if training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def train(self, df):
        """Train the model with optimized hyperparameters"""
        print("Preparing data for training...")

        # Prepare features and target
        X = self.prepare_features(df, training=True)
        y = df['approx_cost(for two people)']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models for stacking
        base_models = [
            ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42)),
            ('gb', GradientBoostingRegressor(random_state=42)),
            ('rf', RandomForestRegressor(random_state=42))
        ]
        meta_model = GradientBoostingRegressor(random_state=42)

        # Hyperparameter optimization using Bayesian Optimization
        xgb_params = {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.1),
            'max_depth': (3, 7),
            'min_child_weight': (1, 5),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0)
        }

        gb_params = {
            'n_estimators': (50, 300),
            'learning_rate': (0.01, 0.1),
            'max_depth': (3, 7),
            'subsample': (0.6, 1.0)
        }

        rf_params = {
            'n_estimators': (50, 300),
            'max_depth': (3, 7),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 4),
            'bootstrap': [True, False]
        }

        # Perform Bayesian Optimization for each base model
        search_spaces = {
            'xgb': xgb_params,
            'gb': gb_params,
            'rf': rf_params
        }

        for name, model in base_models:
            optimizer = BayesSearchCV(
                estimator=model,
                search_spaces=search_spaces[name],
                n_iter=50,
                scoring='neg_mean_squared_error',
                cv=KFold(n_splits=5),
                n_jobs=-1,
                random_state=42
            )
            optimizer.fit(X_train, y_train)
            self.models[name] = optimizer.best_estimator_

        # Create meta features for stacking
        meta_features_train = np.column_stack([model.predict(X_train) for model in self.models.values()])
        meta_features_test = np.column_stack([model.predict(X_test) for model in self.models.values()])

        # Train meta model
        meta_model.fit(meta_features_train, y_train)
        self.models['meta'] = meta_model

        # Predictions and evaluation
        train_pred = cross_val_predict(meta_model, meta_features_train, y_train, cv=KFold(n_splits=5))
        test_pred = meta_model.predict(meta_features_test)

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
        }

        # Save the models
        joblib.dump({
            'models': self.models,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }, 'zomato_model.pkl')

        return metrics

    def predict(self, df):
        """Make predictions for new data"""
        X = self.prepare_features(df, training=False)
        
        base_features = np.column_stack([model.predict(X) for _, model in self.models.items() if _ != 'meta'])
        meta_features = self.models['meta'].predict(base_features)
        
        return meta_features

    @classmethod
    def load_model(cls, model_path):
        """Load a trained model"""
        saved_objects = joblib.load(model_path)
        predictor = cls()
        predictor.models = saved_objects['models']
        predictor.label_encoders = saved_objects['label_encoders']
        predictor.scaler = saved_objects['scaler']
        return predictor

def main():
    # Initialize predictor
    predictor = OptimizedZomatoPredictor()

    # Load and clean data
    df = predictor.load_and_clean_data(r"C:\Users\khush\Downloads\ZOM\khush\zomato.csv")
    print("\nData loaded and cleaned successfully!")

    # Train model and get metrics
    metrics = predictor.train(df)

    # Print performance metrics
    print("\nModel Performance:")
    print(f"Training RMSE: {metrics['train_rmse']:.2f}")
    print(f"Testing RMSE: {metrics['test_rmse']:.2f}")
    print(f"Training R2 Score: {metrics['train_r2']:.3f}")
    print(f"Testing R2 Score: {metrics['test_r2']:.3f}")

    # Example prediction
    sample = df.iloc[[0]]
    predicted_price = predictor.predict(sample)
    print(f"\nSample Prediction:")
    print(f"Actual Price: {sample['approx_cost(for two people)'].values[0]:.2f}")
    print(f"Predicted Price: {predicted_price[0]:.2f}")

if __name__ == "__main__":
    main()
