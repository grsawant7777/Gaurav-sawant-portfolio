import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import joblib
import os
import warnings
import sys

# Suppress warnings for clean MLOps execution log
warnings.filterwarnings("ignore")

# Define file paths
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'credit_risk_model.pkl')
DATA_URL="C:/Users/Gaurav sawant/Desktop/job doc/job project/Project Doc/fiannce/credit-risk-mlops/data/01_raw/credit_risk_dataset.csv"
# DATA_URL = 'https://raw.githubusercontent.com/datasets/credit-risk-dataset/main/credit_risk_dataset.csv'

def train_and_save_model():
    """
    Loads data, builds the preprocessing pipeline, applies SMOTE, trains XGBoost, 
    and saves the full pipeline along with the list of expected features.
    """
    print("--- Starting MLOps Training Pipeline ---")
    
    # --- 1. Load and Clean Data ---
    try:
        print(f"Loading data from: {DATA_URL}")
        df = pd.read_csv(DATA_URL)
        
        if df.empty:
            print("FATAL ERROR: Dataset is empty.")
            sys.exit(1)

        # Robust NA handling only on key features, creating a cleaned copy
        df_cleaned = df.dropna(subset=['person_emp_length', 'loan_int_rate']).copy()
        
    except Exception as e:
        print(f"FATAL ERROR: Could not load or process data. {e}")
        sys.exit(1)


    # Define target and features
    X = df_cleaned.drop(columns=['loan_status'])
    y = df_cleaned['loan_status'] 
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 2. Define Preprocessing Pipeline ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # CRITICAL: Store the feature order/names for API validation
    global_feature_set = X.columns.tolist() 

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
            ]), categorical_features)
        ],
        remainder='drop'
    )
    
    # --- 3. Full ML Pipeline with Imbalance Handling ---
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=150, 
        learning_rate=0.07,
        max_depth=5,
        random_state=42
    )

    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('classifier', xgb_model)
    ])
    
    print("Fitting the full pipeline (Preprocessing + SMOTE + XGBoost)...")
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate on Test Set
    y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"Model Training Complete. Test AUC-ROC: {auc_roc:.4f}")

    # --- 4. Save Robust Artifact ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    artifact = {
        'pipeline': full_pipeline,
        'features': global_feature_set
    }
    joblib.dump(artifact, MODEL_FILE)
    print(f"✅ Successfully saved deployable artifact (pipeline and feature list) to {MODEL_FILE}")
    
if __name__ == '__main__':
    train_and_save_model()