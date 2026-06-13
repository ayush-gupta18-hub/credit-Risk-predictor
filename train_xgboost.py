import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import pickle
import os

def main():
    # Load the transformed data
    data_path = 'Data Cleaned/credit_risk_transformed_final.csv'
    if not os.path.exists(data_path):
        # fallback to Scikit-Learn directory if not in Data Cleaned
        data_path = 'Scikit-Learn/credit_risk_transformed_final.csv'
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Features and Target
    X = df.drop(['Status', 'Id'], axis=1)
    y = df['Status']
    
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Initialize XGBoost with the best hyperparameters from the notebook
    best_xgb = XGBClassifier(
        learning_rate=0.2,
        max_depth=5,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    print("Training XGBoost model...")
    best_xgb.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = best_xgb.predict(X_test)
    y_probs = best_xgb.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_probs)
    
    print("\n--- Test Set Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    
    # Save the model
    os.makedirs('Scikit-Learn', exist_ok=True)
    model_path = 'Scikit-Learn/xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_xgb, f)
        
    print(f"\nModel saved successfully to {model_path}!")

if __name__ == '__main__':
    main()
