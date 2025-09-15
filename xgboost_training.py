"""
XGBoost Training Script for Combined Liver Disease Dataset
Optimized for handling bridged LPD + HEPAR data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from data_bridge import LiverDataBridge

class XGBoostLiverPredictor:
    """
    XGBoost model optimized for combined liver disease datasets
    """
    
    def __init__(self):
        # XGBoost parameters optimized for medical data
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,  # Moderate depth for medical interpretability
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_estimators': 200,
            'importance_type': 'gain',  # Better for mixed feature types
            'missing': np.nan,  # Handle missing values naturally
            'tree_method': 'hist'  # Faster for mixed data types
        }
        
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, combined_df: pd.DataFrame, scale_features: bool = False):
        """
        Prepare combined dataset for XGBoost training
        """
        # Separate features and target
        X = combined_df.drop('target', axis=1)
        y = combined_df['target']
        
        self.feature_names = X.columns.tolist()
        
        # Optional scaling (XGBoost usually doesn't need it, but can help with mixed scales)
        if scale_features:
            # Only scale numeric features, keep binary features as-is
            numeric_features = ['age', 'total_bilirubin', 'direct_bilirubin', 
                              'alkaline_phosphatase', 'alt_sgpt', 'ast_sgot', 
                              'total_proteins', 'albumin', 'ag_ratio']
            
            X_scaled = X.copy()
            for feature in numeric_features:
                if feature in X.columns:
                    X_scaled[[feature]] = self.scaler.fit_transform(X[[feature]])
            X = X_scaled
        
        return X, y
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train XGBoost model with early stopping and validation
        """
        # Create DMatrix for XGBoost (handles missing values efficiently)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        # Validation set for early stopping
        eval_list = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            eval_list.append((dval, 'eval'))
        
        # Train model
        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.xgb_params['n_estimators'],
            evals=eval_list,
            early_stopping_rounds=20,
            verbose_eval=verbose
        )
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, dataset_source=None):
        """
        Comprehensive evaluation including dataset-specific performance
        """
        # Predictions
        dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Overall metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print("=== Overall Model Performance ===")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Dataset-specific evaluation if source information available
        if dataset_source is not None:
            print("\n=== Dataset-Specific Performance ===")
            
            # LPD performance
            lpd_mask = X_test['source_lpd'] == 1
            if lpd_mask.sum() > 0:
                lpd_auc = roc_auc_score(y_test[lpd_mask], y_pred_proba[lpd_mask])
                print(f"LPD Dataset AUC: {lpd_auc:.4f} ({lpd_mask.sum()} samples)")
            
            # HEPAR performance  
            hepar_mask = X_test['source_hepar'] == 1
            if hepar_mask.sum() > 0:
                hepar_auc = roc_auc_score(y_test[hepar_mask], y_pred_proba[hepar_mask])
                print(f"HEPAR Dataset AUC: {hepar_auc:.4f} ({hepar_mask.sum()} samples)")
        
        return {
            'auc': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance to understand dataset contribution
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        # Color code by feature type
        colors = []
        for feature in top_features['feature']:
            if 'source_' in feature:
                colors.append('red')  # Dataset indicators
            elif feature.startswith('has_'):
                colors.append('green')  # Symptoms (HEPAR strength)
            elif feature in ['age', 'gender_male']:
                colors.append('blue')  # Demographics
            else:
                colors.append('orange')  # Lab values (LPD strength)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Features - XGBoost Model')
        plt.gca().invert_yaxis()
        
        # Legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, color='orange', label='Lab Values (LPD)'),
            plt.Rectangle((0,0),1,1, color='green', label='Symptoms (HEPAR)'),
            plt.Rectangle((0,0),1,1, color='blue', label='Demographics'),
            plt.Rectangle((0,0),1,1, color='red', label='Dataset Source')
        ]
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def cross_validate_combined_data(self, X, y, cv_folds=5):
        """
        Cross-validation specifically designed for combined dataset
        """
        # Stratified CV to maintain class balance
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Create XGBoost classifier for sklearn compatibility
        xgb_clf = xgb.XGBClassifier(**self.xgb_params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(xgb_clf, X, y, cv=skf, scoring='roc_auc')
        
        print("=== Cross-Validation Results ===")
        print(f"CV AUC Scores: {cv_scores}")
        print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores

def main():
    """
    Main training pipeline for combined liver disease prediction
    """
    print("=== XGBoost Liver Disease Prediction with Combined Datasets ===")
    
    # Step 1: Create combined dataset
    print("\n1. Bridging datasets...")
    bridge = LiverDataBridge()
    combined_data, stats = bridge.combine_datasets(
        "dataset/Liver_Patient_Dataset_(LPD)_train[1].csv",
        "dataset/HEPAR_simulated_patients_ood.csv"
    )
    
    # Step 2: Prepare data
    print("\n2. Preparing data for XGBoost...")
    predictor = XGBoostLiverPredictor()
    X, y = predictor.prepare_data(combined_data, scale_features=False)
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Step 3: Train-test split (stratified to maintain dataset balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain: {X_train_final.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Step 4: Cross-validation
    print("\n3. Cross-validation...")
    cv_scores = predictor.cross_validate_combined_data(X_train, y_train)
    
    # Step 5: Train final model
    print("\n4. Training final model...")
    predictor.train_model(X_train_final, y_train_final, X_val, y_val)
    
    # Step 6: Evaluation
    print("\n5. Model evaluation...")
    results = predictor.evaluate_model(X_test, y_test, dataset_source=True)
    
    # Step 7: Feature importance analysis
    print("\n6. Feature importance analysis...")
    importance_df = predictor.plot_feature_importance()
    
    # Step 8: Save model and results
    print("\n7. Saving model...")
    predictor.model.save_model("models/xgboost_combined_liver_model.json")
    
    # Save feature importance
    importance_df.to_csv("results/feature_importance.csv", index=False)
    
    # Save combined dataset
    combined_data.to_csv("dataset/combined_liver_dataset.csv", index=False)
    
    print("\n=== Training Complete! ===")
    print(f"Final Model AUC: {results['auc']:.4f}")
    print("Model saved to: models/xgboost_combined_liver_model.json")
    
    return predictor, combined_data, results

if __name__ == "__main__":
    # Create necessary directories
    import os
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Run training pipeline
    predictor, data, results = main()
