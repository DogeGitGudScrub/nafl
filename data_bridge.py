"""
XGBoost-Optimized Feature Bridging for Liver Disease Datasets
Combines LPD (numerical clinical data) with HEPAR (categorical symptom data)
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict

class LiverDataBridge:
    """
    Bridges LPD and HEPAR datasets for unified XGBoost training
    """
    
    def __init__(self):
        self.unified_columns = [
            # Core demographics
            'age', 'gender_male',
            
            # Lab values (XGBoost handles different scales well)
            'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphatase',
            'alt_sgpt', 'ast_sgot', 'total_proteins', 'albumin', 'ag_ratio',
            
            # Symptom indicators (binary features XGBoost loves)
            'has_fatigue', 'has_pain', 'has_jaundice', 'has_nausea',
            'has_itching', 'has_bleeding', 'has_edema',
            
            # Risk factors
            'has_alcoholism', 'has_diabetes', 'has_obesity',
            'has_hepatitis_history', 'has_surgery_history',
            
            # Derived clinical indicators
            'bilirubin_elevated', 'enzymes_elevated', 'proteins_low',
            'multiple_symptoms', 'high_risk_profile',
            
            # Dataset source (helps XGBoost learn dataset-specific patterns)
            'source_lpd', 'source_hepar'
        ]
    
    def extract_numeric_from_range(self, range_str: str) -> float:
        """
        Extract numeric value from HEPAR range format (e.g., 'a699_240' -> 699)
        XGBoost works better with actual numbers than categories
        """
        if pd.isna(range_str) or range_str == 'absent':
            return np.nan
        
        # Extract first number from pattern like 'a699_240'
        match = re.search(r'a?(\d+)_?\d*', str(range_str))
        if match:
            return float(match.group(1))
        return np.nan
    
    def process_lpd_data(self, lpd_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process LPD dataset for XGBoost training
        """
        processed = pd.DataFrame()
        
        # Demographics
        processed['age'] = lpd_df['Age of the patient']
        processed['gender_male'] = (lpd_df['Gender of the patient'] == 'Male').astype(int)
        
        # Direct lab values (XGBoost handles missing values well)
        processed['total_bilirubin'] = lpd_df['Total Bilirubin']
        processed['direct_bilirubin'] = lpd_df['Direct Bilirubin']
        # Handle column names with special characters
        alkphos_col = [col for col in lpd_df.columns if 'Alkphos' in col or 'Alkaline' in col][0]
        sgpt_col = [col for col in lpd_df.columns if 'Sgpt' in col or 'Alamine' in col][0]
        alb_col = [col for col in lpd_df.columns if 'ALB' in col or 'Albumin' in col and 'Ratio' not in col][0]
        
        processed['alkaline_phosphatase'] = lpd_df[alkphos_col]
        processed['alt_sgpt'] = lpd_df[sgpt_col]
        processed['ast_sgot'] = lpd_df['Sgot Aspartate Aminotransferase']
        processed['total_proteins'] = lpd_df['Total Protiens']
        processed['albumin'] = lpd_df[alb_col]
        processed['ag_ratio'] = lpd_df['A/G Ratio Albumin and Globulin Ratio']
        
        # Infer symptoms from lab abnormalities (XGBoost can learn these patterns)
        processed['has_fatigue'] = ((processed['total_bilirubin'] > 1.2) | 
                                  (processed['alt_sgpt'] > 40)).astype(int)
        processed['has_pain'] = (processed['alkaline_phosphatase'] > 300).astype(int)
        processed['has_jaundice'] = (processed['total_bilirubin'] > 2.0).astype(int)
        processed['has_nausea'] = 0  # Not directly available in LPD
        processed['has_itching'] = (processed['total_bilirubin'] > 3.0).astype(int)
        processed['has_bleeding'] = 0  # Not available in LPD
        processed['has_edema'] = (processed['albumin'] < 3.0).astype(int)
        
        # Risk factors (inferred from patterns)
        processed['has_alcoholism'] = 0  # Not available in LPD
        processed['has_diabetes'] = 0   # Not available in LPD
        processed['has_obesity'] = 0    # Not available in LPD
        processed['has_hepatitis_history'] = 0  # Not available in LPD
        processed['has_surgery_history'] = 0    # Not available in LPD
        
        # Derived clinical indicators (XGBoost loves engineered features)
        processed['bilirubin_elevated'] = (processed['total_bilirubin'] > 1.2).astype(int)
        processed['enzymes_elevated'] = ((processed['alt_sgpt'] > 40) | 
                                       (processed['ast_sgot'] > 40)).astype(int)
        processed['proteins_low'] = (processed['total_proteins'] < 6.0).astype(int)
        processed['multiple_symptoms'] = (processed[['has_fatigue', 'has_pain', 'has_jaundice', 
                                                   'has_itching', 'has_edema']].sum(axis=1) >= 2).astype(int)
        processed['high_risk_profile'] = ((processed['bilirubin_elevated'] == 1) & 
                                        (processed['enzymes_elevated'] == 1)).astype(int)
        
        # Dataset source indicators
        processed['source_lpd'] = 1
        processed['source_hepar'] = 0
        
        # Target variable (1=disease, 0=no disease)
        processed['target'] = (lpd_df['Result'] == 1).astype(int)
        
        return processed[self.unified_columns + ['target']]
    
    def process_hepar_data(self, hepar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process HEPAR dataset for XGBoost training
        """
        processed = pd.DataFrame()
        
        # Demographics
        # Convert age ranges to numeric midpoints
        age_mapping = {
            'age0_30': 25, 'age31_50': 40, 'age51_65': 58, 'age65_100': 75
        }
        processed['age'] = hepar_df['age'].map(age_mapping)
        processed['gender_male'] = (hepar_df['sex'] == 'male').astype(int)
        
        # Extract lab values from HEPAR ranges
        processed['total_bilirubin'] = hepar_df['bilirubin'].apply(self.extract_numeric_from_range)
        processed['direct_bilirubin'] = np.nan  # Not available in HEPAR
        processed['alkaline_phosphatase'] = hepar_df['phosphatase'].apply(self.extract_numeric_from_range)
        processed['alt_sgpt'] = hepar_df['alt'].apply(self.extract_numeric_from_range)
        processed['ast_sgot'] = hepar_df['ast'].apply(self.extract_numeric_from_range)
        processed['total_proteins'] = hepar_df['proteins'].apply(self.extract_numeric_from_range)
        processed['albumin'] = hepar_df['albumin'].apply(self.extract_numeric_from_range)
        processed['ag_ratio'] = np.nan  # Not directly available in HEPAR
        
        # Direct symptom indicators (HEPAR's strength!)
        processed['has_fatigue'] = (hepar_df['fatigue'] == 'present').astype(int)
        processed['has_pain'] = ((hepar_df['pain'] == 'present') | 
                               (hepar_df['upper_pain'] == 'present') |
                               (hepar_df['pain_ruq'] == 'present')).astype(int)
        processed['has_jaundice'] = (hepar_df['jaundice'] == 'present').astype(int)
        processed['has_nausea'] = (hepar_df['nausea'] == 'present').astype(int)
        processed['has_itching'] = (hepar_df['itching'] == 'present').astype(int)
        processed['has_bleeding'] = (hepar_df['bleeding'] == 'present').astype(int)
        processed['has_edema'] = (hepar_df['edema'] == 'present').astype(int)
        
        # Risk factors (HEPAR's other strength!)
        processed['has_alcoholism'] = (hepar_df['alcoholism'] == 'present').astype(int)
        processed['has_diabetes'] = (hepar_df['diabetes'] == 'present').astype(int)
        processed['has_obesity'] = (hepar_df['obesity'] == 'present').astype(int)
        processed['has_hepatitis_history'] = ((hepar_df['THepatitis'] == 'present') |
                                            (hepar_df['ChHepatitis'] == 'present') |
                                            (hepar_df['RHepatitis'] == 'present')).astype(int)
        processed['has_surgery_history'] = (hepar_df['surgery'] == 'present').astype(int)
        
        # Derived indicators
        processed['bilirubin_elevated'] = (processed['total_bilirubin'] > 1.2).astype(int)
        processed['enzymes_elevated'] = ((processed['alt_sgpt'] > 40) | 
                                       (processed['ast_sgot'] > 40)).astype(int)
        processed['proteins_low'] = (processed['total_proteins'] < 6.0).astype(int)
        processed['multiple_symptoms'] = (processed[['has_fatigue', 'has_pain', 'has_jaundice', 
                                                   'has_nausea', 'has_itching', 'has_bleeding', 
                                                   'has_edema']].sum(axis=1) >= 2).astype(int)
        processed['high_risk_profile'] = (processed[['has_alcoholism', 'has_diabetes', 
                                                   'has_hepatitis_history']].sum(axis=1) >= 1).astype(int)
        
        # Dataset source indicators
        processed['source_lpd'] = 0
        processed['source_hepar'] = 1
        
        # Create unified target: any liver condition = disease
        liver_conditions = ['Cirrhosis', 'Steatosis', 'fibrosis', 'Hyperbilirubinemia', 'PBC']
        target_cols = [col for col in liver_conditions if col in hepar_df.columns]
        processed['target'] = (hepar_df[target_cols] == 'present').any(axis=1).astype(int)
        
        return processed[self.unified_columns + ['target']]
    
    def combine_datasets(self, lpd_path: str, hepar_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and combine both datasets into XGBoost-ready format
        """
        # Load datasets with encoding handling
        try:
            lpd_df = pd.read_csv(lpd_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                lpd_df = pd.read_csv(lpd_path, encoding='latin-1')
            except UnicodeDecodeError:
                lpd_df = pd.read_csv(lpd_path, encoding='cp1252')
        
        try:
            hepar_df = pd.read_csv(hepar_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                hepar_df = pd.read_csv(hepar_path, encoding='latin-1')
            except UnicodeDecodeError:
                hepar_df = pd.read_csv(hepar_path, encoding='cp1252')
        
        print(f"Loaded LPD: {lpd_df.shape[0]} samples")
        print(f"Loaded HEPAR: {hepar_df.shape[0]} samples")
        
        # Process each dataset
        lpd_processed = self.process_lpd_data(lpd_df)
        hepar_processed = self.process_hepar_data(hepar_df)
        
        # Combine datasets
        combined_df = pd.concat([lpd_processed, hepar_processed], ignore_index=True)
        
        # Dataset statistics for XGBoost optimization
        stats = {
            'total_samples': combined_df.shape[0],
            'lpd_samples': lpd_processed.shape[0],
            'hepar_samples': hepar_processed.shape[0],
            'feature_count': len(self.unified_columns),
            'missing_data_ratio': combined_df.isnull().sum().sum() / (combined_df.shape[0] * combined_df.shape[1]),
            'class_distribution': combined_df['target'].value_counts().to_dict(),
            'dataset_balance': {
                'lpd_positive_rate': lpd_processed['target'].mean(),
                'hepar_positive_rate': hepar_processed['target'].mean()
            }
        }
        
        print(f"\nCombined dataset: {stats['total_samples']} samples, {stats['feature_count']} features")
        print(f"Missing data: {stats['missing_data_ratio']:.2%}")
        print(f"Class distribution: {stats['class_distribution']}")
        
        return combined_df, stats

# Example usage for XGBoost training
if __name__ == "__main__":
    bridge = LiverDataBridge()
    
    # Combine datasets
    combined_data, stats = bridge.combine_datasets(
        "dataset/Liver_Patient_Dataset_(LPD)_train[1].csv",
        "dataset/HEPAR_simulated_patients_ood.csv"
    )
    
    # Save combined dataset for XGBoost
    combined_data.to_csv("dataset/combined_liver_dataset.csv", index=False)
    print("Combined dataset saved for XGBoost training!")
