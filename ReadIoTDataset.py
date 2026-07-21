# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 22:04:30 2025

@author: hp
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """ Finance dataset processor """
    
    def __init__(self, random_state=42):
        self.scaler = StandardScaler()
        self.attribute_columns = None
        self.random_state = random_state
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess data
        
        Returns:
        --------
        tuple: (X_scaled, y)
            X_scaled: Scaled feature matrix
            y: Label array (0=Normal, 1=Attack)
        """
        try:
            # Loading data - skip first row (header)
            print("Loading data...")
            data = pd.read_csv('./Datasets/IOT/UKMNCT_IIoT_FDIA.csv', header=0)  # header=0 means first row is header
            
            # Separate features and labels
            # Last column is the label
            X = data.iloc[:, :-1]  # All columns except last
            y_raw = data.iloc[:, -1]  # Last column as labels
            
            print(f"Data shape: {X.shape}")
            print(f"Unique label values: {y_raw.unique()}")
            
            # Convert labels: Attack=1, Natural=0
            # Assuming labels are strings like 'Attack' and 'Natural'
            y = y_raw.map({'Attack': 1, 'Natural': 0})
            
            # Check if any labels were not converted
            if y.isna().any():
                print(f"Warning: Some labels were not recognized. Unique values: {y_raw.unique()}")
                # If labels are not strings, try numeric conversion
                # Attack might be represented as 1, Natural as 0
                if y_raw.dtype in ['int64', 'float64']:
                    y = y_raw.copy()
                else:
                    # Try to convert based on string matching
                    y = y_raw.apply(lambda x: 1 if str(x).lower() in ['attack', '1', 'anomaly'] else 0)
            
            # Convert to numpy arrays
            X_ = X.values.astype(float)
            y_ = y.values.flatten().astype(int)
            
            # Store attribute column names
            self.attribute_columns = X.columns.tolist()
            
            print(f"Found {len(self.attribute_columns)} attributes")
            print(f"First 5 attribute names: {self.attribute_columns[:5]}")
            
        except FileNotFoundError as e:
            print(f"File errors: {e}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
        # Data preprocessing - Standardization
        print("\nPerforming data standardization...")
        X_scaled = self.scaler.fit_transform(X_)   
        
        print(f"Data shape: {X_scaled.shape}, Label shape: {y_.shape}")
        
        # Calculate anomaly rate
        anomaly_rate = self.calculate_anomaly_rate(y_)
        print(f"Anomaly rate: {anomaly_rate:.4%}")
        
        # Check data quality
        self._check_data_quality(X_scaled, y_)
        
        return X_scaled, y_

    def calculate_anomaly_rate(self, y):
        """
        Calculate the anomaly rate of the dataset
        
        Parameters:
        -----------
        y : array-like
            Label array where 1 represents attack/anomaly and 0 represents normal
            
        Returns:
        --------
        float: Anomaly rate (percentage)
        """
        if y is None or len(y) == 0:
            print("Warning: Label data is empty, cannot calculate anomaly rate")
            return 0.0
        
        total_samples = len(y)
        anomaly_count = np.sum(y == 1)
        normal_count = np.sum(y == 0)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0
        
        print("\n=== Anomaly Rate Statistics ===")
        print(f"Total samples: {total_samples}")
        print(f"Normal samples (0): {normal_count}")
        print(f"Attack samples (1): {anomaly_count}")
        print(f"Anomaly rate: {anomaly_rate:.4%} ({anomaly_count}/{total_samples})")
        print(f"Normal rate: {(1 - anomaly_rate):.4%} ({normal_count}/{total_samples})")
        
        # Check if dataset is imbalanced
        if anomaly_rate < 0.05:
            print("⚠️ Warning: Dataset is severely imbalanced (anomaly rate < 5%)")
        elif anomaly_rate < 0.10:
            print("⚠️ Notice: Dataset is imbalanced (anomaly rate < 10%)")
        else:
            print("✅ Dataset has good balance")
        
        return anomaly_rate

    def _check_data_quality(self, X_, y):
        """
        Check data quality metrics
        
        Parameters:
        -----------
        X_ : array-like
            Feature matrix
        y : array-like
            Label array
        """
        print("\n=== Data Quality Check ===")
        
        # Check for NaN/Inf values
        if X_ is not None:
            nan_count = np.isnan(X_).sum()
            inf_count = np.isinf(X_).sum()
            print(f"Dataset - NaN count: {nan_count}, Inf count: {inf_count}")
            if nan_count > 0 or inf_count > 0:
                print("⚠️ Warning: Dataset contains NaN or Inf values")
        else:
            print("Warning: Dataset is NULL")
        
        # Feature statistics
        if X_ is not None and len(X_) > 0:
            print(f"Dataset feature range: [{X_.min():.4f}, {X_.max():.4f}]")
            print(f"Dataset feature mean: {X_.mean():.4f} ± {X_.std():.4f}")
        
        # Label statistics
        if y is not None and len(y) > 0:
            self.calculate_anomaly_rate(y)
    
    def get_anomaly_statistics(self, y):
        """
        Get detailed anomaly statistics as a dictionary
        
        Parameters:
        -----------
        y : array-like
            Label array
            
        Returns:
        --------
        dict: Dictionary containing anomaly statistics
        """
        if y is None or len(y) == 0:
            return {
                'total_samples': 0,
                'normal_count': 0,
                'anomaly_count': 0,
                'anomaly_rate': 0.0,
                'normal_rate': 0.0
            }
        
        total_samples = len(y)
        anomaly_count = np.sum(y == 1)
        normal_count = np.sum(y == 0)
        anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0
        
        return {
            'total_samples': total_samples,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'normal_rate': 1 - anomaly_rate
        }

    def print_dataset_summary(self, X, y):
        """
        Print a comprehensive summary of the dataset
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Label array
        """
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        # Basic information
        print(f"Total samples: {len(y) if y is not None else 0}")
        print(f"Number of features: {X.shape[1] if X is not None else 0}")
        
        # Feature names
        if self.attribute_columns is not None:
            print(f"\nFeature Names (first 5): {self.attribute_columns[:5]}")
            if len(self.attribute_columns) > 5:
                print(f"  ... and {len(self.attribute_columns) - 5} more features")
        
        # Anomaly statistics
        if y is not None:
            stats = self.get_anomaly_statistics(y)
            print(f"\nClass Distribution:")
            print(f"  Normal class (0): {stats['normal_count']:>8} ({stats['normal_rate']:>6.2%})")
            print(f"  Attack class (1): {stats['anomaly_count']:>8} ({stats['anomaly_rate']:>6.2%})")
            print(f"  {'='*30}")
            print(f"  Total:           {stats['total_samples']:>8} (100.00%)")
        
        # Feature statistics
        if X is not None and len(X) > 0:
            print(f"\nFeature Statistics:")
            print(f"  Mean: {X.mean():.4f}")
            print(f"  Std:  {X.std():.4f}")
            print(f"  Min:  {X.min():.4f}")
            print(f"  Max:  {X.max():.4f}")
        
        print("="*60)


if __name__ == "__main__":
    processor = DataProcessor()
    processed_data = processor.load_and_preprocess_data()
    
    if processed_data is not None:
        X, y = processed_data
        print("\n=== Data Processing Complete ===")
        print(f"Data shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Get detailed statistics
        stats = processor.get_anomaly_statistics(y)
        print(f"\nAnomaly Rate Statistics Details:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Normal samples (0): {stats['normal_count']}")
        print(f"  Attack samples (1): {stats['anomaly_count']}")
        print(f"  Anomaly rate: {stats['anomaly_rate']:.4%}")
        
        # Print comprehensive dataset summary
        processor.print_dataset_summary(X, y)
    else:
        print("Data processing failed!")