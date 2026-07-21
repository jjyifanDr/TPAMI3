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
        tuple: (X_train_scaled, y)
            X_train_scaled: Scaled feature matrix
            y: Label array
        """
        try:
            # Loading data
            print("Loading data...")
            _data = pd.read_csv('./Datasets/Finance/FinanceData.csv', skiprows=1, header=None)
            data = _data.iloc[:, :-2]  # Delete last two columns
            
            labels = pd.read_csv('./Datasets/Finance/FinanceLabel.csv', skiprows=1, header=None)
            
        except FileNotFoundError as e:
            print(f"File errors: {e}")
            return None
        
        # Using numeric indices as column names since headers are skipped
        self.attribute_columns = list(range(data.shape[1]))
        
        print(f"Found {len(self.attribute_columns)} attributes")
        print(f"Data shape: {data.shape}")
        
        # Convert to numpy arrays
        X_ = data.values.astype(float)
        y_ = labels.values.flatten().astype(int)
        
        # Data preprocessing - Standardization
        print("Performing data standardization...")
        X_train_scaled = self.scaler.fit_transform(X_)   
        
        print(f"Data shape: {X_train_scaled.shape}, Label shape: {y_.shape}")
        
        # Calculate anomaly rate
        anomaly_rate = self.calculate_anomaly_rate(y_)
        print(f"Anomaly rate: {anomaly_rate:.4%}")
        
        # Check data quality
        self._check_data_quality(X_train_scaled, y_)
        
        return X_train_scaled, y_

    def calculate_anomaly_rate(self, y):
        """
        Calculate the anomaly rate of the dataset
        
        Parameters:
        -----------
        y : array-like
            Label array where 1 represents anomaly and 0 represents normal
            
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
        print(f"Normal samples: {normal_count}")
        print(f"Anomaly samples: {anomaly_count}")
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
            print(f"Dataset - NaN count: {np.isnan(X_).sum()}, Inf count: {np.isinf(X_).sum()}")
        
        # Use the new anomaly rate calculation method
        if y is not None and len(y) > 0:
            self.calculate_anomaly_rate(y)
        else:
            print("Warning: Label data is empty")
        
        # Feature statistics
        if X_ is not None:
            print(f"Dataset feature range: [{X_.min():.4f}, {X_.max():.4f}]")
            print(f"Dataset feature mean: {X_.mean():.4f} ± {X_.std():.4f}")
        else:
            print("Warning: Dataset is NULL")
    
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
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        
        # Basic information
        print(f"Total samples: {len(y) if y is not None else 0}")
        print(f"Number of features: {X.shape[1] if X is not None else 0}")
        
        # Anomaly statistics
        if y is not None:
            stats = self.get_anomaly_statistics(y)
            print(f"\nClass Distribution:")
            print(f"  Normal class (0): {stats['normal_count']} ({stats['normal_rate']:.2%})")
            print(f"  Anomaly class (1): {stats['anomaly_count']} ({stats['anomaly_rate']:.2%})")
        
        # Feature statistics
        if X is not None:
            print(f"\nFeature Statistics:")
            print(f"  Mean: {X.mean():.4f}")
            print(f"  Std: {X.std():.4f}")
            print(f"  Min: {X.min():.4f}")
            print(f"  Max: {X.max():.4f}")
        
        print("="*50)


if __name__ == "__main__":
    processor = DataProcessor()
    processed_data = processor.load_and_preprocess_data()
    
    if processed_data is not None:
        X, y = processed_data
        print("\n=== Data Processing Complete ===")
        print(f"Data shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        
        # Get detailed statistics
        stats = processor.get_anomaly_statistics(y)
        print(f"\nAnomaly Rate Statistics Details:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Normal samples: {stats['normal_count']}")
        print(f"  Anomaly samples: {stats['anomaly_count']}")
        print(f"  Anomaly rate: {stats['anomaly_rate']:.4%}")
        
        # Print comprehensive dataset summary
        processor.print_dataset_summary(X, y)
    else:
        print("Data processing failed!")