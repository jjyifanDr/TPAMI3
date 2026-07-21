# Real Image Loader 

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

# ----------------------------------------------------------------------
class RealImageLoader:
    def __init__(self, data_root, normal_root='normal', anomaly_root='anomaly',
                 normal_classes=None, anomaly_classes=None,
                 target_size=(32, 32), sample_limit=None):
        """
        Initialize Real Image Loader
        
        Parameters:
        -----------
        data_root : str
            Root directory containing the dataset
        normal_root : str
            Subdirectory name for normal images (default: 'normal')
        anomaly_root : str
            Subdirectory name for anomaly images (default: 'anomaly')
        normal_classes : list
            List of normal class names to load (if empty, loads all subdirectories)
        anomaly_classes : list
            List of anomaly class names to load (if empty, loads all subdirectories)
        target_size : tuple
            Target image size (width, height)
        sample_limit : int
            Maximum number of samples to load (optional)
        """
        self.data_root = data_root
        self.normal_root = normal_root
        self.anomaly_root = anomaly_root
        self.normal_classes = normal_classes or []
        self.anomaly_classes = anomaly_classes or []
        self.target_size = target_size
        self.sample_limit = sample_limit

    def _load_from_folder(self, folder_path, label):
        """
        Load images from a specific folder
        
        Parameters:
        -----------
        folder_path : str
            Path to the folder containing images
        label : int
            Label for the images (0 for normal, 1 for anomaly)
            
        Returns:
        --------
        tuple: (X, y) lists of features and labels
        """
        X = []
        y = []
        if not os.path.isdir(folder_path):
            print(f"Warning: folder {folder_path} not found, skipped.")
            return X, y
        
        print(f"Loading images from: {folder_path}")
        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert('L')  # Grayscale
                    img = img.resize(self.target_size)
                    arr = np.array(img).flatten() / 255.0
                    X.append(arr)
                    y.append(label)  # 0=normal, 1=anomaly
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return X, y

    def _get_class_list(self, root_path):
        """
        Get list of classes from a directory
        
        Parameters:
        -----------
        root_path : str
            Path to the root directory
            
        Returns:
        --------
        list: List of class names (subdirectories)
        """
        if not os.path.isdir(root_path):
            return []
        
        classes = []
        for item in os.listdir(root_path):
            item_path = os.path.join(root_path, item)
            if os.path.isdir(item_path):
                classes.append(item)
        return classes

    def load_data(self):
        """
        Load all images from the dataset without splitting
        
        Returns:
        --------
        tuple: (X, y)
            Complete dataset and corresponding labels
            - 0: normal
            - 1: anomaly
        """
        X_all = []
        y_all = []

        # Determine which classes to load for normal images
        normal_path = os.path.join(self.data_root, self.normal_root)
        if self.normal_classes:
            normal_classes_to_load = self.normal_classes
        else:
            # If no classes specified, load all subdirectories
            normal_classes_to_load = self._get_class_list(normal_path)
            if not normal_classes_to_load:
                # If no subdirectories, try to load images directly from the folder
                print(f"Loading images directly from: {normal_path}")
                X, y = self._load_from_folder(normal_path, 0)  # 0 = normal
                X_all.extend(X)
                y_all.extend(y)
                print(f"Loaded {len(X)} normal images directly from {self.normal_root}")

        # Load normal images (label = 0)
        for class_name in normal_classes_to_load:
            class_path = os.path.join(self.data_root, self.normal_root, class_name)
            X, y = self._load_from_folder(class_path, 0)  # 0 = normal
            X_all.extend(X)
            y_all.extend(y)
            print(f"Loaded {len(X)} normal images from {class_name}")

        # Determine which classes to load for anomaly images
        anomaly_path = os.path.join(self.data_root, self.anomaly_root)
        if self.anomaly_classes:
            anomaly_classes_to_load = self.anomaly_classes
        else:
            # If no classes specified, load all subdirectories
            anomaly_classes_to_load = self._get_class_list(anomaly_path)
            if not anomaly_classes_to_load:
                # If no subdirectories, try to load images directly from the folder
                print(f"Loading images directly from: {anomaly_path}")
                X, y = self._load_from_folder(anomaly_path, 1)  # 1 = anomaly
                X_all.extend(X)
                y_all.extend(y)
                print(f"Loaded {len(X)} anomaly images directly from {self.anomaly_root}")

        # Load anomaly images (label = 1)
        for class_name in anomaly_classes_to_load:
            class_path = os.path.join(self.data_root, self.anomaly_root, class_name)
            X, y = self._load_from_folder(class_path, 1)  # 1 = anomaly
            X_all.extend(X)
            y_all.extend(y)
            print(f"Loaded {len(X)} anomaly images from {class_name}")

        if len(X_all) == 0:
            raise ValueError(f"No images loaded! Check paths: {self.data_root}")

        X = np.array(X_all)
        y = np.array(y_all)
        
        # Calculate anomaly rate
        anomaly_rate = np.mean(y)
        print(f"\n=== Dataset Statistics ===")
        print(f"Total images: {len(X)}")
        print(f"Normal images (label=0): {np.sum(y == 0)}")
        print(f"Anomaly images (label=1): {np.sum(y == 1)}")
        print(f"Anomaly rate: {anomaly_rate:.4%}")

        # Sample limiting if specified
        if self.sample_limit and self.sample_limit < len(X):
            idx_normal = np.where(y == 0)[0]
            idx_anomaly = np.where(y == 1)[0]
            n_normal = int(self.sample_limit * len(idx_normal) / len(X))
            n_anomaly = self.sample_limit - n_normal
            
            # Ensure we don't exceed available samples
            n_normal = min(n_normal, len(idx_normal))
            n_anomaly = min(n_anomaly, len(idx_anomaly))
            
            idx = np.concatenate([
                np.random.choice(idx_normal, n_normal, replace=False),
                np.random.choice(idx_anomaly, n_anomaly, replace=False)
            ])
            X = X[idx]
            y = y[idx]
            print(f"Limited to {self.sample_limit} samples.")
            print(f"Normal (label=0): {np.sum(y == 0)}, Anomaly (label=1): {np.sum(y == 1)}")

        return X, y
    
    def get_data_summary(self, X, y):
        """
        Get comprehensive data summary
        
        Parameters:
        -----------
        X, y : arrays
            The complete dataset and labels
            - 0: normal
            - 1: anomaly
            
        Returns:
        --------
        dict: Dictionary containing summary statistics
        """
        summary = {
            'total_samples': len(X),
            'feature_dim': X.shape[1] if len(X) > 0 else 0,
            'normal_samples': np.sum(y == 0),
            'anomaly_samples': np.sum(y == 1),
            'anomaly_rate': np.mean(y) if len(y) > 0 else 0,
        }
        return summary


# Main execution
if __name__ == "__main__":
    # Configure dataset paths and classes
    DATA_ROOT = './Datasets/Material_Images'
    
    # Set to empty lists to automatically detect subdirectories
    NORMAL_CLASSES = []  # Will auto-detect subdirectories under 'normal'
    ANOMALY_CLASSES = []  # Will auto-detect subdirectories under 'anomaly'
    
    # Create processor instance with proper parameters
    processor = RealImageLoader(
        data_root=DATA_ROOT,
        normal_root='normal',  # Subfolder containing normal images
        anomaly_root='anomaly',  # Subfolder containing anomaly images
        normal_classes=NORMAL_CLASSES,  # Auto-detect classes
        anomaly_classes=ANOMALY_CLASSES,  # Auto-detect classes
        target_size=(32, 32),  # Resize to 32x32 to save memory
        sample_limit=None  # Set to e.g., 1000 to limit samples
    )
    
    try:
        X, y = processor.load_data()
        
        if X is not None and len(X) > 0:
            print("\n=== Data Loading Complete ===")
            print(f"Dataset shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            
            # Get detailed summary
            summary = processor.get_data_summary(X, y)
            print("\n=== Dataset Summary ===")
            print(f"Total samples: {summary['total_samples']}")
            print(f"Feature dimension: {summary['feature_dim']}")
            print(f"Normal samples (label=0): {summary['normal_samples']}")
            print(f"Anomaly samples (label=1): {summary['anomaly_samples']}")
            print(f"Anomaly rate: {summary['anomaly_rate']:.4%}")
            
            # Show a few sample details
            print("\n=== Sample Data Preview ===")
            print(f"First 5 labels (0=normal, 1=anomaly): {y[:5]}")
            print(f"First sample shape: {X[0].shape}")
            print(f"First sample min/max values: {X[0].min():.4f}, {X[0].max():.4f}")
            
            # Verify label distribution
            print(f"\nLabel distribution:")
            print(f"  Normal (0): {np.sum(y == 0)} samples")
            print(f"  Anomaly (1): {np.sum(y == 1)} samples")
        else:
            print("Data loading failed!")
            
    except Exception as e:
        print(f"Error during data loading: {e}")