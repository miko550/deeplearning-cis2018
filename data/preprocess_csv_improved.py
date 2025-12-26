"""
Improved CSV Preprocessing Script
==================================
This script loads CSV files, handles missing values, encodes labels,
and splits data into train/validation/test sets.

Improvements over original:
- Better error handling and validation
- More robust NaN handling for different data types
- Modular function-based structure
- Type hints and documentation
- Configuration options
- Better logging and statistics
- Memory-efficient processing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import os
from typing import Tuple, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """A class to handle CSV data preprocessing with improved error handling."""
    
    def __init__(self, csv_path: str = "csv/", test_size: float = 0.2, 
                 val_size: float = 0.5, random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            csv_path: Path to directory containing CSV files
            test_size: Proportion of data for test set (default: 0.2)
            val_size: Proportion of test set to use for validation (default: 0.5)
            random_state: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.df = None
        self.label_encoder = None
        self.class_names = None
        self.start_time = None
        
    def load_csv_files(self) -> pd.DataFrame:
        """
        Load and combine multiple CSV files from a directory.
        
        Returns:
            Combined DataFrame from all CSV files
            
        Raises:
            FileNotFoundError: If no CSV files are found
            ValueError: If CSV files cannot be loaded
        """
        print("--- Starting Data Preprocessing ---")
        self.start_time = time.time()
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Directory '{self.csv_path}' not found!")
        
        csv_files = []
        for root, dirs, files in os.walk(self.csv_path):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in '{self.csv_path}'!")
        
        print(f"Found {len(csv_files)} CSV file(s)")
        
        # Load files with error handling
        dataframes = []
        for csv_file in csv_files:
            try:
                df_temp = pd.read_csv(csv_file)
                dataframes.append(df_temp)
                print(f"  Loaded: {csv_file} ({df_temp.shape[0]} rows, {df_temp.shape[1]} cols)")
            except Exception as e:
                print(f"  WARNING: Failed to load {csv_file}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No CSV files could be loaded successfully!")
        
        self.df = pd.concat(dataframes, ignore_index=True)
        print(f"\nTotal combined rows: {self.df.shape[0]}")
        print(f"Total columns: {self.df.shape[1]}")
        
        return self.df
    
    def handle_missing_values(self) -> None:
        """
        Handle NaN values intelligently based on column data types.
        - Numeric columns: Fill with mean (or median if mean is NaN)
        - Categorical columns: Fill with mode (most frequent value)
        - If all values are NaN: Fill with 0 for numeric, 'Unknown' for categorical
        """
        print("\n--- Handling Missing Values ---")
        nan_counts = self.df.isna().sum()
        problematic_nan_cols = nan_counts[nan_counts > 0]
        
        if problematic_nan_cols.empty:
            print("✓ No NaN values found in the dataset.")
            return
        
        print(f"Found {len(problematic_nan_cols)} columns with NaN values")
        print(f"Total NaN values: {nan_counts.sum()}")
        
        for col in problematic_nan_cols.index:
            nan_count = nan_counts[col]
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Try mean first
                mean_val = self.df[col].mean()
                if pd.isna(mean_val):
                    # If mean is NaN, try median
                    median_val = self.df[col].median()
                    if pd.isna(median_val):
                        # If both are NaN, use 0
                        fill_value = 0
                        print(f"  Column '{col}': Filled {nan_count} NaN values with 0 (all values were NaN)")
                    else:
                        fill_value = median_val
                        print(f"  Column '{col}': Filled {nan_count} NaN values with median ({median_val:.4f})")
                else:
                    fill_value = mean_val
                    print(f"  Column '{col}': Filled {nan_count} NaN values with mean ({mean_val:.4f})")
                
                self.df[col].fillna(fill_value, inplace=True)
            else:
                # For categorical/string columns, use mode
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    fill_value = mode_val[0]
                    print(f"  Column '{col}': Filled {nan_count} NaN values with mode ('{fill_value}')")
                else:
                    fill_value = 'Unknown'
                    print(f"  Column '{col}': Filled {nan_count} NaN values with 'Unknown' (no mode available)")
                
                self.df[col].fillna(fill_value, inplace=True)
        
        # Verify NaN are fixed
        remaining_nan = self.df.isna().sum().sum()
        if remaining_nan > 0:
            print(f"  WARNING: {remaining_nan} NaN values still remain after filling!")
            # Force fill any remaining NaN
            self.df.fillna(0, inplace=True)
            print(f"  Force-filled remaining NaN values with 0")
        else:
            print("✓ All NaN values have been filled.")
    
    def handle_infinity_values(self) -> None:
        """
        Replace infinity values with appropriate substitutes.
        """
        print("\n--- Handling Infinity Values ---")
        inf_counts = self.df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).sum()
        problematic_inf_cols = inf_counts[inf_counts > 0]
        
        if problematic_inf_cols.empty:
            print("✓ No infinity values found in the dataset.")
            return
        
        print(f"Found {len(problematic_inf_cols)} columns with infinity values")
        total_inf = inf_counts.sum()
        print(f"Total infinity values: {total_inf}")
        
        # Replace infinity with column-specific strategy
        for col in problematic_inf_cols.index:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Replace with max finite value or 0 if no finite values exist
                finite_values = self.df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(finite_values) > 0:
                    max_finite = finite_values.max()
                    min_finite = finite_values.min()
                    # Replace +inf with max, -inf with min
                    self.df[col].replace([np.inf], max_finite, inplace=True)
                    self.df[col].replace([-np.inf], min_finite, inplace=True)
                    print(f"  Column '{col}': Replaced {inf_counts[col]} inf values with finite bounds")
                else:
                    self.df[col].replace([np.inf, -np.inf], 0, inplace=True)
                    print(f"  Column '{col}': Replaced {inf_counts[col]} inf values with 0 (no finite values)")
        
        print("✓ All infinity values have been handled.")
    
    def encode_labels(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Separate features and labels, then encode labels.
        
        Returns:
            Tuple of (features, encoded_labels, class_names)
        """
        print("\n--- Encoding Labels ---")
        
        # Separate features and labels (last column is label)
        if self.df.shape[1] < 2:
            raise ValueError("DataFrame must have at least 2 columns (features + label)!")
        
        X = self.df.iloc[:, :-1].copy()
        y = self.df.iloc[:, -1].copy()
        
        # Extract unique class names before encoding
        unique_classes = sorted(y.unique().tolist())
        print(f"Found {len(unique_classes)} unique classes:")
        for i, class_name in enumerate(unique_classes):
            class_count = (y == class_name).sum()
            print(f"  {i}: {class_name} ({class_count} instances)")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        self.class_names = self.label_encoder.classes_
        
        print(f"✓ Labels encoded. Class mapping:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {i} -> {class_name}")
        
        return X, y_encoded, self.class_names
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray) -> Tuple:
        """
        Split data into train/validation/test sets with stratification.
        
        Args:
            X: Feature matrix
            y: Encoded labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n--- Splitting Data ---")
        
        # First split: train vs (test + val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        train_ratio = len(X_train) / len(X)
        print(f"Train set: {len(X_train)} samples ({train_ratio*100:.1f}%)")
        print(f"Temp set (test+val): {len(X_temp)} samples ({(1-train_ratio)*100:.1f}%)")
        
        # Second split: test vs val
        # Calculate actual validation size relative to total dataset
        val_ratio = self.val_size * self.test_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - self.val_size),  # Split temp set 50/50
            random_state=self.random_state,
            stratify=y_temp
        )
        
        val_ratio_actual = len(X_val) / len(X)
        test_ratio_actual = len(X_test) / len(X)
        print(f"Validation set: {len(X_val)} samples ({val_ratio_actual*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({test_ratio_actual*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def verify_and_fix_arrays(self, arrays: dict) -> dict:
        """
        Verify arrays for NaN/Inf and fix if needed.
        Arrays should already be numeric (float32) at this point.
        
        Args:
            arrays: Dictionary of array names and values
            
        Returns:
            Dictionary of fixed arrays
        """
        print("\n--- Final Data Verification ---")
        fixed_arrays = {}
        
        for name, array in arrays.items():
            # Ensure array is numeric
            if array.dtype == 'object' or not np.issubdtype(array.dtype, np.number):
                print(f"⚠ {name}: Array has non-numeric dtype ({array.dtype}), converting...")
                # Convert to numeric, coercing errors to NaN
                array = pd.DataFrame(array).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
            
            # Check for NaN/Inf (now safe since array is numeric)
            try:
                nan_count = np.isnan(array).sum()
                inf_count = np.isinf(array).sum()
            except (TypeError, ValueError):
                # Fallback: convert to float32 and try again
                array = array.astype(np.float32)
                nan_count = np.isnan(array).sum()
                inf_count = np.isinf(array).sum()
            
            if nan_count > 0 or inf_count > 0:
                print(f"⚠ {name}: Found {nan_count} NaN and {inf_count} Inf values")
                array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"  ✓ Fixed: Replaced NaN/Inf with 0")
            else:
                print(f"✓ {name}: No NaN or Inf values")
            
            fixed_arrays[name] = array
            print(f"  Shape: {array.shape}, Dtype: {array.dtype}")
        
        return fixed_arrays
    
    def save_data(self, train: np.ndarray, val: np.ndarray, test: np.ndarray, 
                  class_names: np.ndarray, output_dir: str = ".") -> None:
        """
        Save preprocessed data to numpy files.
        
        Args:
            train: Training data array
            val: Validation data array
            test: Test data array
            class_names: Class names array
            output_dir: Directory to save files
        """
        print("\n--- Saving Data ---")
        
        os.makedirs(output_dir, exist_ok=True)
        
        files_saved = []
        for name, array in [('train', train), ('val', val), ('test', test)]:
            filepath = os.path.join(output_dir, f"{name}.npy")
            np.save(filepath, array)
            files_saved.append(filepath)
            print(f"  Saved: {filepath} ({array.shape})")
        
        class_names_path = os.path.join(output_dir, "class_names.npy")
        np.save(class_names_path, class_names)
        files_saved.append(class_names_path)
        print(f"  Saved: {class_names_path}")
        
        print(f"\n✓ Saved {len(files_saved)} files to '{output_dir}'")
    
    def print_statistics(self, y_train: np.ndarray, y_val: np.ndarray, 
                        y_test: np.ndarray) -> None:
        """
        Print class distribution statistics.
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
        """
        print("\n--- Class Distribution Statistics ---")
        
        def count_classes(y, name):
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n{name.upper()} set:")
            for class_idx, count in zip(unique, counts):
                # Convert class_idx to integer for array indexing
                class_idx_int = int(np.round(class_idx))  # Use round to handle any float precision issues
                
                # Safely get class name
                if self.class_names is not None and 0 <= class_idx_int < len(self.class_names):
                    class_name = self.class_names[class_idx_int]
                else:
                    class_name = f"Unknown({class_idx_int})"
                
                percentage = (count / len(y)) * 100
                print(f"  Class {class_idx_int} ({class_name}): {count} samples ({percentage:.2f}%)")
        
        count_classes(y_train, "train")
        count_classes(y_val, "validation")
        count_classes(y_test, "test")
    
    def run(self, output_dir: str = ".", print_stats: bool = True) -> dict:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            output_dir: Directory to save output files
            print_stats: Whether to print class distribution statistics
            
        Returns:
            Dictionary containing all preprocessed data and metadata
        """
        try:
            # Load data
            self.load_csv_files()
            
            # Handle missing values
            self.handle_missing_values()
            
            # Handle infinity values
            self.handle_infinity_values()
            
            # Encode labels
            X, y_encoded, class_names = self.encode_labels()
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y_encoded)
            
            # Reshape labels
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            
            # Convert features to numeric arrays (handle any remaining object columns)
            print("\n--- Converting features to numeric arrays ---")
            X_train_numeric = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
            X_val_numeric = pd.DataFrame(X_val).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
            X_test_numeric = pd.DataFrame(X_test).apply(pd.to_numeric, errors='coerce').values.astype(np.float32)
            
            # Ensure labels are also float32 for consistency
            y_train = y_train.astype(np.float32)
            y_val = y_val.astype(np.float32)
            y_test = y_test.astype(np.float32)
            
            # Combine features and labels
            train = np.concatenate((X_train_numeric, y_train), axis=1)
            val = np.concatenate((X_val_numeric, y_val), axis=1)
            test = np.concatenate((X_test_numeric, y_test), axis=1)
            
            print("✓ All arrays converted to float32")
            
            # Verify and fix arrays
            arrays = {'train': train, 'val': val, 'test': test}
            fixed_arrays = self.verify_and_fix_arrays(arrays)
            train, val, test = fixed_arrays['train'], fixed_arrays['val'], fixed_arrays['test']
            
            # Print statistics
            if print_stats:
                self.print_statistics(y_train, y_val, y_test)
            
            # Save data
            self.save_data(train, val, test, class_names, output_dir)
            
            # Print summary
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            
            print("\n" + "="*50)
            print("--- Preprocessing Complete ---")
            print(f"Total preprocessing time: {elapsed_time:.2f} seconds")
            print("="*50)
            
            return {
                'train': train,
                'val': val,
                'test': test,
                'class_names': class_names,
                'label_encoder': self.label_encoder,
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
        except Exception as e:
            print(f"\n❌ ERROR: Preprocessing failed: {e}")
            raise


def main():
    """Main function to run the preprocessing."""
    # Configuration
    CSV_PATH = "csv/"
    OUTPUT_DIR = "."
    TEST_SIZE = 0.2
    VAL_SIZE = 0.5  # Proportion of test set to use for validation
    RANDOM_STATE = 42
    
    # Create preprocessor and run
    preprocessor = DataPreprocessor(
        csv_path=CSV_PATH,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE
    )
    
    result = preprocessor.run(output_dir=OUTPUT_DIR, print_stats=True)
    
    return result


if __name__ == "__main__":
    main()

