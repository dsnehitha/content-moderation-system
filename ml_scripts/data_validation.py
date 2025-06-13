#!/usr/bin/env python3
"""
Data Validation Script for SageMaker Pipeline
=============================================

This script validates and prepares training data for the content moderation model.
It performs quality checks, data cleaning, and outputs validated datasets.
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path

def validate_and_prepare_data(input_path, output_path):
    """
    Validate training data and prepare for training
    
    Args:
        input_path (str): Path to input data directory
        output_path (str): Path to output directory for validated data
    
    Returns:
        bool: True if validation successful, raises exception otherwise
    """
    
    print("🔍 Starting data validation and preparation...")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Ensure paths exist
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # List all files in input path
    print("Files in input path:")
    for file_path in input_path.rglob("*"):
        if file_path.is_file():
            print(f"  {file_path}")
    
    # Look for data files
    train_file = input_path / "train_data.csv"
    test_file = input_path / "test_data.csv"
    
    print(f"Looking for train file: {train_file}")
    print(f"Looking for test file: {test_file}")
    
    # Check if files exist
    if not train_file.exists():
        print(f"❌ Train file not found at {train_file}")
        sys.exit(1)
    
    if not test_file.exists():
        print(f"❌ Test file not found at {test_file}")
        sys.exit(1)
    
    # Load data files
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        print(f"✅ Data files loaded successfully")
        print(f"   Train shape: {train_data.shape}")
        print(f"   Test shape: {test_data.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data files: {e}")
        sys.exit(1)
    
    # Validation report structure
    validation_report = {
        "validation_passed": True,
        "issues": [],
        "statistics": {},
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Check required columns
    required_cols = ['comment_text', 'toxic']
    for df_name, df in [('train', train_data), ('test', test_data)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_report["issues"].append(f"{df_name} missing columns: {missing_cols}")
            validation_report["validation_passed"] = False
    
    if not validation_report["validation_passed"]:
        print("❌ Data validation failed: Missing required columns")
        with open(output_path / "validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2)
        sys.exit(1)
    
    # Check for missing values in required columns
    train_missing = train_data[required_cols].isnull().sum().sum()
    test_missing = test_data[required_cols].isnull().sum().sum()
    
    if train_missing > 0 or test_missing > 0:
        validation_report["issues"].append(f"Missing values: train={train_missing}, test={test_missing}")
        if train_missing > len(train_data) * 0.1 or test_missing > len(test_data) * 0.1:
            validation_report["validation_passed"] = False
    
    # Check data size
    if len(train_data) < 100:
        validation_report["issues"].append(f"Training data too small: {len(train_data)} samples")
        validation_report["validation_passed"] = False
    
    # Check target distribution
    if 'toxic' in train_data.columns:
        toxic_ratio = train_data['toxic'].mean()
        if toxic_ratio < 0.01 or toxic_ratio > 0.99:
            validation_report["issues"].append(f"Imbalanced dataset: toxic ratio = {toxic_ratio:.3f}")
    
    # Calculate statistics
    validation_report["statistics"] = {
        "train_samples": int(len(train_data)),
        "test_samples": int(len(test_data)),
        "train_toxic_ratio": float(train_data['toxic'].mean()) if 'toxic' in train_data.columns else 0,
        "test_toxic_ratio": float(test_data['toxic'].mean()) if 'toxic' in test_data.columns else 0,
        "train_missing_values": int(train_missing),
        "test_missing_values": int(test_missing)
    }
    
    # Save validation report
    with open(output_path / "validation_report.json", 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    if not validation_report["validation_passed"]:
        print("❌ Data validation failed:", validation_report["issues"])
        sys.exit(1)
    
    print("✅ Data validation passed - preparing data...")
    
    # Clean and prepare data
    try:
        # Basic text cleaning
        train_data['comment_text'] = train_data['comment_text'].astype(str).str.strip()
        test_data['comment_text'] = test_data['comment_text'].astype(str).str.strip()
        
        # Remove empty text
        train_data = train_data[train_data['comment_text'] != '']
        test_data = test_data[test_data['comment_text'] != '']
        
        # Remove rows with missing values in required columns
        train_data = train_data.dropna(subset=required_cols)
        test_data = test_data.dropna(subset=required_cols)
        
        # Save cleaned data
        train_data.to_csv(output_path / "train_data.csv", index=False)
        test_data.to_csv(output_path / "test_data.csv", index=False)
        
        # Update statistics after cleaning
        validation_report["statistics"]["final_train_samples"] = int(len(train_data))
        validation_report["statistics"]["final_test_samples"] = int(len(test_data))
        
        # Save final report
        with open(output_path / "validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print(f"✅ Data preparation completed:")
        print(f"   📊 Training samples: {len(train_data)}")
        print(f"   📊 Test samples: {len(test_data)}")
        print(f"   📊 Training toxic ratio: {train_data['toxic'].mean():.3f}")
        print(f"   📊 Test toxic ratio: {test_data['toxic'].mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        validation_report["validation_passed"] = False
        validation_report["issues"].append(f"Data preparation failed: {str(e)}")
        
        with open(output_path / "validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/opt/ml/processing/input",
                       help="Path to input data directory")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output",
                       help="Path to output directory for validated data")
    
    args = parser.parse_args()
    
    try:
        validate_and_prepare_data(args.input_path, args.output_path)
        print("✅ Data validation completed successfully")
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        sys.exit(1)
