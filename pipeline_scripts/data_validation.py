
        import pandas as pd
        import numpy as np
        import json
        import argparse
        import os
        import sys

        def validate_and_prepare_data(input_path, output_path):
            """Validate training data and prepare for training"""
            
            print("🔍 Starting data validation and preparation...")
            print(f"Input path: {input_path}")
            print(f"Output path: {output_path}")
            
            # List all files in input path
            print("Files in input path:")
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    print(f"  {os.path.join(root, file)}")
            
            # Load data files
            train_file = os.path.join(input_path, "train_data.csv")
            test_file = os.path.join(input_path, "test_data.csv")
            
            print(f"Looking for train file: {train_file}")
            print(f"Looking for test file: {test_file}")
            
            if not os.path.exists(train_file):
                print(f"❌ Train file not found at {train_file}")
                sys.exit(1)
            
            if not os.path.exists(test_file):
                print(f"❌ Test file not found at {test_file}")
                sys.exit(1)
            
            try:
                train_data = pd.read_csv(train_file)
                test_data = pd.read_csv(test_file)
                print(f"✅ Data files loaded successfully")
                print(f"   Train shape: {train_data.shape}")
                print(f"   Test shape: {test_data.shape}")
                
            except Exception as e:
                print(f"❌ Error loading data files: {e}")
                sys.exit(1)
            
            # Basic validation checks
            validation_report = {
                "validation_passed": True,
                "issues": [],
                "statistics": {}
            }
            
            # Check required columns
            required_cols = ['comment_text', 'toxic']
            for df_name, df in [('train', train_data), ('test', test_data)]:
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    validation_report["issues"].append(f"{df_name} missing columns: {missing_cols}")
                    validation_report["validation_passed"] = False
            
            # Check for missing values
            if validation_report["validation_passed"]:
                train_missing = train_data[required_cols].isnull().sum().sum()
                test_missing = test_data[required_cols].isnull().sum().sum()
                
                if train_missing > 0 or test_missing > 0:
                    validation_report["issues"].append(f"Missing values: train={train_missing}, test={test_missing}")
                    validation_report["validation_passed"] = False
            
            # Check data size
            if len(train_data) < 100:
                validation_report["issues"].append(f"Training data too small: {len(train_data)} samples")
                validation_report["validation_passed"] = False
            
            # Calculate statistics
            validation_report["statistics"] = {
                "train_samples": int(len(train_data)),
                "test_samples": int(len(test_data)),
                "train_toxic_ratio": float(train_data['toxic'].mean()),
                "test_toxic_ratio": float(test_data['toxic'].mean())
            }
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Save validation report
            with open(os.path.join(output_path, "validation_report.json"), 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            if not validation_report["validation_passed"]:
                print("❌ Data validation failed:", validation_report["issues"])
                sys.exit(1)
            
            # Clean and save data
            print("✅ Data validation passed - preparing data...")
            
            try:
                # Basic cleaning
                train_data['comment_text'] = train_data['comment_text'].astype(str).str.strip()
                test_data['comment_text'] = test_data['comment_text'].astype(str).str.strip()
                
                # Remove empty text
                train_data = train_data[train_data['comment_text'] != '']
                test_data = test_data[test_data['comment_text'] != '']
                
                # Save cleaned data
                train_data.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
                test_data.to_csv(os.path.join(output_path, "test_data.csv"), index=False)
                
                print(f"✅ Data preparation completed:")
                print(f"   📊 Training samples: {len(train_data)}")
                print(f"   📊 Test samples: {len(test_data)}")
                print(f"   📊 Training toxic ratio: {train_data['toxic'].mean():.3f}")
                
            except Exception as e:
                print(f"❌ Error during data preparation: {e}")
                sys.exit(1)
            
            return True

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--input-path", type=str, default="/opt/ml/processing/input")
            parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
            
            args = parser.parse_args()
            
            try:
                validate_and_prepare_data(args.input_path, args.output_path)
                print("✅ Data validation completed successfully")
            except Exception as e:
                print(f"❌ Data validation failed: {e}")
                sys.exit(1)
        