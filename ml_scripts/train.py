#!/usr/bin/env python3
"""
Training Script for Content Moderation Model
===========================================

This script trains a logistic regression model for content moderation
using TF-IDF features and provides comprehensive evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import argparse
import os
import json
from pathlib import Path

def model_fn(model_dir):
    """Load model for SageMaker inference."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def predict_fn(input_data, model):
    """Make predictions using the model for SageMaker inference."""
    print(f"Predict input: {input_data}, type: {type(input_data)}")
    
    # Ensure input_data is a list of strings
    if isinstance(input_data, str):
        input_data = [input_data]
    elif not isinstance(input_data, list):
        input_data = [str(input_data)]
    
    try:
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)
        # Return probability of toxic class (class 1)
        toxicity_scores = probabilities[:, 1].tolist()
        
        # If single prediction, return single value
        if len(toxicity_scores) == 1:
            return toxicity_scores[0]
        else:
            return toxicity_scores
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0.0 if len(input_data) == 1 else [0.0] * len(input_data)

def input_fn(input_data, content_type):
    """Parse input data for SageMaker inference."""
    print(f"Input data type: {type(input_data)}, Content type: {content_type}")
    
    if content_type == 'text/csv':
        # Handle CSV input
        if isinstance(input_data, str):
            return [input_data.strip()]
        return input_data
    elif content_type == 'application/json':
        # Handle JSON input
        import json
        if isinstance(input_data, str):
            data = json.loads(input_data)
        else:
            data = input_data
        
        if isinstance(data, dict):
            return [data.get('text', '')]
        elif isinstance(data, list):
            return data
        else:
            return [str(data)]
    else:
        # Default: treat as text
        if isinstance(input_data, str):
            return [input_data]
        return input_data

def train_model(train_path, test_path, model_dir, hyperparameters=None):
    """
    Train the content moderation model
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data  
        model_dir (str): Directory to save the trained model
        hyperparameters (dict): Training hyperparameters
    
    Returns:
        dict: Training metrics and model information
    """
    
    print("🚀 Starting model training...")
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Model output: {model_dir}")
    
    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Load training and test data
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print(f"✅ Data loaded - Train: {train_data.shape}, Test: {test_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    
    # Validate required columns
    required_cols = ['comment_text', 'toxic']
    for df_name, df in [('train', train_data), ('test', test_data)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} data missing columns: {missing_cols}")
    
    # Prepare data
    X_train = train_data['comment_text'].values
    y_train = train_data['toxic'].values
    X_test = test_data['comment_text'].values
    y_test = test_data['toxic'].values
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"📊 Training toxic ratio: {y_train.mean():.3f}")
    print(f"📊 Test toxic ratio: {y_test.mean():.3f}")
    
    # Set hyperparameters
    default_hyperparams = {
        'tfidf_max_features': 10000,
        'tfidf_ngram_range': (1, 2),
        'lr_C': 1.0,
        'lr_max_iter': 1000,
        'cv_folds': 5
    }
    
    if hyperparameters:
        default_hyperparams.update(hyperparameters)
    
    hyperparams = default_hyperparams
    print(f"🔧 Hyperparameters: {hyperparams}")
    
    # Create model pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=hyperparams['tfidf_max_features'],
            ngram_range=hyperparams['tfidf_ngram_range'],
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            C=hyperparams['lr_C'],
            max_iter=hyperparams['lr_max_iter'],
            random_state=42
        ))
    ])
    
    # Train the model
    print("🎯 Training model...")
    model.fit(X_train, y_train)
    print("✅ Model training completed")
    
    # Perform cross-validation
    print("🔄 Performing cross-validation...")
    cv = StratifiedKFold(n_splits=hyperparams['cv_folds'], shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_probabilities = model.predict_proba(X_train)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Generate classification reports
    train_report = classification_report(y_train, train_predictions, output_dict=True)
    test_report = classification_report(y_test, test_predictions, output_dict=True)
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train, train_predictions).tolist()
    test_cm = confusion_matrix(y_test, test_predictions).tolist()
    
    # Training results
    results = {
        'hyperparameters': hyperparams,
        'metrics': {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'cv_mean_accuracy': float(cv_mean),
            'cv_std_accuracy': float(cv_std),
            'cv_scores': cv_scores.tolist()
        },
        'classification_reports': {
            'train': train_report,
            'test': test_report
        },
        'confusion_matrices': {
            'train': train_cm,
            'test': test_cm
        },
        'data_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_toxic_ratio': float(y_train.mean()),
            'test_toxic_ratio': float(y_test.mean())
        }
    }
    
    # Print results
    print(f"\n📈 Training Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   CV Mean Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
    
    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")
    
    # Save hyperparameters for SageMaker
    hyperparams_path = os.path.join(model_dir, "hyperparameters.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # SageMaker environment arguments
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    
    # Hyperparameters
    parser.add_argument("--tfidf-max-features", type=int, default=10000)
    parser.add_argument("--tfidf-ngram-range", type=str, default="1,2")
    parser.add_argument("--lr-C", type=float, default=1.0)
    parser.add_argument("--lr-max-iter", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=5)
    
    args = parser.parse_args()
    
    # Parse ngram range
    ngram_range = tuple(map(int, args.tfidf_ngram_range.split(',')))
    
    # Prepare hyperparameters
    hyperparameters = {
        'tfidf_max_features': args.tfidf_max_features,
        'tfidf_ngram_range': ngram_range,
        'lr_C': args.lr_C,
        'lr_max_iter': args.lr_max_iter,
        'cv_folds': args.cv_folds
    }
    
    # Find data files - handle SageMaker channel structure
    # In SageMaker, train channel goes to /opt/ml/input/data/train/
    # and test channel goes to /opt/ml/input/data/test/
    
    train_dir = args.train
    test_dir = args.test if hasattr(args, 'test') and args.test else args.train
    
    # Look for data files in multiple possible locations
    possible_train_files = [
        os.path.join(train_dir, "train_data.csv"),
        os.path.join(train_dir, "data", "train_data.csv"),
        # If no separate test channel, both files might be in train channel
        os.path.join(train_dir, "test_data.csv") if train_dir == test_dir else None
    ]
    
    possible_test_files = [
        os.path.join(test_dir, "test_data.csv"),
        os.path.join(test_dir, "data", "test_data.csv"),
        # If test channel same as train, look in train directory
        os.path.join(train_dir, "test_data.csv")
    ]
    
    # Remove None values
    possible_train_files = [f for f in possible_train_files if f is not None]
    
    # Find actual files
    train_file = None
    for filepath in possible_train_files:
        if os.path.exists(filepath):
            train_file = filepath
            break
    
    test_file = None  
    for filepath in possible_test_files:
        if os.path.exists(filepath):
            test_file = filepath
            break
    
    # Debug: list all available files
    print("🔍 Available files in training directories:")
    for directory in [train_dir, test_dir]:
        if os.path.exists(directory):
            print(f"  Directory: {directory}")
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f"    {file_path}")
    
    if not train_file:
        print(f"❌ Training file not found in any of these locations:")
        for filepath in possible_train_files:
            print(f"   {filepath}")
        exit(1)
    
    if not test_file:
        print(f"❌ Test file not found in any of these locations:")
        for filepath in possible_test_files:
            print(f"   {filepath}")
        exit(1)
    
    print(f"✅ Found training file: {train_file}")
    print(f"✅ Found test file: {test_file}")
    
    try:
        results = train_model(train_file, test_file, args.model_dir, hyperparameters)
        print("🎊 Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        exit(1)
