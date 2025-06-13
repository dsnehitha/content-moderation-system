
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
import joblib
import argparse
import os
import json

def model_fn(model_dir):
    """Load model for inference."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def predict_fn(input_data, model):
    """Make predictions using the model."""
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
    """Parse input data for inference."""

    print(f"Input data type: {type(input_data)}, Content type: {content_type}")
    
    if content_type == 'text/csv':
        # Handle CSV input
        if isinstance(input_data, bytes):
            input_data = input_data.decode('utf-8')
        # Split by lines and take non-empty lines
        lines = [line.strip() for line in input_data.split('\n') if line.strip()]
        return lines
    elif content_type == 'application/json':
        # Handle JSON input
        if isinstance(input_data, bytes):
            input_data = input_data.decode('utf-8')
        data = json.loads(input_data)
        if isinstance(data, dict) and 'text' in data:
            return [data['text']]
        elif isinstance(data, list):
            return data
        else:
            return [str(data)]
    else:
        # Default handling - treat as text
        if isinstance(input_data, bytes):
            input_data = input_data.decode('utf-8')
        return [input_data.strip()]

def output_fn(prediction, accept):
    """Format the prediction output."""

    if accept == 'application/json':
        return json.dumps(prediction), accept
    elif accept == 'text/csv':
        if isinstance(prediction, list):
            return ','.join(map(str, prediction)), accept
        else:
            return str(prediction), accept
    else:
        return json.dumps(prediction), 'application/json'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))

    args = parser.parse_args()

    print("Loading training data...")
    train_data = pd.read_csv(os.path.join(args.train, "train_data.csv"))
    
    print("Creating content moderation pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000, 
            stop_words="english",
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,
            max_df=0.95
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            class_weight='balanced',
            C=1.0,
            max_iter=1000
        ))
    ])
    
    # Prepare data
    X = train_data['comment_text']
    y = train_data['toxic']
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    # Perform cross-validation on training data
    cv_scores = cross_validate(
        pipeline, X_train, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scoring
    )
    
    print("Cross-validation results:")
    for metric in scoring:
        scores = cv_scores[f'test_{metric}']
        print(f"  {metric.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    print(f"\nMean CV accuracy: {cv_scores['test_accuracy'].mean():.4f} (+/- {cv_scores['test_accuracy'].std() * 2:.4f})")
    
    # Train model on full training set
    print("\nTraining model on training set...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_predictions = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_predictions, 
                              target_names=['Non-toxic', 'Toxic']))
    
    # Retrain on full dataset for production model
    print("\nRetraining on full dataset for production deployment...")
    pipeline.fit(X, y)
    
    # Final evaluation metrics
    final_train_predictions = pipeline.predict(X_train)
    final_train_accuracy = accuracy_score(y_train, final_train_predictions)
    print(f"Training accuracy: {final_train_accuracy:.4f}")
    
    # Initialize test_accuracy to None
    test_accuracy = None
    
    # If external test data is available, evaluate on it too
    test_file_path = os.path.join(args.test, "test_data.csv") if args.test else None
    if test_file_path and os.path.exists(test_file_path):
        print("\nEvaluating on external test data...")
        test_data = pd.read_csv(test_file_path)
        X_test = test_data['comment_text']
        y_test = test_data['toxic']
        
        test_predictions = pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        print("\nExternal Test Classification Report:")
        print(classification_report(y_test, test_predictions, 
                                  target_names=['Non-toxic', 'Toxic']))
    
    # Save model
    print(f"Saving model to {args.model_dir}...")
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(args.model_dir, "model.joblib"))
    
    try:
        feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()[:100].tolist()
    except AttributeError:
        feature_names = pipeline.named_steps['tfidf'].get_feature_names()[:100]
    
    model_info = {
        'cv_mean_accuracy': cv_scores['test_accuracy'].mean(),
        'cv_std_accuracy': cv_scores['test_accuracy'].std(),
        'cv_mean_precision': cv_scores['test_precision'].mean(),
        'cv_mean_recall': cv_scores['test_recall'].mean(),
        'cv_mean_f1': cv_scores['test_f1'].mean(),
        'cv_mean_roc_auc': cv_scores['test_roc_auc'].mean(),
        'validation_accuracy': val_accuracy,
        'train_accuracy': final_train_accuracy,
        'test_accuracy': test_accuracy,
        'features': feature_names,
        'model_type': 'LogisticRegression + TfidfVectorizer',
        'cross_validation': True,
        'cv_folds': 5
    }

    with open(os.path.join(args.model_dir, "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model training completed successfully!")
    print(f"📁 Model saved to: {args.model_dir}/model.joblib")
    print(f"📊 Cross-validation accuracy: {cv_scores['test_accuracy'].mean():.4f} (+/- {cv_scores['test_accuracy'].std() * 2:.4f})")
    print(f"📊 Validation accuracy: {val_accuracy:.4f}")
    print(f"📊 Final training accuracy: {final_train_accuracy:.4f}")
    if test_accuracy is not None:
        print(f"📊 External test accuracy: {test_accuracy:.4f}")
    else:
        print("📊 No external test data provided")
    