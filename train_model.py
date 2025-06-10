import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session
import pandas as pd


sagemaker_session = Session()
region = boto3.Session(). region_name
role = 'arn:aws:iam::806070028440:role/service-role/AmazonSageMaker-ExecutionRole-20250525T123664'

training_script = '''
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import joblib
    import argparse
    import os

    def model_fn(model_dir):
        """Load model for inference."""
        model = joblib.load(os.path.join(model_dir, "model.joblib"))
        return model

    def predict_fn(input_data, model):
        """Make predictions using the model."""
        return model.predict(input_data)[:1]

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
        parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))

        args = parser.parse_args()

        train_data = pd.read_csv(os.path.join(args.train, "train_data.csv"))

        pipeline = Pipeline([
            ('tfidf', TfidVectorizer(max_features=10000, stop_words="english")),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        #train model
        X = train_data['comment_text']
        y = train_data['toxic']

        pipeline.fit(X, y)

        joblib.dump(pipeline, os.path.join(args.model_dir, "model.joblib"))
        print("Model training completed!")
    '''

with open('train.py', 'w') as f:
    f.write(training_script)
    