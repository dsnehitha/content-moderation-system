import pandas as pd
import boto3
import numpy as np
from sklearn.model_selection import train_test_split

def create_sample_data():

    toxic_comments = [
        "I hate you!",
        "You are a terrible person.",
        "This is the worst thing I've ever seen.",
        "I wish you would just go away.",
        "You are so stupid.",
        "I can't stand you.",
    ]

    non_toxic_comments = [
        "I really appreciate your help.",
        "Thank you for being so kind.",
        "This is a great day!",
        "I love working with you.",
        "You are amazing!",
        "Keep up the good work!",
    ]

    comments = toxic_comments * 100 + non_toxic_comments * 100
    labels = [1] * (len(toxic_comments) * 100) + [0] * (len(non_toxic_comments) * 100)

    data = pd.DataFrame({
        'comment_text': comments,
        'toxic': labels
    })

    return data.sample(frac=1).reset_index(drop=True)

df = create_sample_data()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("Training data length:", len(train_df))
print("Test data length:", len(test_df))