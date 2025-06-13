import pandas as pd
import boto3
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random
import json

def create_enhanced_sample_data():
    """Create a more diverse and realistic dataset for content moderation"""
    
    # Expanded toxic comments with various categories
    toxic_comments = [
        # Hate speech
        "I hate you!", "You are a terrible person.", "I wish you would just go away.",
        "You are so stupid.", "I can't stand you.", "You disgust me.",
        "Go kill yourself.", "You're worthless.", "Nobody likes you.",
        
        # Harassment
        "Stop posting here, loser.", "You don't belong here.", 
        "Shut up and leave.", "You're such an idiot.", "Get out of here.",
        
        # Threats
        "I'm going to find you.", "You better watch out.", 
        "I'll make you pay for this.", "You're going to regret this.",
        
        # Profanity-laden attacks
        "F*** you and everything you stand for.", "You're a piece of s***.",
        "Go to hell.", "You're absolutely pathetic.",
        
        # Discrimination
        "People like you shouldn't exist.", "Your kind doesn't belong here.",
        "You people are all the same.", "Stay in your own country."
    ]
    
    # Expanded non-toxic comments
    non_toxic_comments = [
        # Positive feedback
        "I really appreciate your help.", "Thank you for being so kind.",
        "This is a great day!", "I love working with you.",
        "You are amazing!", "Keep up the good work!",
        
        # Constructive criticism
        "I disagree with your point, but I respect your opinion.",
        "Could you provide more evidence for this claim?",
        "I think there might be a better approach to this.",
        "Have you considered this alternative perspective?",
        
        # Neutral comments
        "What time does the meeting start?", "Can you share the document?",
        "I'll be there in 10 minutes.", "Thanks for the update.",
        "The weather is nice today.", "How was your weekend?",
        
        # Support and encouragement
        "You can do this!", "Don't give up, you're doing great.",
        "I believe in you.", "That's a really good point.",
        "I learned something new from your post.", "Well done!",
        
        # Questions and discussions
        "What do you think about this topic?", "Can you explain this better?",
        "I'm curious about your experience with this.",
        "Has anyone tried this approach before?",
        "What are the pros and cons of this solution?"
    ]
    
    # Create balanced dataset with some variation
    num_samples_per_class = 500
    
    # Generate variations by combining and slightly modifying base comments
    extended_toxic = []
    extended_non_toxic = []
    
    for i in range(num_samples_per_class):
        base_toxic = random.choice(toxic_comments)
        if random.random() < 0.3:
            variations = [base_toxic.upper(), base_toxic + "!!!", "Really, " + base_toxic.lower()]
            extended_toxic.append(random.choice(variations))
        else:
            extended_toxic.append(base_toxic)
    
    for i in range(num_samples_per_class):
        base_non_toxic = random.choice(non_toxic_comments)
        if random.random() < 0.2:
            variations = ["Please, " + base_non_toxic.lower(), base_non_toxic + " Please.", 
                         "I think " + base_non_toxic.lower()]
            extended_non_toxic.append(random.choice(variations))
        else:
            extended_non_toxic.append(base_non_toxic)
    
    comments = extended_toxic + extended_non_toxic
    labels = [1] * len(extended_toxic) + [0] * len(extended_non_toxic)
    
    data = pd.DataFrame({
        'comment_text': comments,
        'toxic': labels
    })
    
    return data.sample(frac=1).reset_index(drop=True)

def upload_to_s3(df_train, df_test, bucket_name):
    """Upload training and test data to S3"""
    try:
        s3_client = boto3.client('s3')
        
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Created S3 bucket: {bucket_name}")
        except s3_client.exceptions.BucketAlreadyOwnedByYou:
            print(f"Bucket {bucket_name} already exists")
        except Exception as e:
            if "BucketAlreadyExists" in str(e):
                print(f"Bucket {bucket_name} already exists")
            else:
                print(f"Error creating bucket: {e}")
        
        # Upload from the data directory
        train_path = os.path.join('data', 'train_data.csv')
        test_path = os.path.join('data', 'test_data.csv')
        
        s3_client.upload_file(train_path, bucket_name, 'data/raw/train_data.csv')
        s3_client.upload_file(test_path, bucket_name, 'data/raw/test_data.csv')
        
        print(f"Data uploaded to s3://{bucket_name}/data/raw/")
        return True
        
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return False

if __name__ == "__main__":
    print("Creating enhanced sample data...")
    df = create_enhanced_sample_data()
    
    # Create data directory if it doesn't exist
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Split the data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['toxic'])
    
    # Save files to data directory
    train_path = os.path.join(data_dir, 'train_data.csv')
    test_path = os.path.join(data_dir, 'test_data.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data length: {len(train_df)}")
    print(f"Test data length: {len(test_df)}")
    print(f"Training toxic ratio: {train_df['toxic'].mean():.2f}")
    print(f"Test toxic ratio: {test_df['toxic'].mean():.2f}")
    
    try:
        with open('aws_config.json', 'r') as f:
            config = json.load(f)
        bucket_name = config['datastore_bucket']
        print(f"\nUploading to S3 bucket: {bucket_name}...")
        upload_to_s3(train_df, test_df, bucket_name)
    except FileNotFoundError:
        print("\nSkipping S3 upload - run setup_infrastructure.py first")
        print(f"Data saved locally in '{data_dir}/' folder as train_data.csv and test_data.csv")
    except Exception as e:
        print(f"\nError uploading to S3: {e}")
        print(f"Data saved locally in '{data_dir}/' folder as train_data.csv and test_data.csv")
