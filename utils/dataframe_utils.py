import pandas as pd

input_path = './submission/submission.csv'
target_path = './submission/submission_binarize.csv'

def binarize(input_path, target_path):
    # Load the CSV
    df = pd.read_csv(input_path)

    # # Sort by the 'id' column (assuming it's named 'id')
    # df = df.sort_values(by='unique_id')
    df['gender'] = df['gender'].apply(lambda x: 1 if x >= 0.5 else 0)
    df['hold racket handed'] = df['hold racket handed'].apply(lambda x: 1 if x >= 0.5 else 0)

    # Save back to CSV
    df.to_csv(target_path, index=False)
    
def sort_by_id(input_path, target_path):
    # Load the CSV
    df = pd.read_csv(input_path)

    # # Sort by the 'id' column (assuming it's named 'id')
    df = df.sort_values(by='unique_id')

    # Save back to CSV
    df.to_csv(target_path, index=False)
