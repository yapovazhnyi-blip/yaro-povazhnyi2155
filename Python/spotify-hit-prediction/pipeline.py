import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, threshold=70):
    df = df.copy()

    # Target
    df['hit'] = (df['popularity'] >= threshold).astype(int)

    # One-hot encode genre BEFORE dropping
    genre_dummies = pd.get_dummies(df['track_genre'], prefix='track_genre')
    df = pd.concat([df, genre_dummies], axis=1)

    # Drop useless columns
    df = df.drop(columns=['track_id', 'track_name', 'album_name',
                           'artists', 'popularity', 'track_genre'])

    # Boolean to int
    df['explicit'] = df['explicit'].astype(int)

    return df