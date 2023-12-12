import pandas as pd

def preprocess_movie_data(df_movie, df_credits, df_keywords):
    df_movie = df_movie.merge(df_credits, on='movieId')
    df_movie = df_movie.merge(df_keywords, on='movieId')

    df_movie['release_date'] = df_movie['release_date'].astype(str).str[:-2]
    df_movie = df_movie[['movieId','title','overview','genres','keywords','cast','director', 'poster_path', 'release_date', 'popularity']]    
        
    df_movie['combined_features'] = df_movie['genres'] + "-" + df_movie['keywords'] + "-" + df_movie['cast'] + "-" + df_movie['director']
    df_movie['combined_features'] = df_movie['combined_features'].str.lower().str.replace(" ", "").str.replace(",", " ").str.replace("-", " ")
    df_movie['combined_features'] = df_movie['combined_features'].values.astype('U')
    return df_movie


def preprocess_rating_data(df_rating, df_movie):
    df_rating = df_rating.drop(columns=['timestamp'])
    df_movie_rating = pd.merge(df_rating, df_movie, on='movieId')
    df_movie_rating_pivot = df_movie_rating.pivot_table(index='movieId', columns='userId', values='normalized_rating')
    df_movie_rating_pivot = df_movie_rating_pivot.fillna(0)
    return df_movie_rating_pivot