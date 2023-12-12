import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def IBCF_model(df_movie_rating_pivot):
    movie_similarity = cosine_similarity(df_movie_rating_pivot)
    similar_movies_df = pd.DataFrame(movie_similarity, index=df_movie_rating_pivot.index, columns=df_movie_rating_pivot.index)
    knn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute', n_jobs=-1)
    knn.fit(df_movie_rating_pivot.values)
    return similar_movies_df, knn

def get_IBCF_recommendations(df_movie, df_movie_rating_pivot, movie_name, top_n=30):
    similar_movies_df = IBCF_model(df_movie_rating_pivot)[0]
    df_movie.set_index('movieId', inplace=True)
    target_movie = df_movie[df_movie['title'] == movie_name]
    if not target_movie.empty:
        target_movie_id = df_movie[df_movie['title'] == movie_name].index[0]
        similar_movies = similar_movies_df[target_movie_id].sort_values(ascending=False)
        similar_movies = similar_movies.drop(target_movie_id) 
        top_similar_movies = similar_movies.head(top_n)
        
        recommendations_df = pd.DataFrame({
            'title': df_movie['title'].loc[top_similar_movies.index],
            'similarity_score': top_similar_movies.values,
            'poster_url': df_movie['poster_path'].loc[top_similar_movies.index],
            'genres': df_movie['genres'].loc[top_similar_movies.index],
            'release_date': df_movie['release_date'].loc[top_similar_movies.index],
            'overview': df_movie['overview'].loc[top_similar_movies.index],
            'cast': df_movie['cast'].loc[top_similar_movies.index],
            'director': df_movie['director'].loc[top_similar_movies.index],
            'popularity': df_movie['popularity'].loc[top_similar_movies.index]
        })
        
        recommendations_df.reset_index(inplace=True)
        df_movie.reset_index(inplace=True)
        
        return recommendations_df
    else:
        return "Movie not found in the dataset."
    
def get_knn_IBCF_recommendations(df_movie, df_movie_rating_pivot ,movie_name, top_n=30):
    knn = IBCF_model(df_movie_rating_pivot)[1]
    df_movie.set_index('movieId', inplace=True)
    target_movie = df_movie[df_movie['title'] == movie_name]
    if not target_movie.empty:
        target_movie_id = df_movie[df_movie['title'] == movie_name].index[0]
        
        distances, indices = knn.kneighbors(df_movie_rating_pivot.loc[target_movie_id].values.reshape(1, -1), n_neighbors=top_n)
        similarity_score = 1 - distances.flatten()
        recommendations_df = pd.DataFrame({
            'title': df_movie['title'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
            'similarity_score': similarity_score[1:],
            'poster_url': df_movie['poster_path'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
            'genres': df_movie['genres'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
            'release_date': df_movie['release_date'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
            'overview': df_movie['overview'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
            'cast': df_movie['cast'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
            'director': df_movie['director'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
            'popularity': df_movie['popularity'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:]
        })
        
        recommendations_df = recommendations_df.sort_values(by='similarity_score', ascending=False)
        recommendations_df.reset_index(inplace=True)
        df_movie.reset_index(inplace=True)
        
        return recommendations_df
    else:
        return "Movie not found in the dataset."
    
    
def get_KNN_IBCF_group_recommendations(df_movie, df_movie_rating_pivot, movie_names, top_n=50):
    knn = IBCF_model(df_movie_rating_pivot)[1]
    df_movie.set_index('movieId', inplace=True)
    movie_ids = []
    for movie_name in movie_names:
        movie_id = df_movie[df_movie['title'] == movie_name].index[0]
        movie_ids.append(movie_id)
    
    distances, indices = knn.kneighbors(df_movie_rating_pivot.loc[movie_ids].values.sum(axis=0).reshape(1, -1), n_neighbors=top_n)
    similarity_score = 1 - distances.flatten()
    
    recommendations_df = pd.DataFrame({
        'title': df_movie['title'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
        'similarity_score': similarity_score[1:],
        'poster_url': df_movie['poster_path'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
        'genres': df_movie['genres'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
        'release_date': df_movie['release_date'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
        'overview': df_movie['overview'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
        'cast': df_movie['cast'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
        'director': df_movie['director'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:],
        'popularity': df_movie['popularity'].loc[df_movie_rating_pivot.iloc[indices[0]].index.tolist()][1:]
        
    })
    
    recommendations_df = recommendations_df.sort_values(by='similarity_score', ascending=False)
    
    recommendations_df.reset_index(inplace=True)
    df_movie.reset_index(inplace=True)
    
    return recommendations_df
    
 
