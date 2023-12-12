import sklearn.feature_extraction.text as sk_text
import sklearn.metrics.pairwise as sk_pairwise
import pandas as pd


def create_CB_model(df_movie):
    vectorizer = sk_text.TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df_movie['combined_features'])
    cosine_sim = sk_pairwise.cosine_similarity(X, X)
    return cosine_sim

def get_CB_recommendations(df_movie, title, top_n=30):
    cosine_sim = create_CB_model(df_movie)
    idx = df_movie[df_movie['title'] == title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations_df = pd.DataFrame({
        'title': df_movie['title'].iloc[movie_indices],
        'similarity_score': [x[1] for x in sim_scores],
        'poster_url': df_movie['poster_path'].iloc[movie_indices],
        'genres': df_movie['genres'].iloc[movie_indices],
        'release_date': df_movie['release_date'].iloc[movie_indices],
        'overview': df_movie['overview'].iloc[movie_indices],
        'cast': df_movie['cast'].iloc[movie_indices],
        'director': df_movie['director'].iloc[movie_indices],
        'popularity': df_movie['popularity'].iloc[movie_indices]
    })
    
    return recommendations_df


def get_CB_recommendations_for_two_users(df_movie,title1, title2, top_n=30):
    cosine_sim = create_CB_model(df_movie)
    idx1 = df_movie[df_movie['title'] == title1].index[0]
    idx2 = df_movie[df_movie['title'] == title2].index[0]
    
    combined_similarity = cosine_sim[idx1] * cosine_sim[idx2]

    sim_scores = list(enumerate(combined_similarity))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    recommendations_df = pd.DataFrame({
        'title': df_movie['title'].iloc[movie_indices],
        'similarity_score': [x[1] for x in sim_scores],
        'poster_url': df_movie['poster_path'].iloc[movie_indices],
        'genres': df_movie['genres'].iloc[movie_indices],
        'release_date': df_movie['release_date'].iloc[movie_indices],
        'overview': df_movie['overview'].iloc[movie_indices], 
        'cast': df_movie['cast'].iloc[movie_indices],
        'director': df_movie['director'].iloc[movie_indices],
        'popularity': df_movie['popularity'].iloc[movie_indices]
    })
    
    return recommendations_df
