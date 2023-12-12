from utils import content_based as cb
from utils import collaborative_filtering as cf
from utils import stats_visualization as sv
from utils import data_preprocessing as dpp
from utils.database_connection import df_movie, df_credits, df_keywords, df_rating
from loguru import logger
import pandas as pd

df_movie = dpp.preprocess_movie_data(df_movie, df_credits, df_keywords)
df_movie_rating_pivot = dpp.preprocess_rating_data(df_rating, df_movie)


def get_hybrid_recommendations(title, top_n=30, CB_weight=0.4, IBCF_weight = 0.6):
    logger.info(f" title: {title}")
    CB_recommendation = cb.get_CB_recommendations(df_movie, title, top_n = 100)
    IBCF_recommendation = cf.get_IBCF_recommendations(df_movie, df_movie_rating_pivot,title, top_n = 100)
    combined_recommendations = pd.merge(CB_recommendation, IBCF_recommendation,how='outer', on='title').fillna(0)
    
    combined_recommendations['weighted_average'] = CB_recommendation['similarity_score'] * CB_weight \
                                                + IBCF_recommendation['similarity_score'] * IBCF_weight
    
    combined_recommendations['release_date'] = combined_recommendations['release_date_x'].fillna(combined_recommendations['release_date_y'])
    combined_recommendations['overview'] = combined_recommendations['overview_x'].fillna(combined_recommendations['overview_y'])
    combined_recommendations['poster_url'] = combined_recommendations['poster_url_x'].fillna(combined_recommendations['poster_url_y'])
    combined_recommendations['genres'] = combined_recommendations['genres_x'].fillna(combined_recommendations['genres_y'])
    combined_recommendations['cast'] = combined_recommendations['cast_x'].fillna(combined_recommendations['cast_y'])
    combined_recommendations['director'] = combined_recommendations['director_x'].fillna(combined_recommendations['director_y'])
    combined_recommendations['popularity'] = combined_recommendations['popularity_x'].fillna(combined_recommendations['popularity_y'])
    combined_recommendations = combined_recommendations.drop(columns=['popularity_x', 'popularity_y','director_x', 'director_y','cast_x','cast_y','release_date_x', 'release_date_y','overview_x', 'overview_y','poster_url_x', 'poster_url_y','genres_x', 'genres_y'])
    combined_recommendations = combined_recommendations.sort_values(by='weighted_average', ascending=False)
    combined_recommendations = combined_recommendations.head(top_n)

    sv.get_pie_plot_of_genres(combined_recommendations)
    sv.get_bar_plot_of_release_dates(combined_recommendations)
    sv.get_horizontal_bar_plot_of_cast(combined_recommendations)
    sv.get_directors_network_graph(combined_recommendations)
    sv.get_most_popular_movies(combined_recommendations)
       
    return combined_recommendations


#--------------------------------------------hybrid CF for two users---------------------------------------------------
def get_hybrid_recommendations_for_two_users(title1, title2, top_n=30, CB_weight=0.4, IBCF_weight = 0.6):
    logger.info(f" title1: {title1}")
    logger.info(f" title2: {title2}")
    CB_recommendation = cb.get_CB_recommendations_for_two_users(df_movie,title1, title2, top_n = 30)
    IBCF_recommendation = cf.get_KNN_IBCF_group_recommendations(df_movie, df_movie_rating_pivot, [title1, title2], top_n = 30)
    combined_recommendations = pd.merge(CB_recommendation, IBCF_recommendation,how='outer', on='title').fillna(0)
    
    combined_recommendations['weighted_average'] = CB_recommendation['similarity_score'] * CB_weight \
                                                + IBCF_recommendation['similarity_score'] * IBCF_weight
                                                
    combined_recommendations['release_date'] = combined_recommendations['release_date_x'].fillna(combined_recommendations['release_date_y'])
    combined_recommendations['overview'] = combined_recommendations['overview_x'].fillna(combined_recommendations['overview_y'])
    combined_recommendations['poster_url'] = combined_recommendations['poster_url_x'].fillna(combined_recommendations['poster_url_y'])
    combined_recommendations['genres'] = combined_recommendations['genres_x'].fillna(combined_recommendations['genres_y'])
    combined_recommendations['cast'] = combined_recommendations['cast_x'].fillna(combined_recommendations['cast_y'])
    combined_recommendations['director'] = combined_recommendations['director_x'].fillna(combined_recommendations['director_y'])
    combined_recommendations['popularity'] = combined_recommendations['popularity_x'].fillna(combined_recommendations['popularity_y'])
    combined_recommendations = combined_recommendations.drop(columns=['popularity_x', 'popularity_y','director_x', 'director_y','cast_x','cast_y','release_date_x', 'release_date_y','overview_x', 'overview_y','poster_url_x', 'poster_url_y','genres_x', 'genres_y'])
    combined_recommendations = combined_recommendations.sort_values(by='weighted_average', ascending=False)
    combined_recommendations = combined_recommendations.head(top_n)
    
    sv.get_pie_plot_of_genres(combined_recommendations)
    sv.get_bar_plot_of_release_dates(combined_recommendations)
    sv.get_horizontal_bar_plot_of_cast(combined_recommendations)
    sv.get_directors_network_graph(combined_recommendations)
    sv.get_most_popular_movies(combined_recommendations)
    
    return combined_recommendations

def get_recommendations_for_multiple_users(movie_names, top_n = 30):
    
    combined_recommendations = cf.get_KNN_IBCF_group_recommendations(df_movie, df_movie_rating_pivot,movie_names, top_n = 100)
    combined_recommendations = combined_recommendations.head(top_n)
    logger.info(f"combined_recommendations: {combined_recommendations}")

    sv.get_pie_plot_of_genres(combined_recommendations)
    sv.get_bar_plot_of_release_dates(combined_recommendations)
    sv.get_horizontal_bar_plot_of_cast(combined_recommendations)
    sv.get_directors_network_graph(combined_recommendations)
    sv.get_most_popular_movies(combined_recommendations)
    
    return combined_recommendations
