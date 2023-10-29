import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as sk_text
import sklearn.metrics.pairwise as sk_pairwise
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# ------------------------CONTENT BASED RECOMMENDATION------------------------------------------------#
path = 'C:\\Users\\dabro\\Desktop\\Studia\\INZ\\data\\archive\\'
df_movie = pd.read_csv(path+"movies_metadata.csv", low_memory=False)
# take only half of the dataset
# df_movie = df_movie[:10000]
df_credits = pd.read_csv(path+"credits.csv")
df_keywords = pd.read_csv(path+"keywords.csv")

# change "id" column name to "movieId"
df_movie = df_movie.rename(columns={'id': 'movieId'})
df_credits = df_credits.rename(columns={'id': 'movieId'})
df_keywords = df_keywords.rename(columns={'id': 'movieId'})

# change id to int
df_movie['movieId'] = pd.to_numeric(df_movie['movieId'], errors='coerce')
df_movie['movieId'] = df_movie['movieId'].fillna(0).astype(np.int64)

# merge df_movie, df_credits, df_keywords
df_movie = df_movie.merge(df_credits, on='movieId')
df_movie = df_movie.merge(df_keywords, on='movieId')

# let only the columns we need
df_movie = df_movie[['movieId', 'title', 'overview',
                     'genres', 'keywords', 'cast', 'crew']]

features = ['cast', 'crew', 'keywords', 'genres']

# convert string to list


def literal_eval(x):
    try:
        return eval(x)
    except:
        return np.nan


# apply literal_eval to features
for feature in features:
    if feature == 'crew':
        df_movie[feature] = df_movie[feature].apply(literal_eval).apply(lambda x: [
            i['name'] for i in x if i['job'] == 'Director'] if isinstance(x, list) else [])
    else:
        df_movie[feature] = df_movie[feature].apply(literal_eval).apply(
            lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# convert all strings to lowercase and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# apply clean_data to features
for feature in features:
    df_movie[feature] = df_movie[feature].apply(clean_data)

    # combine all features into one column


def create_combined_column(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['crew']) + ' ' + ' '.join(x['genres'])


df_movie['combined'] = df_movie.apply(create_combined_column, axis=1)

# take only the columns we need
df_movie = df_movie[['movieId', 'title', 'combined']]
df_movie.dropna(inplace=True)
df_movie.drop_duplicates(subset=['title'], inplace=True)

# convert combined column from text to feature vector
vectorizer = sk_text.TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_movie['combined'])
cosine_sim = sk_pairwise.cosine_similarity(X, X)


def get_CB_recommendations(title, cosine_sim=cosine_sim, top_n=30):
    idx = df_movie[df_movie['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
        1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations_df = pd.DataFrame({
        'title': df_movie['title'].iloc[movie_indices],
        'similarity_score': [x[1] for x in sim_scores]
    })

    return recommendations_df


def get_CB_recommendations_for_two_users(title1, title2, cosine_sim=cosine_sim, top_n=30):
    idx1 = df_movie[df_movie['title'] == title1].index[0]
    idx2 = df_movie[df_movie['title'] == title2].index[0]

    # Calculate cosine similarity between the two provided movies
    combined_similarity = cosine_sim[idx1] * cosine_sim[idx2]

    # Get movie indices sorted by combined similarity
    sim_scores = list(enumerate(combined_similarity))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
        1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations_df = pd.DataFrame({
        'title': df_movie['title'].iloc[movie_indices],
        'similarity_score': [x[1] for x in sim_scores]
    })

    return recommendations_df

# --------------------------------------------Item-based CF ---------------------------------------------------



df_rating = pd.read_csv(path+"ratings.csv")
df_rating = df_rating.drop(columns=['timestamp'])

# merge the two dataframes on the movieId column and id column
df_movie_rating = pd.merge(df_movie, df_rating, on='movieId')

# drop the rows with null values
df_movie_rating = df_movie_rating.dropna()

# smaller dataset by users who have rated a minimum number of movies
filtered_users = df_movie_rating['userId'].value_counts()
filtered_users = filtered_users[filtered_users >= 50].index
df_movie_rating = df_movie_rating[df_movie_rating['userId'].isin(
    filtered_users)]

filtered_movies = df_movie_rating['movieId'].value_counts()
filtered_movies = filtered_movies[filtered_movies >= 100].index
df_movie_rating = df_movie_rating[df_movie_rating['movieId'].isin(
    filtered_movies)]

# scale the rating column to be between 0 and 1
min_rating = df_movie_rating['rating'].min()
max_rating = df_movie_rating['rating'].max()

# Normalize ratings to the range [0, 1]
df_movie_rating['normalized_rating'] = (
    df_movie_rating['rating'] - min_rating) / (max_rating - min_rating)

# pivot the dataframe to get the movie titles as the columns and the userId as the rows and the ratings as the values
df_movie_rating_pivot = df_movie_rating.pivot_table(
    index='movieId', columns='userId', values='normalized_rating')

# fill the null values with 0
df_movie_rating_pivot = df_movie_rating_pivot.fillna(0)

movie_similarity = cosine_similarity(df_movie_rating_pivot)
similar_movies_df = pd.DataFrame(
    movie_similarity, index=df_movie_rating_pivot.index, columns=df_movie_rating_pivot.index)

# apply k nearest neighbors to find the most similar movies

knn = NearestNeighbors(n_neighbors=20, metric='cosine',
                       algorithm='brute', n_jobs=-1)
knn.fit(similar_movies_df)


def get_IBCF_recommendations(movie_name, top_n=30):
    target_movie = df_movie[df_movie['title'] == movie_name]
    # print(target_movie)
    if not target_movie.empty:
        target_movie_id = target_movie['movieId'].values[0]
        # print(target_movie_id)
        similar_movies = similar_movies_df[target_movie_id].sort_values(
            ascending=False)
        # print(similar_movies)
        similar_movies = similar_movies.drop(
            target_movie_id)  # Remove the target movie
        # print(similar_movies)
        top_similar_movies = similar_movies.head(top_n)
        # print(top_similar_movies)

        recommendations_df = pd.DataFrame({
            'title': df_movie['title'].loc[top_similar_movies.index],
            'similarity_score': top_similar_movies.values
        })

        # Reset the index to make 'movieId' a regular column
        recommendations_df.reset_index(inplace=True)

        return recommendations_df
    else:
        return "Movie not found in the dataset."


def get_knn_IBCF_recommendations(movie_name, top_n=30):
    target_movie = df_movie[df_movie['title'] == movie_name]
    # print(target_movie)
    if not target_movie.empty:
        target_movie_id = target_movie['movieId'].values[0]
        # print(target_movie_id)
        distances, indices = knn.kneighbors(
            [similar_movies_df[target_movie_id]], n_neighbors=top_n + 1)
        # print(distances)
        # print(indices)
        recommended_movie_indices = indices[0][1:]

        recommendations_df = pd.DataFrame({
            'title': df_movie['title'].loc[recommended_movie_indices],
            'similarity_score': distances[0][1:]
        })
        # Sort the DataFrame by similarity score in descending order
        recommendations_df = recommendations_df.sort_values(
            by='similarity_score', ascending=False)
        # Reset the index to make 'movieId' a regular column
        recommendations_df.reset_index(inplace=True)

        return recommendations_df[['title', 'similarity_score']].head(top_n)
    else:
        return "Movie not found in the dataset."


# --------------------------------------------hybrid CF ---------------------------------------------------

# weighted hybrid CF function to combine content based and item based collaborative filtering using a weights parameter

def get_hybrid_recommendations(title, cosine_sim=cosine_sim, top_n=30, CB_weight=0.4, IBCF_weight=0.6):
    CB_recommendation = get_CB_recommendations(title, cosine_sim, top_n=100)
    IBCF_recommendation = get_IBCF_recommendations(title, top_n=100)

    # combine the two dataframes
    combined_recommendations = pd.merge(
        CB_recommendation, IBCF_recommendation, how='outer', on='title').fillna(0)

    # calculate the weighted average of the two similarity scores
    combined_recommendations['weighted_average'] = CB_recommendation['similarity_score'] * CB_weight \
        + IBCF_recommendation['similarity_score'] * IBCF_weight

    # sort the dataframe by the weighted average in descending order
    combined_recommendations = combined_recommendations.sort_values(
        by='weighted_average', ascending=False)

    return combined_recommendations.head(top_n)
