import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as sk_text
import sklearn.metrics.pairwise as sk_pairwise
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import psycopg2
from sqlalchemy import create_engine, text
from matplotlib import cm
import matplotlib.pyplot as plt
from loguru import logger
import networkx as nx


host = "localhost"
database = "movierecommenderdb"
user = "postgres"
password = "postgres"
port = "5432"


alchemyEngine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
)
dbConnection = alchemyEngine.connect()

query_movies = text('SELECT * FROM "Movies"')
query_credits = text('SELECT * FROM "Credits"')
query_keywords = text('SELECT * FROM "Keywords"')
query_ratings = text('SELECT * FROM "Ratings"')
df_movie = pd.read_sql_query(query_movies, dbConnection)
df_credits = pd.read_sql_query(query_credits, dbConnection)
df_keywords = pd.read_sql_query(query_keywords, dbConnection)
df_rating = pd.read_sql_query(query_ratings, dbConnection)

pd.set_option("display.expand_frame_repr", False)
dbConnection.close()

# ------------------------CONTENT BASED RECOMMENDATION------------------------------------------------#

df_movie = df_movie.merge(df_credits, on="movieId")
df_movie = df_movie.merge(df_keywords, on="movieId")

df_movie["release_date"] = df_movie["release_date"].astype(str).str[:-2]
df_movie = df_movie[
    [
        "movieId",
        "title",
        "overview",
        "genres",
        "keywords",
        "cast",
        "director",
        "poster_path",
        "release_date",
        "popularity",
    ]
]

df_movie["combined_features"] = (
    df_movie["genres"]
    + "-"
    + df_movie["keywords"]
    + "-"
    + df_movie["cast"]
    + "-"
    + df_movie["director"]
)
df_movie["combined_features"] = (
    df_movie["combined_features"]
    .str.lower()
    .str.replace(" ", "")
    .str.replace(",", " ")
    .str.replace("-", " ")
)
df_movie["combined_features"] = df_movie["combined_features"].values.astype("U")

vectorizer = sk_text.TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df_movie["combined_features"])
cosine_sim = sk_pairwise.cosine_similarity(X, X)


def get_CB_recommendations(title, cosine_sim=cosine_sim, top_n=30):
    idx = df_movie[df_movie["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations_df = pd.DataFrame(
        {
            "title": df_movie["title"].iloc[movie_indices],
            "similarity_score": [x[1] for x in sim_scores],
            "poster_url": df_movie["poster_path"].iloc[movie_indices],
            "genres": df_movie["genres"].iloc[movie_indices],
            "release_date": df_movie["release_date"].iloc[movie_indices],
            "overview": df_movie["overview"].iloc[movie_indices],
            "cast": df_movie["cast"].iloc[movie_indices],
            "director": df_movie["director"].iloc[movie_indices],
            "popularity": df_movie["popularity"].iloc[movie_indices],
        }
    )

    return recommendations_df


def get_CB_recommendations_for_two_users(
    title1, title2, cosine_sim=cosine_sim, top_n=30
):
    idx1 = df_movie[df_movie["title"] == title1].index[0]
    idx2 = df_movie[df_movie["title"] == title2].index[0]
    
    combined_similarity = cosine_sim[idx1] * cosine_sim[idx2]
    
    sim_scores = list(enumerate(combined_similarity))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    recommendations_df = pd.DataFrame(
        {
            "title": df_movie["title"].iloc[movie_indices],
            "similarity_score": [x[1] for x in sim_scores],
            "poster_url": df_movie["poster_path"].iloc[movie_indices],
            "genres": df_movie["genres"].iloc[movie_indices],
            "release_date": df_movie["release_date"].iloc[movie_indices],
            "overview": df_movie["overview"].iloc[movie_indices],
            "cast": df_movie["cast"].iloc[movie_indices],
            "director": df_movie["director"].iloc[movie_indices],
            "popularity": df_movie["popularity"].iloc[movie_indices],
        }
    )

    return recommendations_df


# --------------------------------------------Item-based CF ---------------------------------------------------
df_rating = df_rating.drop(columns=["timestamp"])
df_movie_rating = pd.merge(df_rating, df_movie, on="movieId")
df_ratings = df_movie_rating.copy()


df_movie_rating_pivot = df_movie_rating.pivot_table(
    index="movieId", columns="userId", values="normalized_rating"
)

df_movie_rating_pivot = df_movie_rating_pivot.fillna(0)

movie_similarity = cosine_similarity(df_movie_rating_pivot)
similar_movies_df = pd.DataFrame(
    movie_similarity,
    index=df_movie_rating_pivot.index,
    columns=df_movie_rating_pivot.index,
)

knn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute", n_jobs=-1)
knn.fit(df_movie_rating_pivot.values)


def get_IBCF_recommendations(movie_name, top_n=30):
    df_movie.set_index("movieId", inplace=True)
    target_movie = df_movie[df_movie["title"] == movie_name]
   
    if not target_movie.empty:
        target_movie_id = df_movie[df_movie["title"] == movie_name].index[0]
        similar_movies = similar_movies_df[target_movie_id].sort_values(ascending=False)
        similar_movies = similar_movies.drop(target_movie_id)  # Remove the target movie
        top_similar_movies = similar_movies.head(top_n)

        recommendations_df = pd.DataFrame(
            {
                "title": df_movie["title"].loc[top_similar_movies.index],
                "similarity_score": top_similar_movies.values,
                "poster_url": df_movie["poster_path"].loc[top_similar_movies.index],
                "genres": df_movie["genres"].loc[top_similar_movies.index],
                "release_date": df_movie["release_date"].loc[top_similar_movies.index],
                "overview": df_movie["overview"].loc[top_similar_movies.index],
                "cast": df_movie["cast"].loc[top_similar_movies.index],
                "director": df_movie["director"].loc[top_similar_movies.index],
                "popularity": df_movie["popularity"].loc[top_similar_movies.index],
            }
        )

        recommendations_df.reset_index(inplace=True)
        df_movie.reset_index(inplace=True)

        return recommendations_df
    else:
        return "Movie not found in the dataset."


def get_knn_IBCF_recommendations(movie_name, top_n=30):
    df_movie.set_index("movieId", inplace=True)
    target_movie = df_movie[df_movie["title"] == movie_name]
    
    if not target_movie.empty:
        target_movie_id = df_movie[df_movie["title"] == movie_name].index[0]
        distances, indices = knn.kneighbors(
            df_movie_rating_pivot.loc[target_movie_id].values.reshape(1, -1),
            n_neighbors=top_n,
        )
        similarity_score = 1 - distances.flatten()
    
        recommendations_df = pd.DataFrame(
            {
                "title": df_movie["title"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
                "similarity_score": similarity_score[1:],
                "poster_url": df_movie["poster_path"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
                "genres": df_movie["genres"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
                "release_date": df_movie["release_date"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
                "overview": df_movie["overview"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
                "cast": df_movie["cast"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
                "director": df_movie["director"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
                "popularity": df_movie["popularity"].loc[
                    df_movie_rating_pivot.iloc[indices[0]].index.tolist()
                ][1:],
            }
        )

        recommendations_df = recommendations_df.sort_values(
            by="similarity_score", ascending=False
        )

        recommendations_df.reset_index(inplace=True)
        df_movie.reset_index(inplace=True)

        return recommendations_df
    else:
        return "Movie not found in the dataset."


def get_KNN_IBCF_group_recommendations(movie_names, top_n=50):
    df_movie.set_index("movieId", inplace=True)

    movie_ids = []
    for movie_name in movie_names:
        movie_id = df_movie[df_movie["title"] == movie_name].index[0]
        movie_ids.append(movie_id)

    distances, indices = knn.kneighbors(
        df_movie_rating_pivot.loc[movie_ids].values.sum(axis=0).reshape(1, -1),
        n_neighbors=top_n,
    )
    similarity_score = 1 - distances.flatten()

    recommendations_df = pd.DataFrame(
        {
            "title": df_movie["title"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
            "similarity_score": similarity_score[1:],
            "poster_url": df_movie["poster_path"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
            "genres": df_movie["genres"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
            "release_date": df_movie["release_date"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
            "overview": df_movie["overview"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
            "cast": df_movie["cast"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
            "director": df_movie["director"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
            "popularity": df_movie["popularity"].loc[
                df_movie_rating_pivot.iloc[indices[0]].index.tolist()
            ][1:],
        }
    )

    recommendations_df = recommendations_df.sort_values(
        by="similarity_score", ascending=False
    )

    recommendations_df.reset_index(inplace=True)
    df_movie.reset_index(inplace=True)

    return recommendations_df


# -------------------------------------------stats-------------------------------------------------------


def get_pie_plot_of_genres(df):
    df["genre"] = df["genres"].astype(str)
    df["genre"] = df["genre"].apply(lambda x: x.split(","))
    df["genre"] = df["genre"].apply(lambda x: [i.replace(" ", "") for i in x])

    genres = []
    for i in df["genre"]:
        genres.extend(i)
    genres = list(set(genres))
    
    genres_count = {}
    for i in genres:
        genres_count[i] = 0
    for i in df["genre"]:
        for j in i:
            genres_count[j] += 1

    plt.figure(figsize=(7, 7))

    colors = cm.rainbow(np.linspace(0, 1, len(genres_count.keys())))
    wedges, texts, autotexts = plt.pie(
        genres_count.values(),
        labels=genres_count.keys(),
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
        colors=colors,
    )

    for text in texts + autotexts:
        text.set_fontsize(8)

    plt.legend(
        labels=genres_count.keys(),
        loc="upper right",
        bbox_to_anchor=(1.4, 1.0),
        fontsize=9,
    )

    plt.title(
        "Genres percentage distribution",
        fontsize=15,
        pad=15,
        color="#333333",
        weight="bold",
    )

    plt.savefig(
        "C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\genres.png",
        bbox_inches="tight",
    )
    plt.close()


def get_bar_plot_of_release_dates(df):
    df["year"] = df["release_date"].fillna(0).astype(int)

    df_counts = df["year"].value_counts().sort_index().reset_index()
    df_counts.columns = ["Year", "Number of Movies"]

    df_filtered = df_counts[df_counts["Number of Movies"] > 0]

    plt.rcParams["figure.figsize"] = (10, 5)

    ax = df_filtered.plot(
        kind="bar", x="Year", y="Number of Movies", colormap=cm.rainbow, legend=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")

    ax.tick_params(bottom=False, left=False)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)

    for p in ax.patches:
        ax.annotate(
            str(p.get_height()),
            (p.get_x() + p.get_width() / 2, p.get_height() + 0.1),
            ha="center",
            color="#333333",
            weight="bold",
        )

    vals = ax.get_yticks()
    ax.set_yticklabels([int(x) for x in vals])

    ax.set_xlabel("Year of release", labelpad=15, color="#333333")
    ax.set_ylabel("Number of Movies", labelpad=15, color="#333333")
    ax.set_title(
        "Recommended Movies Release Year",
        pad=15,
        color="#333333",
        weight="bold",
        fontsize=15,
    )

    plt.savefig(
        "C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\release_dates.png",
        bbox_inches="tight",
    )
    plt.close()


def get_horizontal_bar_plot_of_cast(df):
    # df['casts'] = df['cast'].astype(str)
    # df['casts'] = df['casts'].apply(lambda x: x.split(","))
    df["casts"] = df["cast"].apply(lambda x: x.split(","))
    cast = []
    for i in df["casts"]:
        cast.extend(i)
    cast = [i.strip() for i in cast]
    cast_count = pd.Series(cast).value_counts()
    
    cast_count_filtered = cast_count.sort_values(ascending=False).head(10)
    cast_count_filtered = cast_count_filtered[::-1]

    plt.figure(figsize=(10, 5))
    ax = cast_count_filtered.plot(kind="barh", color="#2adddd")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")

    ax.tick_params(bottom=False, left=False)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#EEEEEE")
    ax.yaxis.grid(False)

    for p in ax.patches:
        ax.annotate(
            str(p.get_width()),
            (p.get_width() + 0.1, p.get_y() + p.get_height() / 2),
            va="center",
            color="black",
            weight="bold",
        )

    vals = ax.get_xticks()
    ax.set_xticklabels([int(x) for x in vals], color="#333333")

    ax.set_ylabel("Actors", labelpad=15, color="#333333")
    ax.set_xlabel("Number of Movies", labelpad=15, color="#333333")
    ax.set_title(
        "Top 10 Actors From Recommendations",
        pad=15,
        color="#333333",
        weight="bold",
        fontsize=15,
    )
    plt.savefig(
        "C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\actors.png",
        bbox_inches="tight",
    )
    plt.close()


def get_directors_network_graph(df):
    df["directors"] = df["director"].astype(str)
    df["directors"] = df["directors"].apply(lambda x: x.split(","))
    df["directors"] = df["directors"].apply(lambda x: [i.strip() for i in x])

    directors = []
    for i in df["directors"]:
        directors.extend(i)
    directors_count = pd.Series(directors).value_counts()

    directors_count_filtered = directors_count[directors_count >= 2]

    G = nx.Graph()

    for index, row in df.iterrows():
        for director in row["directors"]:
            if director in directors_count_filtered.index:
                G.add_node(director, bipartite=0)  # 0 for directors
                G.add_node(row["title"], bipartite=1)  # 1 for movies
                G.add_edge(director, row["title"])

    plt.figure(figsize=(12, 6))

    pos = nx.spring_layout(G, k=0.6)
    directors_nodes = {
        node for node, data in G.nodes(data=True) if data["bipartite"] == 0
    }
    movies_nodes = {node for node, data in G.nodes(data=True) if data["bipartite"] == 1}
    # label_pos = {k: [v[0], v[1] + 0.01] for k, v in pos.items()}

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=directors_nodes,
        node_color="#8000ff",
        node_size=1000,
        alpha=0.7,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=movies_nodes, node_color="#2adddd", node_size=300, alpha=0.7
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    plt.title(
        "Most frequently appearing directors on Director-Movie Graph",
        fontsize=15,
        fontweight="bold",
        pad=15,
        color="#333333",
    )
    plt.axis("off")
    plt.savefig(
        "C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\directors.png",
        bbox_inches="tight",
    )
    plt.close()


def get_most_popular_movies(df):
    df_sorted = df.sort_values(by="popularity", ascending=False).head(15)
    my_range = range(1, len(df_sorted) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.hlines(
        y=my_range,
        xmin=0,
        xmax=df_sorted["popularity"],
        color="#007ACC",
        alpha=0.4,
        linewidth=10,
    )
    
    plt.plot(
        df_sorted["popularity"],
        my_range,
        "o",
        markersize=10,
        color="#007ACC",
        alpha=0.8,
    )

    ax.set_xlabel("Popularity", labelpad=15, color="#333333", fontsize=12)
    ax.set_ylabel("Movie Title", labelpad=15, color="#333333", fontsize=12)

    ax.tick_params(axis="both", which="major", labelsize=10)
    plt.yticks(
        my_range, df_sorted["title"], fontsize=6, fontweight="black", color="#333333"
    )

    plt.title(
        "Top 15 Most Popular Movies",
        fontsize=15,
        fontweight="bold",
        color="#333333",
        pad=15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_bounds((1, len(my_range)))
    ax.set_xlim(0, max(df_sorted["popularity"]) + 1)

    ax.spines["left"].set_position(("outward", 8))
    ax.spines["bottom"].set_position(("outward", 5))

    plt.savefig(
        "C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\popular_movies.png",
        bbox_inches="tight",
    )
    plt.close()


# --------------------------------------------hybrid CF ---------------------------------------------------

def get_hybrid_recommendations(
    title, cosine_sim=cosine_sim, top_n=30, CB_weight=0.2, IBCF_weight=1
):
    CB_recommendation = get_CB_recommendations(title, cosine_sim, top_n=100)
    IBCF_recommendation = get_IBCF_recommendations(title, top_n=100)

    combined_recommendations = pd.merge(
        CB_recommendation, IBCF_recommendation, how="outer", on="title"
    ).fillna(0)

    combined_recommendations["weighted_average"] = (
        CB_recommendation["similarity_score"] * CB_weight
        + IBCF_recommendation["similarity_score"] * IBCF_weight)

    combined_recommendations = combined_recommendations.sort_values(
        by="weighted_average", ascending=False
    )
    combined_recommendations = combined_recommendations.head(top_n)
    
    combined_recommendations["release_date"] = combined_recommendations[
        "release_date_x"
    ].fillna(combined_recommendations["release_date_y"])

    combined_recommendations["overview"] = combined_recommendations[
        "overview_x"
    ].fillna(combined_recommendations["overview_y"])

    combined_recommendations["poster_url"] = combined_recommendations[
        "poster_url_x"
    ].fillna(combined_recommendations["poster_url_y"])

    combined_recommendations["genres"] = combined_recommendations["genres_x"].fillna(
        combined_recommendations["genres_y"]
    )
    logger.info(
        f"combined_recommendations['cast_x']: {combined_recommendations['cast_x']}"
    )
    logger.info(
        f"combined_recommendations['cast_y']: {combined_recommendations['cast_y']}"
    )
    combined_recommendations["cast"] = combined_recommendations["cast_x"].fillna(
        combined_recommendations["cast_y"]
    )
    logger.info(f"combined_recommendations['cast']: {combined_recommendations['cast']}")

    combined_recommendations["director"] = combined_recommendations[
        "director_x"
    ].fillna(combined_recommendations["director_y"])
    combined_recommendations["popularity"] = combined_recommendations[
        "popularity_x"
    ].fillna(combined_recommendations["popularity_y"])

    combined_recommendations = combined_recommendations.drop(
        columns=[
            "popularity_x",
            "popularity_y",
            "director_x",
            "director_y",
            "cast_x",
            "cast_y",
            "release_date_x",
            "release_date_y",
            "overview_x",
            "overview_y",
            "poster_url_x",
            "poster_url_y",
            "genres_x",
            "genres_y",
        ]
    )

    print(combined_recommendations["cast"], type(combined_recommendations["cast"]))

    get_pie_plot_of_genres(combined_recommendations)
    get_bar_plot_of_release_dates(combined_recommendations)
    get_horizontal_bar_plot_of_cast(combined_recommendations)
    get_directors_network_graph(combined_recommendations)
    get_most_popular_movies(combined_recommendations)

    return combined_recommendations


# --------------------------------------------hybrid CF for two users---------------------------------------------------
def get_hybrid_recommendations_for_two_users(
    title1, title2, cosine_sim=cosine_sim, top_n=30, CB_weight=0.2, IBCF_weight=1.0
):
    logger.info(f" title1: {title1}")
    logger.info(f" title2: {title2}")
    CB_recommendation = get_CB_recommendations_for_two_users(
        title1, title2, cosine_sim, top_n=30
    )
    IBCF_recommendation = get_KNN_IBCF_group_recommendations([title1, title2], top_n=30)

    combined_recommendations = pd.merge(
        CB_recommendation, IBCF_recommendation, how="outer", on="title"
    ).fillna(0)

    combined_recommendations["weighted_average"] = (
        CB_recommendation["similarity_score"] * CB_weight
        + IBCF_recommendation["similarity_score"] * IBCF_weight
    )
    
    combined_recommendations["release_date"] = combined_recommendations[
        "release_date_x"
    ].fillna(combined_recommendations["release_date_y"])

    combined_recommendations["overview"] = combined_recommendations[
        "overview_x"
    ].fillna(combined_recommendations["overview_y"])
    
    combined_recommendations["poster_url"] = combined_recommendations[
        "poster_url_x"
    ].fillna(combined_recommendations["poster_url_y"])

    combined_recommendations["genres"] = combined_recommendations["genres_x"].fillna(
        combined_recommendations["genres_y"]
    )
    combined_recommendations["cast"] = combined_recommendations["cast_x"].fillna(
        combined_recommendations["cast_y"]
    )
    combined_recommendations["director"] = combined_recommendations[
        "director_x"
    ].fillna(combined_recommendations["director_y"])
    combined_recommendations["popularity"] = combined_recommendations[
        "popularity_x"
    ].fillna(combined_recommendations["popularity_y"])

    combined_recommendations = combined_recommendations.drop(
        columns=[
            "popularity_x",
            "popularity_y",
            "director_x",
            "director_y",
            "cast_x",
            "cast_y",
            "release_date_x",
            "release_date_y",
            "overview_x",
            "overview_y",
            "poster_url_x",
            "poster_url_y",
            "genres_x",
            "genres_y",
        ]
    )
    
    combined_recommendations = combined_recommendations.sort_values(
        by="weighted_average", ascending=False
    )
    combined_recommendations = combined_recommendations.head(top_n)

    get_pie_plot_of_genres(combined_recommendations)
    get_bar_plot_of_release_dates(combined_recommendations)
    get_horizontal_bar_plot_of_cast(combined_recommendations)
    get_directors_network_graph(combined_recommendations)
    get_most_popular_movies(combined_recommendations)

    return combined_recommendations


def get_recommendations_for_multiple_users(movie_names, top_n=30):
    combined_recommendations = get_KNN_IBCF_group_recommendations(
        movie_names, top_n=100
    )
    combined_recommendations = combined_recommendations.head(top_n)
    logger.info(f"combined_recommendations: {combined_recommendations}")

    get_pie_plot_of_genres(combined_recommendations)
    get_bar_plot_of_release_dates(combined_recommendations)
    get_horizontal_bar_plot_of_cast(combined_recommendations)
    get_directors_network_graph(combined_recommendations)
    get_most_popular_movies(combined_recommendations)

    return combined_recommendations
