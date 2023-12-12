import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
import pandas as pd

def get_pie_plot_of_genres(df):
    df['genre'] = df['genres'].astype(str)
    df['genre'] = df['genre'].apply(lambda x: x.split(","))
    df['genre'] = df['genre'].apply(lambda x: [i.replace(" ", "") for i in x])

    genres = []
    for i in df['genre']:
        genres.extend(i)
    genres = list(set(genres))

    genres_count = {}
    for i in genres:
        genres_count[i] = 0
    for i in df['genre']:
        for j in i:
            genres_count[j] += 1

    plt.figure(figsize=(7, 7))   
    colors = cm.rainbow(np.linspace(0, 1, len(genres_count.keys())))
    wedges, texts, autotexts = plt.pie(
        genres_count.values(),
        labels=genres_count.keys(),
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        colors=colors
    )   
    for text in texts + autotexts:
        text.set_fontsize(8)
  
    plt.legend(labels=genres_count.keys(), loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=9)
    plt.title('Genres percentage distribution', fontsize=15, pad=15, color='#333333', weight='bold')
    plt.savefig('C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\genres.png', bbox_inches='tight')
    plt.close()
    
def get_bar_plot_of_release_dates(df):
    df['year'] = df['release_date'].fillna(0).astype(int)
    df_counts = df['year'].value_counts().sort_index().reset_index()
    df_counts.columns = ['Year', 'Number of Movies']
    df_filtered = df_counts[df_counts['Number of Movies'] > 0]
    
    plt.rcParams['figure.figsize'] = (10, 5)
    ax = df_filtered.plot(kind='bar', x='Year', y='Number of Movies', colormap = cm.rainbow, legend=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 0.1), ha='center', color = '#333333' , weight='bold')
    
    vals = ax.get_yticks()
    ax.set_yticklabels([int(x) for x in vals])
    ax.set_xlabel('Year of release', labelpad=15, color='#333333')
    ax.set_ylabel('Number of Movies', labelpad=15, color='#333333')
    ax.set_title('Recommended Movies Release Year', pad=15, color='#333333', weight='bold', fontsize=15)
    plt.savefig('C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\release_dates.png', bbox_inches='tight')
    plt.close()
    
def get_horizontal_bar_plot_of_cast(df):
    df['casts'] = df['cast'].apply(lambda x: x.split(","))
    cast = []
    for i in df['casts']:
        cast.extend(i) 

    cast = [i.strip() for i in cast]
    cast_count = pd.Series(cast).value_counts()
    
    cast_count_filtered = cast_count.sort_values(ascending=False).head(10)
    cast_count_filtered = cast_count_filtered[::-1]
    
    plt.figure(figsize=(10, 5))
    ax = cast_count_filtered.plot(kind='barh', color = '#2adddd')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#EEEEEE')
    ax.yaxis.grid(False)
    
    for p in ax.patches:
        ax.annotate(str(p.get_width()), (p.get_width() + 0.1, p.get_y() + p.get_height() / 2), va='center', color='black', weight='bold')
    
    vals = ax.get_xticks()
    ax.set_xticklabels([int(x) for x in vals], color='#333333')
    
    ax.set_ylabel('Actors', labelpad=15, color='#333333')
    ax.set_xlabel('Number of Movies', labelpad=15, color='#333333')
    ax.set_title('Top 10 Actors From Recommendations', pad=15, color='#333333', weight='bold', fontsize=15)
    plt.savefig('C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\actors.png', bbox_inches='tight')
    plt.close()
    


def get_directors_network_graph(df):
    df['directors'] = df['director'].astype(str)
    df['directors'] = df['directors'].apply(lambda x: x.split(","))
    df['directors'] = df['directors'].apply(lambda x: [i.strip() for i in x])
    
    directors = []
    for i in df['directors']:
        directors.extend(i)
    directors_count = pd.Series(directors).value_counts()
    directors_count_filtered = directors_count[directors_count >= 2]
    
    G = nx.Graph()
    for index, row in df.iterrows():
        for director in row['directors']:
            if director in directors_count_filtered.index:
                G.add_node(director, bipartite=0)  # 0 for directors
                G.add_node(row['title'], bipartite=1)  # 1 for movies
                G.add_edge(director, row['title'])

    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, k=0.6)
    directors_nodes = {node for node, data in G.nodes(data=True) if data['bipartite'] == 0}
    movies_nodes = {node for node, data in G.nodes(data=True) if data['bipartite'] == 1}
    nx.draw_networkx_nodes(G, pos, nodelist=directors_nodes, node_color='#8000ff', node_size=1000, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=movies_nodes, node_color='#2adddd', node_size=300, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    plt.title('Most frequently appearing directors on Director-Movie Graph', fontsize=15, fontweight='bold', pad=15, color='#333333')
    plt.axis('off')
    plt.savefig('C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\directors.png', bbox_inches='tight')
    plt.close()
    
def get_most_popular_movies(df):
    df_sorted = df.sort_values(by='popularity', ascending=False).head(15)
    my_range = range(1, len(df_sorted) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    plt.hlines(y=my_range, xmin=0, xmax=df_sorted['popularity'], color='#007ACC', alpha=0.4, linewidth=10)
    plt.plot(df_sorted['popularity'], my_range, "o", markersize=10, color='#007ACC', alpha=0.8)

    ax.set_xlabel('Popularity',labelpad=15, color='#333333', fontsize=12)
    ax.set_ylabel('Movie Title',labelpad=15, color='#333333', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.yticks(my_range, df_sorted['title'], fontsize=6, fontweight='black', color='#333333')
    plt.title('Top 15 Most Popular Movies', fontsize=15, fontweight='bold', color='#333333', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds((1, len(my_range)))
    ax.set_xlim(0, max(df_sorted['popularity']) + 1)
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 5))
    
    plt.savefig('C:\\Users\\aldabrow\\Desktop\\PowerBI Excercises\\INZ\\static\\popular_movies.png', bbox_inches='tight')
    plt.close()
    
    