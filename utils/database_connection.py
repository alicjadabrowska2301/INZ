import psycopg2
from sqlalchemy import create_engine, text
import pandas as pd 

host= "localhost"
database="movierecommenderdb"
user= "postgres"
password= "postgres"
port= "5432"

  
alchemyEngine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}')
dbConnection = alchemyEngine.connect()

query_movies = text('SELECT * FROM "Movies"')
query_credits = text('SELECT * FROM "Credits"')
query_keywords = text('SELECT * FROM "Keywords"')
query_ratings = text('SELECT * FROM "Ratings"')
df_movie = pd.read_sql_query(query_movies, dbConnection)
df_credits = pd.read_sql_query(query_credits, dbConnection)
df_keywords = pd.read_sql_query(query_keywords, dbConnection)
df_rating = pd.read_sql_query(query_ratings, dbConnection)

pd.set_option('display.expand_frame_repr', False)
dbConnection.close()