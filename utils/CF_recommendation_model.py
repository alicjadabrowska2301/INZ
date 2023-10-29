import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

df_movie = pd.read_csv("archive2/movies_metadata.csv",  engine='python' , encoding='utf8', error_bad_lines=False)
df_movie = df_movie[['id', 'title']]
df_movie = df_movie.rename(columns={'id': 'movieId'})

df_rating = pd.read_csv("archive2/ratings_small.csv")
df_rating = df_rating.drop(columns=['timestamp'])

#convert the movieId column in the df_movie dataframe to numeric
df_movie['movieId'] = pd.to_numeric(df_movie['movieId'], errors='coerce')

#merge the two dataframes on the movieId column and id column
df_movie_rating = pd.merge(df_movie, df_rating, on='movieId')
#drop the rows with null values
df_movie_rating = df_movie_rating.dropna()

#scale the rating column to be between 0 and 1
min_rating = df_movie_rating['rating'].min()
max_rating = df_movie_rating['rating'].max()

# Normalize ratings to the range [0, 1]
df_movie_rating['normalized_rating'] = (df_movie_rating['rating'] - min_rating) / (max_rating - min_rating)
#check which film has the most ratings
df_movie_rating.groupby('title')['rating'].count().sort_values(ascending=True)
#delete the films with less than 100 ratings from the dataframe
df_movie_rating = df_movie_rating.groupby('title').filter(lambda x: x['rating'].count() >= 10)
#pivot the dataframe to get the movie titles as the columns and the userId as the rows and the ratings as the values
df_movie_rating_pivot = df_movie_rating.pivot_table(index='userId', columns='title', values='normalized_rating')
#fill the null values with 0
df_movie_rating_pivot.fillna(0, inplace=True)
#split the data into training and testing sets
train, test = train_test_split(df_movie_rating_pivot.values, test_size=0.2, random_state=42)

class myRecommender:
  def __init__(self, epochs = 100, n_latent_factors = 8, lmbda = 0.1, learning_rate = 0.001, w_0 = 0.02):
    self.epochs = epochs
    self.n_latent_factors = n_latent_factors
    self.lmbda = lmbda
    self.learning_rate = learning_rate
    self.w_0 = w_0

  def fit_the_data(self, Xtrain):
    self.n_users, self.n_movies = Xtrain.shape[0], Xtrain.shape[1]
    #print(self.n_users, self.n_movies)
    self.U = np.random.rand(self.n_users, self.n_latent_factors)
    self.V = np.random.rand(self.n_movies, self.n_latent_factors)
    self.Vt = self.V.T
    #print(self.U.shape, self.V.shape, self.Vt.shape)
    #print(self.U, "\n", self.Vt)

    self.training_process = []

    for epoch in range(self.epochs):
      for u in range(self.n_users):
        for m in range(self.n_movies):
          #error = Xtrain[u,m] - np.dot(self.U[u, :], self.Vt[:,m])
          #print(np.dot(self.U[u, :], self.Vt[:,m]), error)
          error = Xtrain[u,m] - np.dot(self.U[u, :], self.V[m,:])

          if Xtrain[u,m] != 0:
            weight = 1
          else:
            weight = self.w_0

          self.U[u,:] += self.learning_rate * (error * self.V[m,:] - self.lmbda * self.U[u,:]) * weight
          self.V[m,:] += self.learning_rate * (error * self.U[u,:] - self.lmbda * self.V[m,:]) * weight

      train_err = sqrt(mean_squared_error(np.dot(self.U, self.Vt), Xtrain))
      #test_err = sqrt(mean_squared_error(np.dot(self.U, self.Vt), xtest))
      print("Epoch: ", epoch , ", train rmse: ", train_err)
      #print(self.U, self.V, np.dot(self.U, self.Vt) )
      self.training_process.append((train_err))

    return self

  def predict(self, Xtrain, user):
    predicted_rating_dict = {}
    #get the prediction matrix
    rating_prediction_matrix = np.dot(self.U, self.Vt)
    #choose row for given user
    user_row = Xtrain[user, :]
    #get the index of movies that user did not watch -> rating = 0
    index = np.where(user_row == 0)[0]
    #return predicted movie ratings for items that given user did not watch
    predicted_rating = rating_prediction_matrix[user, index].flatten()
    #print(predicted_rating)
    for i,r in zip(index, predicted_rating):
      predicted_rating_dict[i] = r
    return predicted_rating_dict
  
  def plot_training_process(self):
    plt.plot(self.training_process)
    plt.xlabel("Number of Epochs")
    plt.ylabel("RMSE")
    plt.show()

  def recommend(self, Xtrain, user):
    predicted_rating_dict = self.predict(Xtrain, user)
    #return the top 10 movie titles with the highest predicted ratings

model = myRecommender()
model.fit_the_data(train)

#save the model
import pickle
pickle.dump(model, open('finalized_model.sav', 'wb'))


#load the model
import pickle
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
user_id = 3
rating_predictions = loaded_model.predict(train, user_id) #get the predicted ratings for the given user
recommendations = sorted(rating_predictions.items(), key=lambda x: x[1], reverse=True)[:10]
recommendations