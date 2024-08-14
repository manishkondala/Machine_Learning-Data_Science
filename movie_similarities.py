import pandas as pd

r_cols = ['user-id', 'movie-id', 'rating']
ratings = pd.read_csv('C:/Users/Lenovo/Downloads/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie-id']
movies = pd.read_csv('C:/Users/Lenovo/Downloads/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

rating_main = pd.merge(movies, ratings)

print(rating_main)

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()

