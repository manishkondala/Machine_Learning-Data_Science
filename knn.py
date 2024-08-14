import pandas as pd
import numpy as np
from scipy import spatial
import operator

read_col = ['user_id', 'movie_id', 'rating']
movie_data = pd.read_csv(r"C:\Users\Lenovo\Downloads\u.data", sep='\t', names=read_col, usecols=range(3))

movie_prp = movie_data.groupby('movie_id').agg({'rating': ['size', 'mean']})

movieNumRatings = pd.DataFrame(movie_prp['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

movie_dict = {}

with open(r"C:\Users\Lenovo\Downloads\u.item") as f:
    for line in f:
        fields = line.rstrip('\n').split('|')
        movie_ID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = list(map(int, genres))
        movie_dict[movie_ID] = (
            name, 
            np.array(genres), 
            movieNormalizedNumRatings.loc[movie_ID].get('size'), 
            movie_prp.loc[movie_ID].rating.get('mean')
        )

def compute_distance(a, b):
    genre_a = a[1]
    genre_b = b[1]
    genre_distance = spatial.distance.cosine(genre_a, genre_b)
    pop_a = a[2]
    pop_b = b[2]
    pop_dist = abs(pop_a - pop_b)
    return genre_distance + pop_dist

def get_neighbour(movie_ID, k):
    distances = []
    for movie in movie_dict:
        if movie != movie_ID:
            dist = compute_distance(movie_dict[movie_ID], movie_dict[movie])
            distances.append((movie, dist))

    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours

k = 4
avg_rating = 0
neighbours = get_neighbour(1, k)
for neighbour in neighbours:
    avg_rating += movie_dict[neighbour][3]
    print(movie_dict[neighbour][0] + " " + str(movie_dict[neighbour][3]))

avg_rating /= float(k)
print("Average rating of neighbours:", avg_rating)
print("****************************************************************")
print(avg_rating)
print("****************************************************************")
print(movie_dict[1])