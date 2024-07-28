from numpy import random, array
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random

def create_cluster(n, k):
    random.seed(10)
    points_per_cluster = float(n)/k
    x = []
    for i in range(k):
        income_centroid = random.uniform(2000, 2000)
        age_centroid = random.uniform(20, 70)
        for j in range(int(points_per_cluster)):
            x.append([random.normal(income_centroid, 10000), random.normal(age_centroid, 2)])
    x = array(x)
    return x

data = create_cluster(1000, 4)
model = KMeans(n_clusters=2)

model = model.fit(scale(data))
print(model.labels_)
plt.figure(figsize=(8, 6))
plt.scatter(x=data[:,0], y=data[:,1], c=model.labels_.astype(float))
plt.show()