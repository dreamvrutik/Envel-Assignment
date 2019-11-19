import pandas as pd
import numpy as np
import itertools
import random,math
from sklearn import decomposition
import matplotlib.pyplot as plt

def dist(a,b):
    d=0
    for i in range(len(a)):
        if not(math.isnan(a[i]) or math.isnan(b[i])):
            d+=abs(int(a[i])-int(b[i]))
    return d
dataset=pd.read_csv("Q_test_data.csv")
dataset.fillna(dataset.mean())
rows=dataset.shape[0]
data =[[] for i in range(rows)]
exclude=["availableDate","createdDate","voidedDate","settledDate","transactionId","masterId","description","transactionCount"]
for key,value in dataset.iteritems():
    if key not in exclude:
        i=0
        for j in value:
            if math.isnan(j):
                data[i].append(0)
            else:
                data[i].append(j)
            i+=1
colors = itertools.cycle(["r", "b", "g"])
k=5
centroids=[]
for i in range(k):
    h=random.randint(0,rows-1)
    centroids.append(data[h])
cluster=[-1 for i in range(rows)]
for i in range(100):
    ct=0
    for j in data:
        mind=1e50
        cl=-1
        g=0
        for h in centroids:
            if dist(h,j)<mind:
                mind=dist(h,j)
                cl=g
            g+=1
        cluster[ct]=cl
        for jj in range(len(centroids[cl])):
            centroids[cl][jj]=(centroids[cl][jj]+j[jj])/2
        ct+=1

X_pca = decomposition.PCA(n_components=2).fit_transform(data)
points_x=[[] for i in range(k)]
points_y=[[] for i in range(k)]
for i in range(len(X_pca)):
    points_x[cluster[i]].append(X_pca[i][0])
    points_y[cluster[i]].append(X_pca[i][1])


for y in range(k):
    plt.scatter(points_x[y], points_y[y], color=next(colors))
plt.xlabel('PCA 1')
# frequency label
plt.ylabel('PCA 2')
# plot title
plt.title('K='+str(k))
plt.savefig("K="+str(k)+".png")

# function to show the plot
plt.show()
