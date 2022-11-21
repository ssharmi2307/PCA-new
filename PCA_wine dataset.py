# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:57:56 2022

@author: Gopinath
"""

import warnings
warnings.filterwarnings('ignore')
#Loading dataset
import pandas as pd
import numpy as np
df=pd.read_csv("wine.csv")
list(df)
df.shape
df.corr()
df.info()
df.values
df
#Normalisation
from sklearn.preprocessing import scale
df_norm=scale(df.iloc[:,1:])
df_norm
# principal compononents
from sklearn.decomposition import PCA
pca = PCA(n_components=13)
PC = pca.fit_transform(df_norm)
PC
# PCA Components matrix or covariance Matrix
pca.components_

# The amount of variance that each PCA has
var=pca.explained_variance_ratio_
var

#Cummulative variance of each PCA
var1=np.cumsum(np.round(var,4)*100)
var1

#variance plot for PCA components obtained
import matplotlib.pyplot as plt
plt.plot(var1,color='red')
plt.grid()
plt.xlabel("NO of PC")
#Plt between PC1 and PC2
X=PC[:,1]
Y=PC[:,2]
plt.scatter(X,Y)
import seaborn as sns
df = pd.DataFrame({"PC1":PC[:,0],"PC2":PC[:,1]})
sns.scatterplot(x='PC1',y="PC2", data=df, color="c");

#plt between PC1,PC2,PC3
final_df=pd.concat([pd.DataFrame(PC[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
final_df
# Visualization
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=final_df)

#Using CLUSTERING:
    #1.HEIRARCHICAL
import pandas as pd
df=pd.read_csv("wine.csv")
df
#Normalisation
from sklearn.preprocessing import scale
df_norm=scale(df.iloc[:,1:])
df_norm
#dendograms using different linkage and affinity
#affinity and linkage criteria are versatile tools to segment a dataset
#creating dendogram using ward or mean linkage
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
#create clusters
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
#saving cluster as chart
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
A=df1['ID'].value_counts()
#creating dendogram using average linkage
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='average')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
B=df1['ID'].value_counts()
#creating dendogram using complete linkage
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
C=df1['ID'].value_counts()
#creating dendogram using single linkage
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
D=df1['ID'].value_counts()

out=pd.concat([A,B,C,D], axis=1)
out.columns = ['ward', 'average', 'complete','single']
print(out)
#The table below shows the number of individuals in each cluster.
#The clustering approach using all linkages had 3 clusters.
#Model A was nicely balanced compared with Model C
#model B and model D was skewed and had the majority of donors in a single cluster.

#using different metrics or affinity
#l1
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=4,affinity='l1',linkage='single')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
a=df1['ID'].value_counts()
#l2
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=4,affinity='l2',linkage='complete')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
b=df1['ID'].value_counts()
#manhattan
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=4,affinity='manhattan',linkage='average')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
c=df1['ID'].value_counts()
#cosine
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=4,affinity='cosine',linkage='single')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
d=df1['ID'].value_counts()
#using eucledain
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(df,method='complete'))
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
clusters=pd.DataFrame(hc.fit_predict(df_norm),columns=['clustetrs'])
clusters['clustetrs'].value_counts()
df1=df.copy()
df1['ID']=hc.labels_
df1
e=df1['ID'].value_counts()

out=pd.concat([a,b,c,d,e], axis=1)
out.columns = ['l1', 'l2', 'manhattan', 'cosine','eucledian']
print(out)
#2.KMeans clustering
import pandas as pd
df=pd.read_csv("wine.csv")
df
#Normalisation
from sklearn.preprocessing import scale
df_norm=scale(df.iloc[:,1:])
df_norm
#To find optimum method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    dist=kmeans.inertia_
    wcss.append(dist)
#data visualization
import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("NO OF CLUSTERS")
plt.ylabel("Within cluster sum of square")
plt.grid()
plt.show()
#cluster algorithm
clusters_new=KMeans(n_clusters=3,random_state=0)
clusters_new.fit(df_norm)
clusters_new.labels_
df1=df.copy()
df1['ID']=clusters_new.labels_
df1
df1['ID'].value_counts()

