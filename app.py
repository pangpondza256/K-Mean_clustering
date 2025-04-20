# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:08:20 2025

@author: pangp
"""
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
st.title("K-Means Clustering Visualizer by Thanakorn Ritsub")

st.set_page_config(page_title="K-Means Clustering", layout='centered')

X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

y_kmens = loaded_model.predict(X)

fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
