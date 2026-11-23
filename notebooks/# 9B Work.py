# 9B Work

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

observations = pd.read_csv("C:\\Users\\abadj\\Documents\\GitHub\\CSC1171\\Week 1\\week9-DataIntro\\data\\week9_sleep_stress.csv")

features = ["sleep_hours", "stress_score"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(observations[features])

scaled_data = pd.DataFrame(X_scaled, columns=features)

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=20,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)

centres_scaled = kmeans.cluster_centers_
centres_original = scaler.inverse_transform(centres_scaled)

print(f"Inertia: {kmeans.inertia_:.2f}")
print("Number of observations in each cluster:")
print(pd.Series(cluster_labels).value_counts().sort_index())

clustered_data = observations.copy()
clustered_data["cluster"] = cluster_labels

plt.figure(figsize=(7, 5))
plt.scatter(
    clustered_data["sleep_hours"],
    clustered_data["stress_score"],
    c=clustered_data["cluster"],
    alpha=0.65
)
plt.scatter(
    centres_original[:, 0],
    centres_original[:, 1],
    marker="o",
    s=220,
    edgecolor="black",
    label="Centroids"
)
plt.xlabel("Average daily sleep (hours)")
plt.ylabel("Self-reported stress (1–10)")
plt.title("K-Means result (K = 3)")
plt.legend()
plt.grid(alpha=0.2)
plt.show()

cluster_summary = (
    clustered_data
    .groupby("cluster")[features]
    .agg(["count", "mean", "std"])
    .round(2)
)

cluster_summary

from sklearn.metrics import silhouette_score

k_values = range(1, 9)
inertias = []

for k in k_values:
    candidate = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=42
    )
    candidate.fit(X_scaled)
    inertias.append(candidate.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(k_values), inertias, marker="o")
plt.xticks(list(k_values))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow method")
plt.grid(alpha=0.25)
plt.show()

silhouette_scores = []
silhouette_k_values = range(2, 9)

for k in silhouette_k_values:
    candidate = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        random_state=42
    )
    labels = candidate.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(6, 4))
plt.plot(list(silhouette_k_values), silhouette_scores, marker="o")
plt.xticks(list(silhouette_k_values))
plt.xlabel("Number of clusters (K)")
plt.ylabel("Mean silhouette score")
plt.title("Silhouette score by K")
plt.grid(alpha=0.25)
plt.show()

pd.DataFrame({
    "K": list(silhouette_k_values),
    "mean_silhouette": silhouette_scores
}).round(3)