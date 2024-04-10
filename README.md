# Spotify-Recommendation-Engine

**Project Title**: Spotify Recommendation System

**Skills take away From This Project**: Data Scraaping, Pre-Processing, EDA, Unsupervised Model Building, Streamlit

**Domain**: Music

Spotify is a digital music, podcast, and video service that gives you access to millions of songs and other content from creators all over the world. Spotify is available across a range of devices, including computers, phones, tablets, speakers, TVs, and cars, and you can easily transition from one to another with Spotify Connect.

### Need for this Project:

A music recommender solution is essentially a solution that allows music streaming platforms to offer their users relevant music recommendations in real-time. It provides personalization and thus boosts user engagement. The recommender system is helpful to both service providers and users. It saves time for the user in finding and selecting a perfect song and at the same time, it also helps service providers retain customers for a longer time on their platform. 

### Concepts covered in this Project:

- Data Pre-Processing
- Exploratory Data Analysis (On Different attributes)
- Feature Segregation and Standardization ( PCE, tSNE)
- Unsupervised Machine Learning Algorithm(K-Means Cluster)
- Cosine Similarity, Euclidean Distances


### K-Means Cluster
K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into a set of K distinct, non-overlapping clusters. It's widely used in various fields such as data mining, pattern recognition, and image analysis.

#### Advantages:
- Simple and easy to implement.
- Efficient in terms of computational cost, especially for large datasets.
- Scales well to high-dimensional data.
- Produces tight clusters when the clusters are well-separated.

#### Limitations:
- Requires the number of clusters (K) to be specified in advance.
- Sensitive to the initial placement of centroids, which can lead to suboptimal solutions.
- Prone to getting stuck in local optima, depending on the initialization.
- Assumes clusters are spherical and of similar size, which may not always hold true for real-world data.


### Cosine Similarity:
Cosine similarity is a metric used to measure the similarity between two vectors in a high-dimensional space. It calculates the cosine of the angle between the two vectors, indicating their directional similarity regardless of their magnitudes.
Cosine similarity values range from -1 to 1. A value of 1 indicates that the vectors are perfectly similar.

### Euclidean Distances:
Euclidean distance is a measure of the straight-line distance between two points in Euclidean space. It's the "ordinary" straight-line distance between two points in a plane or three-dimensional space.

#### Comparison:
- Cosine Similarity: Measures similarity in direction, disregarding magnitude. Useful for text data, recommendation systems, and any scenario where the magnitude of the vectors is not as important as their orientation.

- Euclidean Distance: Measures the geometric distance between two points in space. It's more intuitive and widely applicable, particularly in clustering, dimensionality reduction, and distance-based algorithms.
