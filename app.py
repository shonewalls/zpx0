from flask import Flask, request, jsonify, render_template
from azure.cosmos import CosmosClient
import json
import time
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import redis

app = Flask(__name__)

# Student Information
student_info = "76405    Pengxiang Zhou"

# Azure Cosmos DB configuration
cosmos_endpoint = "https://tutorial-uta-cse6332.documents.azure.com:443/"
cosmos_key = "fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw=="
cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
database_name = "tutorial"
reviews_container_name = "reviews"
us_cities_container_name = "us_cities"

# Redis configuration
redis_host = "your_redis_host"
redis_port = 6379
redis_db = 0
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

# Function to calculate Eular distance
def calculate_distance(lat1, lng1, lat2, lng2):
    return math.sqrt((lat1 - lat2)**2 + (lng1 - lng2)**2)

# Function to perform KNN clustering on reviews
def knn_reviews(classes, k, words):
    # Check if the result is cached in Redis
    cache_key = f"knn_reviews:{classes}:{k}:{words}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return json.loads(cached_result)

    # Query all cities and reviews
    query_cities = f"SELECT * FROM c"
    all_cities = list(cosmos_client.get_database_client(database_name).get_container_client(us_cities_container_name).query_items(query_cities, enable_cross_partition_query=True))

    query_reviews = f"SELECT * FROM c"
    all_reviews = list(cosmos_client.get_database_client(database_name).get_container_client(reviews_container_name).query_items(query_reviews, enable_cross_partition_query=True))

    # Create a matrix of city coordinates
    city_coordinates = np.array([[city['lat'], city['lng']] for city in all_cities])

    # Create a matrix of review texts
    review_texts = [review['review'] for review in all_reviews]
    vectorizer = CountVectorizer(stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(review_texts)

    # Calculate Euclidean distances between reviews
    review_distances = euclidean_distances(X, X)

    # Perform KNN clustering
    labels = np.zeros(len(all_reviews))
    for i, review_distance in enumerate(review_distances):
        # Find the indices of the k nearest neighbors
        knn_indices = np.argsort(review_distance)[1:k+1]
        # Assign the label of the most frequent class among the neighbors
        labels[i] = np.argmax(np.bincount([int(label) for label in labels[knn_indices]]))

    # Placeholder for the actual implementation of clustering and other details

    # Prepare the response
    response_data = {
        "clusters": [],
        "response_time": 0,
        "from_cache": False
    }

    # Cache the result in Redis
    redis_client.set(cache_key, json.dumps(response_data), ex=3600)  # Cache for 1 hour

    return response_data

# API endpoint to render the webpage
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to get KNN reviews
@app.route('/stat/knn_reviews', methods=['GET'])
def get_knn_reviews():
    start_time = time.time()

    # Get request parameters
    classes = int(request.args.get('classes', 6))
    k = int(request.args.get('k', 3))
    words = int(request.args.get('words', 100))

    # Perform KNN clustering on reviews
    response_data = knn_reviews(classes, k, words)
    response_data["response_time"] = int((time.time() - start_time) * 1000)

    # Add information about cache involvement to the response
    response_data["from_cache"] = bool(redis_client.get(f"knn_reviews:{classes}:{k}:{words}"))

    return jsonify(response_data)

# API endpoint to flush the cache
@app.route('/flush_cache', methods=['GET'])
def flush_cache():
    redis_client.flushdb()
    return jsonify({"message": "Cache flushed successfully"})

if __name__ == '__main__':
    app.run(debug=True)

