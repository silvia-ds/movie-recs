# import streamlit as st

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reads info from dataset
movies = pd.read_csv("ml-32m/movies.csv")
ratings = pd.read_csv("ml-32m/ratings.csv", usecols=["userId", "movieId", "rating"])


# Removes special chars from titles in dataset
def cleanTitle(title):
    new_title = ""
    for char in title:
        if char==' ':
            new_title += char
        if char.isalnum():
            new_title += char
    return new_title
    
# Creates new column with cleaned titles
movies["clean title"] = movies["title"].apply(cleanTitle)
# print(movies)

# Creates TFIDF matrix
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["clean title"])

# Function that returns 5 most similar movie results
def search(title):
    title = cleanTitle(title)
    queryVec = vectorizer.transform([title])
    similarity = cosine_similarity(queryVec, tfidf).flatten()
    indices = np.argpartition(similarity, -1)[-1:]
    results = movies.iloc[indices][::-1]
    #indices = np.argpartition(similarity, -5)[-5:]
    #results = movies.iloc[indices][::-1]
    return results

# Function that takes in movie id & returns 10 similar movies
def findSimilarMovies(movieId):
    # Function variables:
    #   1. Array of users that enjoyed the inputted movie

    similarUsers = ratings[(ratings["movieId"] == movieId) & (ratings["rating"] >= 4)]["userId"].unique()
    
    #   2. Array of other movies they enjoyed
    similarURecs = ratings[(ratings["userId"].isin(similarUsers)) & (ratings["rating"] >= 4)]["movieId"]
    similarURecs = similarURecs.value_counts() / len(similarUsers)
    similarURecs = similarURecs[similarURecs > 0.1]

    #   3. All users ratings on movies recommended by one movies' niche audience
    allUsers = ratings[(ratings["movieId"].isin(similarURecs.index)) & (ratings["rating"] >= 4)]
    allURecs = allUsers["movieId"].value_counts() / len(allUsers["userId"].unique())

    # Creates a recommendation % based on all user recs vs. users similar to you

    recPercent = pd.concat([similarURecs, allURecs], axis=1)
    recPercent.columns = ["similar", "all"]
    recPercent["score"] = (recPercent["similar"] / recPercent["all"]).round(1)
    recPercent = recPercent.sort_values("score", ascending=False)

    # Returns top 10 movies
    return recPercent.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# testing :
# print("Enter a movie title:\n")
# movieId = 
# command-line ui: takes movie from user:
print("Enter a movie title:")
movieTitle = search(input())

print(findSimilarMovies(66097))

# print("\nMovies like " , movieTitle, ":\n")

# https://www.youtube.com/watch?v=eyEabQRBMQA&t=121s&ab_channel=Dataquest
# streamlit UI: https://www.youtube.com/watch?v=D0D4Pa22iG0&ab_channel=pixegami
