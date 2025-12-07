import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
import re
import string
import nltk
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
import json
# TF-IDF and Cosine Similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Sentence Transformer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
import pickle
from scipy import sparse

# Preprocessing function
def preprocess(inputs, Cast = False, Crew = False, Tagline = False, Overview = False):
    temp = []
    
    if Cast == True:
        inputs = json.loads(inputs)
        for dicts in inputs:
            if dicts["name"].replace(" ", "") not in temp:
                temp.append(dicts["name"].replace(" ", ""))
            if len(temp) >= 5:
                break
        # lowercase
        text = " ".join(temp).lower()
        
    elif Crew == True:
        inputs = json.loads(inputs)
        for dicts in inputs:
            if dicts["job"] == "Director" or dicts["job"] == "Writer" or dicts["job"] == "Producer":
                if dicts["name"].replace(" ", "") not in temp:
                    temp.append(dicts["name"].replace(" ", ""))
        # lowercase
        text = " ".join(temp).lower()
        
    elif Tagline == True or Overview == True:
        text = inputs.lower()
        
    else:
        inputs = json.loads(inputs)
        for dicts in inputs:
            temp.append(dicts["name"].replace(" ", ""))
        # lowercase
        text = " ".join(temp).lower()

    
    # remove special character + digits
    text = re.sub(r'\d+','', text)
    text = re.sub(r'[^\w\s]','',text)
    #tokenize
    tokens = nltk.word_tokenize(text)
    
    # remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    #Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    clean_text = " ".join(lemmatized_tokens)
    
    return clean_text

# recommender function using TF-IDF and Cosine Similarity
def recommender_tfidf (movie_list, n=10):
    sims = []
    
    for movie in movie_list:
        idx = indices[movie]
        sims.append(cosine_sim[idx])
        
    # average similarity across all input movies
    mean_sim = np.mean(sims, axis=0)

    # sort similarity
    similar_indices = mean_sim.argsort()[::-1][1:n+1]

    return tmdb['original_title'].iloc[similar_indices]

def get_index_from_title(title):
    return tmdb[tmdb['original_title'] == title].index[0]

def get_user_profile(movie_list):
    idxs = [get_index_from_title(title) for title in movie_list]
    selected_embeddings = movie_embeddings[idxs]
    return torch.mean(selected_embeddings, dim=0)

def recommender_sbm(movie_list, top_n=10):
    user_vec = get_user_profile(movie_list)
    scores = cos_sim(user_vec, movie_embeddings)[0]
    top_results = torch.topk(scores, k=top_n + len(movie_list))
    movie_idxs = top_results.indices

    # remove input movies from recommendations
    final = []
    for idx in movie_idxs:
        # print(idx)
        if tmdb.iloc[int(idx)]['original_title'] not in movie_list:
            final.append((tmdb.iloc[int(idx)]['original_title'], float(scores[int(idx)])))
        if len(final) >= top_n:
            break
    return final

# Download latest version of movie dataset from Kaggle
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
tmdb_credits = pd.read_csv(os.path.join(path, "tmdb_5000_credits.csv"))
tmdb_movies = pd.read_csv(os.path.join(path, "tmdb_5000_movies.csv"))

# Rename key columns for merging
tmdb_movies = tmdb_movies.rename(columns = {"id":"movie_id"})

# Merge datasets
tmdb = tmdb_movies.merge(tmdb_credits, how="left", on="movie_id")

#impute missing values
tmdb.loc[tmdb["original_title"] == "Chiamatemi Francesco - Il Papa della gente", "runtime"] = 113.0
tmdb.loc[tmdb["original_title"] == "To Be Frank, Sinatra at 100", "runtime"] = 81.0
tmdb.loc[tmdb["original_title"] == "America Is Still the Place", "release_date"] = 2017

#drop nan
tmdb = tmdb.dropna(subset = ['overview'])

# Preprocess relevant columns
for col in ["genres", "keywords", "overview", "production_companies", "cast", "crew"]:
    if col == 'overview' or col == 'tagline':
        print()
        tmdb[col] = tmdb[col].apply(preprocess, Overview = True)
    elif col == 'cast':
        tmdb[col] = tmdb[col].apply(preprocess, Cast = True)
    elif col == 'crew':
        tmdb[col] = tmdb[col].apply(preprocess, Cast = True)
    else:
        tmdb[col] = tmdb[col].apply(preprocess)

# Create 'soup' column by combining relevant text features
tmdb['soup'] = (
    tmdb['overview'] + ' ' +
    tmdb['keywords'] + ' ' +
    tmdb['genres'] + ' ' +
    tmdb['production_companies'] + ' ' +
    tmdb['cast'] + ' ' +
    tmdb['crew']
#     str(tmdb['vote_count'] + ' ' +
#     str(tmdb['vote_average']) + ' ' +
#     str(tmdb['release_date']) + ' ' +
#     str(tmdb['runtime']) + ' ' +
#     str(tmdb['original_language'])
)

# TF-IDF Method
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=50000
)

tfidf_matrix = tfidf.fit_transform(tmdb['soup'])
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# indices = pd.Series(tmdb.index, index=tmdb['original_title'])

# Save vectorizer
with open(r"C:\Users\kenny\OneDrive - purdue.edu\Documents\Kenny's File\Transportation Literature\Data Science Tutorial\Movie Recommendation\tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save TF-IDF matrix (sparse)
sparse.save_npz(r"C:\Users\kenny\OneDrive - purdue.edu\Documents\Kenny's File\Transportation Literature\Data Science Tutorial\Movie Recommendation\tfidf_matrix.npz", tfidf_matrix)

# SBERT Method
model = SentenceTransformer('all-mpnet-base-v2')

movie_embeddings = model.encode(
    tmdb['soup'].tolist(),
    show_progress_bar=True,
    convert_to_tensor=True
)
# ------------- SBERT: compute embeddings and save -------------
model = SentenceTransformer('all-mpnet-base-v2')
# Save embeddings
np.save(r"C:\Users\kenny\OneDrive - purdue.edu\Documents\Kenny's File\Transportation Literature\Data Science Tutorial\Movie Recommendation\sbert.npy", movie_embeddings)

# ------------- Save processed dataframe -------------
tmdb.to_pickle(r"C:\Users\kenny\OneDrive - purdue.edu\Documents\Kenny's File\Transportation Literature\Data Science Tutorial\Movie Recommendation\tmdb_processed.pkl")

print("Saved: tfidf_vectorizer.pkl, tfidf_matrix.npz, sbert_embeddings.npy, tmdb_processed.pkl")