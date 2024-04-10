import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("Set2")
import streamlit as st
from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

st.set_option('deprecation.showPyplotGlobalUse', False)

dat_recommendation = pd.read_csv('dat_for_recommender.csv')

song_features_normalized = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
song_features_not_normalized = ['duration_ms', 'key', 'loudness', 'mode', 'tempo']

features = song_features_normalized + song_features_not_normalized + ['decade', 'popularity']

def get_feature_vector(song_name, year):
    dat_song = dat_recommendation.query('name == @song_name & year == @year')
    song_repeated = 0
    if dat_song.shape[0] == 0:
        raise Exception('The song does not exist in the dataset! \n Use search function first if you are not sure')
    if dat_song.shape[0] > 1:
        song_repeated = dat_song.shape[0]
        print(f'Warning: Multiple ({song_repeated}) songs with the same name and artist, the first one is selected!')
        dat_song = dat_song.head(1)
    feature_vector = dat_song[features].values
    return feature_vector, song_repeated

def get_similar_songs(song_name, year, top_n=10, plot_type='wordcloud'):
    feature_vector, song_repeated = get_feature_vector(song_name, year)

    # calculate the cosine similarity
    similarities = cosine_similarity(dat_recommendation[features].values, feature_vector).flatten()

    # get the index of the top_n similar songs not including itself
    if song_repeated == 0:
        related_song_indices = similarities.argsort()[-(top_n+1):][::-1][1:]
    else:
        related_song_indices = similarities.argsort()[-(top_n+1+song_repeated):][::-1][1+song_repeated:]
        
    similar_songs = dat_recommendation.iloc[related_song_indices][['name', 'artists', 'year']]
    
    if plot_type == 'wordcloud':
        # make a word cloud of the most similar songs and year, use the simalirity score as the size of the words
        similar_songs['name+year'] = similar_songs['name'] + ' (' + similar_songs['year'].astype(str) + ')'
        # create a dictionary of song and their similarity
        song_similarity = dict(zip(similar_songs['name+year'], similarities[related_song_indices]))
        # sort the dictionary by value
        song_similarity = sorted(song_similarity.items(), key=lambda x: x[1], reverse=True)
        # # create a mask for the word cloud
        # mask = np.array(Image.open("spotify-logo.png"))
        # create a word cloud
        wordcloud = WordCloud(width=1600, height=800, max_words=50, 
                            background_color='white', colormap='viridis').generate_from_frequencies(dict(song_similarity))
        plt.figure(figsize=(12,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{top_n} most similar songs to: {song_name} ({year})', fontsize=20)
        plt.tight_layout(pad=0)
        plt.show()
    
    elif plot_type == 'bar':
        # plot the text of the most similar songs and year in order, like a stacked bar chart
        similar_songs['name+year'] = similar_songs['name'] + ' (' + similar_songs['year'].astype(str) + ')'
        # create a dictionary of song and their similarity
        song_similarity = dict(zip(similar_songs['name+year'], similarities[related_song_indices]))
        # sort the dictionary by value
        song_similarity = sorted(song_similarity.items(), key=lambda x: x[1], reverse=True)
        # plot the text of the most similar songs and year in order, like a stacked bar chart
        plt.figure(figsize=(12,10))
        plt.barh(range(len(song_similarity)), [val[1] for val in song_similarity], 
                 align='center', color=sns.color_palette('pastel', len(song_similarity)))
        plt.yticks(range(len(song_similarity)), [val[0] for val in song_similarity])
        plt.gca().invert_yaxis()
        plt.title(f'{top_n} most similar songs to: {song_name} ({year})', fontsize=20)
        min_similarity = min(similarities[related_song_indices])
        max_similarity = max(similarities[related_song_indices])

        for i, v in enumerate([val[0] for val in song_similarity]):
            plt.text(min_similarity*0.955, i, v, color='black', fontsize=12)
        plt.xlim(min_similarity*0.95, max_similarity)
        plt.box(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        
        plt.show()

def search_song(song_name, dat_recommendation):
    dat_song = dat_recommendation.query('name == @song_name')
    if dat_song.shape[0] == 0:
        found_flag = False
        found_song = None
    else:
        found_flag = True
        found_song = f"Great! This song is in the dataset: {dat_song[['name', 'artists', 'release_date']].to_numpy()}"
    return found_flag, found_song

st.set_page_config(page_title="Spotify Recommendation System",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This dashboard app is brought to you by S. Pavitran Kartick.
                                        Data has been gathered from Spotify"""})

st.markdown("<h1 style='text-align: center; color: #000000;'>GUVI Capstone Project 7</h1>",unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #000000;'>Spotify Recommendation System</h1>",unsafe_allow_html=True)


st.markdown('''
<h3><span style="color:black">Welcome to your personalized Music Recommendation System!</span></h4>
\n
Tired of searching for songs that are similar to the ones you like? Do not worry!
\n
Using this simple web application, we can obtain recommendations based on the songs that you like.
''', unsafe_allow_html=True)

st.write('- This project is created with the primary concept of unsupervised Machine Learning, and the method used here is K-means clustering.')

# add selectbox for selecting the features
st.sidebar.markdown("### Select Features")
features = st.sidebar.multiselect('Select the features you care about', features, default=features)
# add a slider for selecting the number of recommendations
st.sidebar.markdown("### Number of Recommendations")
num_recommendations = st.sidebar.slider('Select the number of recommendations', 10, 50, 10)

# add a search box for searching the song by giving capital letters and year
st.markdown("#### Searching for the song:")
song_name = st.text_input('Enter the name of the song')
if song_name != '':
    song_name = song_name.upper()
year = st.text_input('Enter the year of the song (e.g. 2019). \
                        \nIf you are not sure if the song is in the database or not sure about the year, \
                        please leave the year blank and click the button below to search for the song.')
if year != '':
    year = int(year)

# exmaples of song name and year:
# song_name = 'YOUR HAND IN MINE'
# year = 2003

# add a button for searching the song if the user does not know the year
if st.button('Search for my song'):
    found_flag, found_song = search_song(song_name,dat_recommendation)
    if found_flag:
        st.markdown("Perfect, this song is in the dataset:")
        st.markdown(found_song)
    else:
        st.markdown("Sorry, this song is not in the dataset. Please try another song!")

# add a button for getting recommendations
if st.button('Get Recommendations'):
    if song_name == '':
        st.markdown("Please enter the name of the song!")
    elif year == '':
        st.markdown("Please enter the year of the song!")
    else:
        
        # show the most similar songs in wordcloud
        fig_cloud = get_similar_songs(song_name, year, top_n = num_recommendations, plot_type='wordcloud')
        st.markdown(f"### Great! Here are your recommendation for \
                    \n#### {song_name} ({year})!")
        st.pyplot(fig_cloud)

        # show the most similar songs in bar chart
        fig_bar = get_similar_songs(song_name, year, top_n=num_recommendations, plot_type='bar')
        st.markdown("### Get a closer look at the top 10 recommendations for you!")
        st.pyplot(fig_bar)
