import pandas as pds
from database import dao
from sklearn.metrics.pairwise import cosine_similarity

NUMBER_OF_MOVIES_TO_RETURN = 10 # Number of movies we want to recommend in our top N recommender

# generates a list of top N recommendations for a particular user Id. Can choose to exclude movies the user has already watched or not, based on if they have alraedy rated that movie
# can provide user-item matrix and user-user similarity matrix as parameters if already computed to speed up computation
def generate_recommendations(userId, excludeAlreadyWatchedMovies, user_item_matrix, user_similarity):
    if user_item_matrix is None:
        user_item_matrix = build_user_item_matrix()
        if user_item_matrix.empty:
            return None
    
    if user_similarity is None:
        user_similarity = build_user_to_user_similarity_matrix(user_item_matrix)
        if user_similarity.empty:
            return None

    top_similar_users = find_most_similar_users_for_specific_user(userId, user_similarity)
    if not top_similar_users:
        return None

    if excludeAlreadyWatchedMovies:
        watchedMovies = get_user_already_watched_movies(userId, user_item_matrix)

    movie_to_score_map = {}
    for user, user_similarity_score in top_similar_users:
        for (movieId, rating) in user_item_matrix.loc[user].items():
            if excludeAlreadyWatchedMovies and movieId in watchedMovies:
                continue
            if rating > 0:
                movieScore = (rating * user_similarity_score)
                movie_to_score_map[movieId] = movie_to_score_map.get(movieId, 0) + movieScore
    
    # Converts the map to a tuple of (movieId, score), sorts based on the score descending (top scores first) and returns top N movies
    top_movies = dict(sorted(movie_to_score_map.items(), key=lambda item: item[1], reverse=True)[:NUMBER_OF_MOVIES_TO_RETURN])

    return list(top_movies.keys())

def get_user_already_watched_movies(userId, user_item_matrix):
    if userId not in user_item_matrix.index:
        return set()
    
    watched_movies = user_item_matrix.loc[userId]
    # movie is considered 'watched' if it has a rating (i.e. rating is > 0)
    watched_movies = watched_movies[watched_movies > 0].index  # filter movies that have a rating of 0 from the user-item matrix to get the watched movies
    
    return set(watched_movies)

def build_user_item_matrix():
    ratings = dao.get_ratings_data()
    if not ratings:
        return None

    df = pds.DataFrame(ratings)

    user_item_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')
    user_item_matrix.fillna(0, inplace=True) # fill with 0 if no rating for this movie from this user
    return user_item_matrix

def build_user_to_user_similarity_matrix(user_item_matrix):
    user_similarity = pds.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_similarity

def find_most_similar_users_for_specific_user(userId, user_similarity, n=5):
    if userId not in user_similarity:
        return []
    similar_users = user_similarity[userId].sort_values(ascending=False).iloc[1:n+1] #skip the first element as the most similar user to a user will be the user themselves
    return list(similar_users.items())
