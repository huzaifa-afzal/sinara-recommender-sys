from surprise.model_selection import LeaveOneOut
import pandas as pds
from surprise import Dataset
from surprise import Reader
import math
from database import dao
from recommendations import movie_recommendations

TOTAL_NUMBER_OF_MOVIES = 9737 # the movies in the dataset are fixed so we can use a constant to define how many movies there are in total

# Evaluates our recommender system, generating metrics for our results based on averages across all users.
# For more of an understanding on these metrics, these two articles are recommended:
# https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems#diversity
# https://medium.com/nerd-for-tech/evaluating-recommender-systems-590a7b87afa5
def evaluate():
    print("Evaluating Metrics for recommender system")
    # for hit rate, we don't want to exclude already watched movies otherwise our hit rate would be 0 because then the recommended items list will never have one of the items the user has rated
    recommendations_for_all_users_for_hit_rate = generate_recommendations_for_all_users(excludeAlreadyWatchedMovies=False)
    LOOCVTestData = generateLOOCVTestData()
    print("Hit rate: ", calculateHitRate(recommendations_for_all_users_for_hit_rate, LOOCVTestData))
    print("Average Reciprocal Hit rate: ", calculateAverageReciprocalHitRate(recommendations_for_all_users_for_hit_rate, LOOCVTestData))

    # for all other metrics, we can exclude already watched movies to make the recommendations resemble exactly what is shown to the user (where we exclude user's already watched movies)
    recommendations_for_all_users = generate_recommendations_for_all_users(excludeAlreadyWatchedMovies=True)
    print("Coverage: ", (calculateCoverage(recommendations_for_all_users)*100), "%") # convert to a percentage
    print("Diversity: ", calculateAverageGiniSimpsonDiversityAcrossAllUsers(recommendations_for_all_users, dao.build_movie_genre_map()))
    print("Novelty: ",  calculateAverageNoveltyAcrossAllUsers(recommendations_for_all_users))

def generate_recommendations_for_all_users(excludeAlreadyWatchedMovies):    
    recommendation_map = {}  # This will store the userId to list of movieIds for user-movie recommendations

    user_item_matrix = movie_recommendations.build_user_item_matrix()
    user_similarity = movie_recommendations.build_user_to_user_similarity_matrix(user_item_matrix)
    
    for userId in user_item_matrix.index:
        userRecs = movie_recommendations.generate_recommendations(userId, excludeAlreadyWatchedMovies, user_item_matrix, user_similarity)
        if userRecs:
            recommendation_map[userId] = userRecs
    
    return recommendation_map


def generateLOOCVTestData():
    ratings = dao.get_ratings_data()
    df = pds.DataFrame(ratings)
    reader = Reader(rating_scale=(0.5, 5))

    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    LOOCV = LeaveOneOut(n_splits=1, random_state=1) 
    for train, test in LOOCV.split(data):
        LOOCVTest = test
    return LOOCVTest


def calculateHitRate(recommendationsForAllUsersMap, leftOutPredictions):
    hits = 0
    total = 0

    # for each left-out rating, we want to check if it is in the predicted top N list for the user, if it is we consider it a hit
    for leftOut in leftOutPredictions:
        try:
            userID = int(leftOut[0])
            leftOutMovieID = int(leftOut[1])
        except ValueError:
            continue
        for movieID in recommendationsForAllUsersMap[userID]:
            if (leftOutMovieID == int(movieID)):
                hits +=1
                break

        total += 1

    return hits/total

def calculateAverageReciprocalHitRate(recommendationsForAllUsersMap, leftOutPredictions):
    reciprocalSum = 0
    total = 0

    for leftOut in leftOutPredictions:
        try:
            userID = int(leftOut[0])
            leftOutMovieID = int(leftOut[1])
        except ValueError:
            continue

        if userID in recommendationsForAllUsersMap:
            try:
                # considers position of the item in the top N recommended list rather than just a binary yes/no for a hit
                rank = recommendationsForAllUsersMap[userID].index(leftOutMovieID) + 1  # rank starts at 1
                reciprocalSum += 1 / rank 
            except ValueError:
                pass 

        total += 1
    
    return reciprocalSum / total

def calculateCoverage(recommendationsForAllUsersMap):
    recommendedMovies = set()
    for userID in recommendationsForAllUsersMap:
        recommendedMovies.update(recommendationsForAllUsersMap[userID]) 
    
    return len(recommendedMovies) / TOTAL_NUMBER_OF_MOVIES

def calculateAverageGiniSimpsonDiversityAcrossAllUsers(recommendationsForAllUsersMap, movieGenreMap):
    total_diversity = 0
    users = 0

    if not movieGenreMap:
        return 0
    
    for userID, recommendedMovies in recommendationsForAllUsersMap.items():

        # for each user, compute the gini simpson diversity of their recommended top N list, theen average out at the end
        gini_simpson = calculateGiniSimpsonDiversity(recommendedMovies, movieGenreMap)
        
        total_diversity += gini_simpson
        users += 1
    
    return total_diversity / users if users > 0 else 0

def calculateGiniSimpsonDiversity(recommendationsList, movieGenreMap):
    genre_counts = {}
    total_genres = 0

    # count occurrences of each genre - diversity will be calculated based on the genre
    for movieId in recommendationsList:
        genres = movieGenreMap.get(movieId, "").split("|") 
        for genre in genres:
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1
            total_genres += 1 
    
    proportions = [count / total_genres for count in genre_counts.values()]
    
    # compute Gini-Simpson index
    gini_simpson = 1 - sum(p**2 for p in proportions)
    return gini_simpson

def calculateAverageNoveltyAcrossAllUsers(recommendationsForAllUsersMap):
    totalNoveltyAcrossAllUsers = 0
     # since this map contains recommendations for every user, the size of this map is simply the total number of users
    totalNumberOfUsers = len(recommendationsForAllUsersMap)
    moviePopularityMap = generateMoviePopularityForAllMovies(recommendationsForAllUsersMap)

    # We want to calculate a novelty score for each user based on the recommendations they got, in the end we find the average novelty across all users
    for userID, recommendedMovies in recommendationsForAllUsersMap.items():
        averageNoveltyForUserRecommendations = calculateAverageNoveltyForAListOfRecommendations(recommendedMovies, totalNumberOfUsers, moviePopularityMap)
        totalNoveltyAcrossAllUsers += averageNoveltyForUserRecommendations
    
    return totalNoveltyAcrossAllUsers / totalNumberOfUsers # calculate and return average novelty across all users

def calculateAverageNoveltyForAListOfRecommendations(recommendationsList, totalNumberOfUsers, moviePopularityMap):
    totalNovelty = 0
    for movieId in recommendationsList:
        popularity = moviePopularityMap[movieId]
        # calculate the novelty of this item using formula N = log2(M/K) where N is the novelty of this item, M is the total number of users, K is the popularity of this item, i.e. how many users this item was recommended to
        novelty = math.log2(totalNumberOfUsers / popularity)
        totalNovelty += novelty
    averageNovelty = totalNovelty / len(recommendationsList) # calculate average novelty for items in this list by dividing total novelty by the total number of recommendations
    return averageNovelty
            
# calculate movie popularity for each movie by seeing how many times it was rated in our data
def generateMoviePopularityForAllMovies(recommendationsForAllUsersMap):
    # Calculate the popularity of each movie, i.e. how many times this movie was recommended across all users, used for calculating novelty
    # Map used to store the popularity of each movie, i.e. {1 = 30, 2 = 40}, depicting movieId 1 was recommened to 30 users, movieId 2 to 40 users, ...
    moviePopularityMap = {}
    for userID, recommendedMovies in recommendationsForAllUsersMap.items():
        for movieId in recommendedMovies:
            moviePopularityMap[movieId] = moviePopularityMap.get(movieId, 0) + 1

    return moviePopularityMap