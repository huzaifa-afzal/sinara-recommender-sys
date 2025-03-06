from recommendations import movie_recommendations
from database import dao
from evaluator import evaluator

# Select the user ID for the user for which we want to generate recommendations for.
# Change to see the different results, for example user with user ID '1000002' enjoys Marvel movies
USER_ID = 1

print("Top 10 Recommendations for User with ID", USER_ID)
movieRecommendations = movie_recommendations.generate_recommendations(USER_ID, excludeAlreadyWatchedMovies=True, user_similarity=None, user_item_matrix=None)

rank = 1
for movieId in movieRecommendations:
    movieTitle = dao.get_movie_title_from_id(movieId)
    if movieTitle:
        print(rank, "- ", movieTitle)
        rank +=1

print("")
# Generates different evaluation metrics for the recommender system that helps us analyse the performance of the recommender system
evaluator.evaluate()