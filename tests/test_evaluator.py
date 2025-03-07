import unittest
from unittest.mock import patch
from evaluator import evaluator
import pandas as pd
import math

class TestEvaluator(unittest.TestCase):

    @patch("recommendations.movie_recommendations.build_user_item_matrix")
    @patch("recommendations.movie_recommendations.build_user_to_user_similarity_matrix")
    @patch("recommendations.movie_recommendations.generate_recommendations")
    def test_generate_recommendations_for_all_users(self, mock_generate_recommendations, mock_user_to_user_similarity_matrix, mock_build_user_item_matrix):
        # user to user similarity matrix not relevant in this test case as we are mocking generate_recommendations method 
        mock_user_to_user_similarity_matrix.return_value = {}

        # create the mock user-item matrix
        mock_user_item_matrix = pd.DataFrame({
            101: {1: 5.0, 2: 5.0, 3: 2.0},
            102: {1: 0.0, 2: 4.0, 3: 0.0},
            103: {1: 0.0, 2: 0.0, 3: 4.0}  # default value is 0.0 if no rating specified
        }).fillna(0)
        mock_user_item_matrix.index.name = "userId"
        mock_build_user_item_matrix.return_value = mock_user_item_matrix

        mock_generate_recommendations.side_effect = [
            [103, 104],  # Mock user 1's recommendations
            [105],        # Mock user 2's recommendations
            [101]         # Mock user 3's recommendations
        ]

        result = evaluator.generate_recommendations_for_all_users(excludeAlreadyWatchedMovies=True)

        expected_recommendation_map = {
            1: [103, 104],
            2: [105],
            3: [101]
        }
        self.assertEqual(result, expected_recommendation_map)

    @patch("recommendations.movie_recommendations.build_user_item_matrix")
    @patch("recommendations.movie_recommendations.build_user_to_user_similarity_matrix")
    @patch("recommendations.movie_recommendations.generate_recommendations")
    def test_generate_recommendations_for_all_users_no_recs(self, mock_generate_recommendations, mock_build_user_to_user_similarity_matrix, mock_build_user_item_matrix):
        # user to user similarity matrix not relevant in this test case as we are mocking generate_recommendations method 
        mock_build_user_to_user_similarity_matrix.return_value = {}

        # create the mock user-item matrix
        mock_user_item_matrix = pd.DataFrame({
            101: {1: 5.0, 2: 5.0, 3: 2.0},
            102: {1: 0.0, 2: 4.0, 3: 0.0},
            103: {1: 0.0, 2: 0.0, 3: 4.0}  # default value is 0.0 if no rating specified
        }).fillna(0)
        mock_user_item_matrix.index.name = "userId"
        mock_build_user_item_matrix.return_value = mock_user_item_matrix
        mock_generate_recommendations.return_value = None  # return no recommendations for any users
        
        result = evaluator.generate_recommendations_for_all_users(excludeAlreadyWatchedMovies=True)

        self.assertEqual(result, {}) # assert empty result
    
    @patch("database.dao.get_ratings_data")
    def test_generate_LOOCV_test_data(self, mock_get_ratings_data):
        mock_get_ratings_data.return_value = self.getMockRatingsData()

        result = evaluator.generateLOOCVTestData()

        # since the random state variable is fixed in 'generateLOOCVTestData' when creating LOOCV test data, the expected results can be fixed
        expected_LOOCV_test_data = [
            (1, 101, 5.0),
            (2, 102, 4.0),
            (3, 103, 4.0)
        ]

        self.assertEqual(result, expected_LOOCV_test_data)

    def test_calculate_hit_rate_with_partial_hits(self):
        recommendations_for_all_users = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109],
        }
        
        left_out_predictions = [
            (1, 101, 3.0),  # user 1 has movie 101 in recommendation list, so this is a hit
            (2, 104, 5.0),  # user 2 has movie 104 in recommendation list, so this is a hit
            (3, 105, 6.0),  # user 3 does not have movie 105 in recommendation list, so this is not a hit
        ]
        
        result = evaluator.calculateHitRate(recommendations_for_all_users, left_out_predictions)
        
        # Our hit rate is 2/3, we got hits for user 1 and 2, but not for user 3.
        self.assertEqual(result, 2/3)

    def test_calculate_hit_rate_with_no_hits(self):
        recommendations_for_all_users = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109],
        }
        
        left_out_predictions = [
            (1, 110, 3.0),  # user 1 does not have movie 110 in recommendation list, so this is not a hit
            (2, 111, 5.0),  # user 2 does not have movie 111 in recommendation list, so this is not a hit
            (3, 112, 6.0),  # user 3 does not have movie 112 in recommendation list, so this is not a hit
        ]
        
        result = evaluator.calculateHitRate(recommendations_for_all_users, left_out_predictions)
        
        # We got no hits, so our hit rate should be 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_hit_rate_with_empty_left_out_predictions(self):
        recommendations_for_all_users = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109],
        }
        
        left_out_predictions = []
        
        result = evaluator.calculateHitRate(recommendations_for_all_users, left_out_predictions)
        
        # No left_out_predictions data, so we expect our function to return 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_hit_rate_with_empty_recommendations_map(self):
        recommendations_for_all_users = { }
        
        left_out_predictions = [
            (1, 101, 3.0), 
            (2, 104, 5.0),
            (3, 105, 6.0),
        ]
        
        result = evaluator.calculateHitRate(recommendations_for_all_users, left_out_predictions)
        
        # No recommendations for any user, so we expect our function to return 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_hit_rate_with_invalid_data(self):
        recommendations_for_all_users = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109],
        }
        
        left_out_predictions = [
            ("invalid user", 101, 2.0), 
            (2, "invalid movie", 5.0), 
        ]
        
        result = evaluator.calculateHitRate(recommendations_for_all_users, left_out_predictions)
        
        # the invalid data in left_out_predictions should be ignored, meaning hit rate should be 0
        self.assertEqual(result, 0.0)

    def test_calculate_average_reciprocal_hit_rate(self):
        recommendationsForAllUsersMap = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109]
        }

        leftOutPredictions = [
            (1, 101, 5.0),  # 1st rank hit
            (2, 106, 3.5),  # 3rd rank hit
            (2, 200, 5.0),  # no hit
            (3, 107, 4.0),  # 1st rank hit
            (3, 200, 3.0),  # no hit
        ]

        # expected result = (1/1 + 1/3 + 1/1 + 0) / 4 = (3/3 + 1/3 + 3/3) / 5 = 7/3 / 5 = 7/15
        result = evaluator.calculateAverageReciprocalHitRate(recommendationsForAllUsersMap, leftOutPredictions)
        self.assertAlmostEqual(result, 7/15, places=5)

    def test_calculate_average_reciprocal_hit_rate_with_no_hits(self):
        recommendations_for_all_users = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109],
        }
        
        left_out_predictions = [
            (1, 110, 3.0),  # user 1 does not have movie 110 in recommendation list, so this is not a hit
            (2, 111, 5.0),  # user 2 does not have movie 111 in recommendation list, so this is not a hit
            (3, 112, 6.0),  # user 3 does not have movie 112 in recommendation list, so this is not a hit
        ]
        
        result = evaluator.calculateAverageReciprocalHitRate(recommendations_for_all_users, left_out_predictions)
        
        # We got no hits, so our average reciprocal hit rate should be 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_average_reciprocal_hit_rate_with_empty_recommendations_map(self):
        recommendations_for_all_users = { }
        
        left_out_predictions = [
            (1, 101, 3.0), 
            (2, 104, 5.0),
            (3, 105, 6.0),
        ]
        
        result = evaluator.calculateAverageReciprocalHitRate(recommendations_for_all_users, left_out_predictions)
        
        # No recommendations for any user, so we expect our function to return 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_average_reciprocal_hit_rate_with_empty_left_out_predictions(self):
        recommendations_for_all_users = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109],
        }
        
        left_out_predictions = []
        
        result = evaluator.calculateAverageReciprocalHitRate(recommendations_for_all_users, left_out_predictions)
        
        # No left_out_predictions data, so we expect our function to return 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_average_reciprocal_hit_rate_with_invalid_data(self):
        recommendations_for_all_users = {
            1: [101, 102, 103],
            2: [104, 105, 106],
            3: [107, 108, 109],
        }
        
        left_out_predictions = [
            ("invalid user", 101, 2.0), 
            (2, "invalid movie", 5.0), 
        ]
        
        result = evaluator.calculateAverageReciprocalHitRate(recommendations_for_all_users, left_out_predictions)
        
        # the invalid data in left_out_predictions should be ignored, meaning average reciprocal hit rate should be 0
        self.assertEqual(result, 0.0)

    def test_calculate_coverage(self):
        # unique movie ids in this map: 101, 102, 103, 104, 105 - 5 total unique movie Ids
        recommendations_for_all_users_map = {
            1: [101, 102, 103],
            2: [101, 104],
            3: [102, 105],
        } 
        number_of_unique_movies = 5
        expected_coverage = number_of_unique_movies / evaluator.TOTAL_NUMBER_OF_MOVIES

        self.assertEqual(evaluator.calculateCoverage(recommendations_for_all_users_map), expected_coverage)

    def test_calculate_coverage_with_empty_recommendations_map(self):
        recommendations_for_all_users_map = {}
        self.assertEqual(evaluator.calculateCoverage(recommendations_for_all_users_map), 0.0)

    def test_calculate_gini_simpson_diversity(self):
        recommendations_list = [101, 102, 103, 104]
        
        # movie genre map for each movie id in the recommendation list
        movie_genre_map = {
            101: "Action|Adventure|Drama",
            102: "Action|Comedy",
            103: "Drama|Romance",
            104: "Action|Comedy"
        }

        result = evaluator.calculateGiniSimpsonDiversity(recommendations_list, movie_genre_map)

        # compute expected gini simpson diversity
        genre_counts = {"Action": 3, "Comedy": 2, "Adventure": 1, "Drama": 2, "Romance": 1}
        total_genres = 9 # each movie has 2 genres except movie 101 which has 3, so total genres is 3+2+2+2=9
        proportions = [count / total_genres for count in genre_counts.values()]
        expected_diversity = 1 - sum(p ** 2 for p in proportions) # calculate using gini-simpson formula
        
        self.assertEqual(result, expected_diversity)

    def test_calculate_gini_simpson_diversity_with_empty_movie_genre_map(self):
        recommendations_list = [101, 102, 103, 104]
        # sample movie genre map
        movie_genre_map = {}
        result = evaluator.calculateGiniSimpsonDiversity(recommendations_list, movie_genre_map)
        
        # if recommendations list is empty, diversity is expected to be 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_gini_simpson_diversity_with_empty_recommendations_list(self):
        recommendations_list = []
        # sample movie genre map
        movie_genre_map = {
            101: "Action|Adventure|Drama",
            102: "Action|Comedy",
            103: "Drama|Romance",
            104: "Action|Comedy"
        }
        result = evaluator.calculateGiniSimpsonDiversity(recommendations_list, movie_genre_map)
        
        # if recommendations list is empty, diversity is expected to be 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_average_gini_simpson_diversity_for_three_users(self):
        recommendations_map = {
            1: [101, 102, 103],
            2: [103, 104],
            3: [101, 102],
        }
        
        # movie genre map for each movie id in the recommendation list
        movie_genre_map = {
            101: "Action|Adventure|Drama",
            102: "Action|Comedy",
            103: "Drama|Romance",
            104: "Action|Comedy"
        }

        result = evaluator.calculateAverageGiniSimpsonDiversityAcrossAllUsers(recommendations_map, movie_genre_map)

        # compute expected gini simpson diversity for each user

        # user one - movies 101, 102, 103
        genre_counts = {"Action": 2, "Comedy": 1, "Adventure": 1, "Drama": 2, "Romance": 1}
        total_genres = 7
        proportions = [count / total_genres for count in genre_counts.values()]
        expected_user_one_diversity = 1 - sum(p ** 2 for p in proportions) # calculate using gini-simpson formula

        # user two - movies 103, 104
        genre_counts = {"Action": 1, "Comedy": 1, "Drama": 1, "Romance": 1}
        total_genres = 4
        proportions = [count / total_genres for count in genre_counts.values()]
        expected_user_two_diversity = 1 - sum(p ** 2 for p in proportions) # calculate using gini-simpson formula

        # user three - movies 101, 102
        genre_counts = {"Action": 2, "Comedy": 1, "Drama": 1, "Adventure": 1}
        total_genres = 5
        proportions = [count / total_genres for count in genre_counts.values()]
        expected_user_three_diversity = 1 - sum(p ** 2 for p in proportions) # calculate using gini-simpson formula

        expected_average_diversity = (expected_user_one_diversity + expected_user_two_diversity + expected_user_three_diversity) / 3
        
        self.assertEqual(result, expected_average_diversity)

    def test_calculate_average_gini_simpson_diversity_with_empty_recommendations_map(self):
        recommendations_map = {}
        # sample movie genre map
        movie_genre_map = {
            101: "Action|Adventure|Drama",
            102: "Action|Comedy",
            103: "Drama|Romance",
            104: "Action|Comedy"
        }
        result = evaluator.calculateAverageGiniSimpsonDiversityAcrossAllUsers(recommendations_map, movie_genre_map)
        
        # if recommendations map is empty, diversity is expected to be 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_average_gini_simpson_diversity_with_empty_movie_genre_map(self):
        recommendations_map = {
            1: [101, 102, 103],
            2: [103, 104],
            3: [101, 102],
        }
        # sample movie genre map
        movie_genre_map = {}
        result = evaluator.calculateAverageGiniSimpsonDiversityAcrossAllUsers(recommendations_map, movie_genre_map)
        
        # if movie genre map is empty, diversity is expected to be 0.0
        self.assertEqual(result, 0.0)

    def test_calculate_movie_popularity_for_all_movies(self):
        recommendations_map = {
            1: [101, 102, 103],
            2: [103, 104],
            3: [101, 102, 103, 104],
        }

        result = evaluator.generateMoviePopularityForAllMovies(recommendations_map)

        expected_movie_popularity_map = {
            101: 2, # movie 101 appears twice across all recommendations
            102: 2, # movie 102 appears twice across all recommendations
            103: 3, # movie 103 appears three times across all recommendations
            104: 2 # movie 104 appears twice across all recommendations
        }

        self.assertEqual(result, expected_movie_popularity_map)

    def test_calculate_movie_popularity_for_all_movies_with_empty_recommendations_map(self):
        recommendations_map = {}

        result = evaluator.generateMoviePopularityForAllMovies(recommendations_map)

        self.assertEqual(result, {}) # expected empty map if input of recommendations map is an empty map

    def test_calculate_average_novelty_for_a_list_of_recommendations(self):
        recommendations_list = [101, 102, 103, 104, 105]
        total_number_of_users = 1000
        
        movie_popularity_map = {
            101: 50,  # Movie 101 recommended to 50 users
            102: 100, # Movie 102 recommended to 100 users
            103: 10,  # Movie 103 recommended to 10 users
            104: 200, # Movie 104 recommended to 200 users
            105: 500  # Movie 105 recommended to 500 users
        }

        result = evaluator.calculateAverageNoveltyForAListOfRecommendations(
            recommendations_list, total_number_of_users, movie_popularity_map
        )

        # compute the expected average novelty for all movies in the recommendations in the list
        total_novelty = 0
        for movieId in recommendations_list:
            popularity = movie_popularity_map[movieId]
            novelty = math.log2(total_number_of_users / popularity)
            total_novelty += novelty
        
        expected_average_novelty = total_novelty / len(recommendations_list)

        # Assert that the calculated average novelty is as expected
        self.assertEqual(result, expected_average_novelty)

    def test_calculate_average_novelty_with_empty_recommendations_list(self):
        recommendations_list = {}
        total_number_of_users = 1000
        movie_popularity_map = {
            101: 50, 
            102: 100,
        }

        self.assertEqual(evaluator.calculateAverageNoveltyForAListOfRecommendations(recommendations_list, total_number_of_users, movie_popularity_map), 0.0)


    def getMockRatingsData(self):
        return [
            {"userId": 1, "movieId": 101, "rating": 5},
            {"userId": 2, "movieId": 101, "rating": 5},
            {"userId": 2, "movieId": 102, "rating": 4},
            {"userId": 3, "movieId": 101, "rating": 2},
            {"userId": 3, "movieId": 103, "rating": 4},
        ]
