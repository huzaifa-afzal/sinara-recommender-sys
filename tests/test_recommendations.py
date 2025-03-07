import unittest
from unittest.mock import patch
import pandas as pd
from recommendations import movie_recommendations

class TestMovieRecommendations(unittest.TestCase):

    @patch("database.dao.get_ratings_data") 
    def test_generate_recommendations(self, mock_get_ratings_data):
        mock_get_ratings_data.return_value = self.getMockRatingsData()

        user_item_matrix = movie_recommendations.build_user_item_matrix()
        user_similarity = movie_recommendations.build_user_to_user_similarity_matrix(user_item_matrix)
        recommendations = movie_recommendations.generate_recommendations(
            userId=1, 
            excludeAlreadyWatchedMovies=True, 
            user_item_matrix=user_item_matrix, 
            user_similarity=user_similarity
        )

        # User most similar to user Id 1 is user 2 (as they have both rated movie 101 highly as 5), more similar than user 3
        # because user 3 rated movie 101 2 (while user 1 rated it as 5). As a result, user 2's movies, that user 1 has not seen,
        # are rated higher than user 3's movies in regards to user 1's recommendations. Therefore, 102 should be the first recommendation, followed by 103
        expected_recommendations = [102, 103]
        self.assertEqual(recommendations, expected_recommendations)

    @patch("database.dao.get_ratings_data") 
    def test_generate_recommendations_without_excluding_already_watched_movies(self, mock_get_ratings_data):
        mock_get_ratings_data.return_value = self.getMockRatingsData()

        user_item_matrix = movie_recommendations.build_user_item_matrix()
        user_similarity = movie_recommendations.build_user_to_user_similarity_matrix(user_item_matrix)
        recommendations = movie_recommendations.generate_recommendations(
            userId=1, 
            excludeAlreadyWatchedMovies=False, 
            user_item_matrix=user_item_matrix, 
            user_similarity=user_similarity
        )

        # User most similar to user Id 1 is user 2 (as they have both rated movie 101 highly as 5), more similar than user 3
        # because user 3 rated movie 101 2 (while user 1 rated it as 5). As a result, user 2's movies, that user 1 has not seen,
        # are rated higher than user 3's movies in regards to user 1's recommendations.
        # The highest ranked recommendation will however be 101 because user 2 rated that as 5, as did user 1, and excludeAlreadyWatchedMovies flag is set to False
        expected_recommendations = [101, 102, 103]
        self.assertEqual(recommendations, expected_recommendations)

    @patch("database.dao.get_ratings_data") 
    def test_build_matrix_with_data(self, mock_get_ratings_data):
        mock_get_ratings_data.return_value = self.getMockRatingsData()
        result = movie_recommendations.build_user_item_matrix()

        expected_df = pd.DataFrame({
            101: {1: 5.0, 2: 5.0, 3: 2.0},
            102: {1: 0.0, 2: 4.0, 3: 0.0},
            103: {1: 0.0, 2: 0.0, 3: 4.0} # default value is 0.0 if no rating specified
        }).fillna(0)
        expected_df.index.name = "userId"
        expected_df.columns.name = "movieId" # set the row and column titles for the expected dataframe

        pd.testing.assert_frame_equal(result, expected_df)

    @patch("database.dao.get_ratings_data")
    def test_build_matrix_with_empty_data(self, mock_get_ratings_data):
        mock_get_ratings_data.return_value = [] # mock no result
        result = movie_recommendations.build_user_item_matrix()
        self.assertIsNone(result)  # assert the result is 'None'

    def test_build_user_to_user_similarity_matrix(self):
        # sample user-item-matrix
        user_item_matrix = pd.DataFrame({
            101: {1: 5.0, 2: 3.0},  
            102: {1: 4.0, 2: 0.0}
        }).fillna(0)

        result = movie_recommendations.build_user_to_user_similarity_matrix(user_item_matrix)

        # similarity between users calculated via cosine similarity based on their ratings
        expected_similarity = pd.DataFrame({
            1: {1: 1.0, 2: 0.780869},  
            2: {1: 0.780869, 2: 1.0}  
        })

        pd.testing.assert_frame_equal(result, expected_similarity)
    
    def test_get_user_already_watched_movies(self):
        user_item_matrix = pd.DataFrame({
            101: {1: 5.0, 2: 4.0, 3: 0.0},
            102: {1: 0.0, 2: 3.0, 3: 4.0},
            103: {1: 0.0, 2: 0.0, 3: 5.0}
        })

        user_one_watched_movies = {101}
        user_two_watched_movies = {101, 102}
        user_three_watched_movies = {102, 103}
        self.assertEqual(user_one_watched_movies, movie_recommendations.get_user_already_watched_movies(1, user_item_matrix))
        self.assertEqual(user_two_watched_movies, movie_recommendations.get_user_already_watched_movies(2, user_item_matrix))
        self.assertEqual(user_three_watched_movies, movie_recommendations.get_user_already_watched_movies(3, user_item_matrix))

    def test_get_user_already_watched_movies_with_invalid_user(self):
        user_item_matrix = pd.DataFrame({
            101: {1: 5.0, 2: 4.0, 3: 0.0},
            102: {1: 0.0, 2: 3.0, 3: 4.0},
            103: {1: 0.0, 2: 0.0, 3: 5.0}
        })

        self.assertEqual(set(), movie_recommendations.get_user_already_watched_movies(5, user_item_matrix))

    def test_build_user_to_user_similarity_matrix_with_empty_input(self):
        # Use empty df as input
        result = movie_recommendations.build_user_to_user_similarity_matrix(pd.DataFrame())
        self.assertIsNone(result)

    def test_build_user_to_user_similarity_matrix_with_None_input(self):
        result = movie_recommendations.build_user_to_user_similarity_matrix(None)
        self.assertIsNone(result)

    def test_find_most_similar_users_for_specific_user(self):

        user_similarity = pd.DataFrame({
            1: {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.6, 5: 0.8},
            2: {1: 0.9, 2: 1.0, 3: 0.8, 4: 0.7, 5: 0.6},
            3: {1: 0.7, 2: 0.8, 3: 1.0, 4: 0.9, 5: 0.5},
            4: {1: 0.6, 2: 0.7, 3: 0.9, 4: 1.0, 5: 0.4},
            5: {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.4, 5: 1.0},
        })

        # find users most similar to user Id 1,
        result = movie_recommendations.find_most_similar_users_for_specific_user(1, user_similarity)

        # similiar users to 1, in order, are: 2, 5, 3, 4
        expected_result = [(2, 0.9), (5, 0.8), (3, 0.7), (4, 0.6)]

        # Assert if the result is equal to expected result
        self.assertEqual(result, expected_result)

    def test_find_most_similar_users_for_non_existent_user(self):

        user_similarity = pd.DataFrame({
            1: {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.6, 5: 0.8},
            2: {1: 0.9, 2: 1.0, 3: 0.8, 4: 0.7, 5: 0.6},
            3: {1: 0.7, 2: 0.8, 3: 1.0, 4: 0.9, 5: 0.5},
            4: {1: 0.6, 2: 0.7, 3: 0.9, 4: 1.0, 5: 0.4},
            5: {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.4, 5: 1.0},
        })

        # userId 6 does not exist
        result = movie_recommendations.find_most_similar_users_for_specific_user(6, user_similarity)

        self.assertEqual(result, [])

    def test_find_most_similar_users_with_empty_input(self):
        # find users most similar to user Id 1,
        result = movie_recommendations.find_most_similar_users_for_specific_user(1, pd.DataFrame())

        # Assert if the result is equal to expected result
        self.assertEqual(result, [])

    def test_find_most_similar_user_only_one_user_output(self):
        user_similarity = pd.DataFrame({
            1: {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.6, 5: 0.8},
            2: {1: 0.9, 2: 1.0, 3: 0.8, 4: 0.7, 5: 0.6},
            3: {1: 0.7, 2: 0.8, 3: 1.0, 4: 0.9, 5: 0.5},
            4: {1: 0.6, 2: 0.7, 3: 0.9, 4: 1.0, 5: 0.4},
            5: {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.4, 5: 1.0},
        })

        # find user most similar to user Id 3
        result = movie_recommendations.find_most_similar_users_for_specific_user(4, user_similarity, n=1) # Only want the top similar user, so change value of n

        # User with id 4 has the most similar user as user 3
        self.assertEqual(result, [(3, 0.9)])

    def getMockRatingsData(self):
        return [
            {"userId": 1, "movieId": 101, "rating": 5},
            {"userId": 2, "movieId": 101, "rating": 5},
            {"userId": 2, "movieId": 102, "rating": 4},
            {"userId": 3, "movieId": 101, "rating": 2},
            {"userId": 3, "movieId": 103, "rating": 4},
        ]
