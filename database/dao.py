from pymongo import MongoClient

# Generally want to avoid hard-coding the connection string, however to avoid the user having to create their own account, a connection string for a default user was generated
client = MongoClient("mongodb+srv://movieuser:RBgISdEE2NHZJ5A6@cluster0.zhkvf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0") 
db = client["movieDB"]

def get_ratings_data():
    # 'ratings' collection in the database has documents with fields including: 'userId', 'movieId', 'rating', ratings (1-5) that users make for a particular movie
    ratings_data = list(db["ratings"].find({}, {"_id": 0, "userId": 1, "movieId": 1, "rating": 1}))
    return ratings_data

def get_movie_title_from_id(movieId):
    if not movieId or not isinstance(movieId, int):
        return None

    movie = db["movies"].find_one({"movieId": movieId})

    if movie is None:
        return None

    return movie["title"]

# builds and returns a map of movie -> genre mappings
def build_movie_genre_map():
    
    movieGenreMap = {}
    movies = db["movies"].find({}, {"movieId":1, "genres":1})

    if movies is None:
        return None

    for movie in movies:
        movieGenreMap[movie["movieId"]] = movie["genres"]

    return movieGenreMap