from pyspark.sql.types import *
from pyspark.sql.functions import explode, col, count
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

# - `pyspark.sql.types` : Importe les types de données nécessaires pour définir la structure du schéma des ensembles de données.
# - `pyspark.sql.functions` : Importe les fonctions nécessaires pour effectuer des opérations sur les colonnes des ensembles de données.
# - `pyspark.ml.recommendation.ALS` : Importe l'algorithme ALS (Alternating Least Squares) utilisé pour le filtrage collaboratif.
# - `pyspark.ml.evaluation.RegressionEvaluator` : Importe l'évaluateur de régression utilisé pour évaluer les performances du modèle.
# - `pyspark.sql.SQLContext` : Importe le contexte SQL utilisé pour créer une session Spark.

# **Étape 2: Définition de la classe RecommendationEngine**

# **Méthode create_user(self, user_id) :**

# - Cette méthode permet de créer un nouvel utilisateur.
# - Elle prend en paramètre un `user_id` facultatif pour spécifier l'identifiant de l'utilisateur. Si `user_id` est `None`, un nouvel identifiant est généré automatiquement.
# - Si `user_id` est supérieur à `max_user_identifier`, `max_user_identifier` est mis à jour avec la valeur de `user_id`.
# - La méthode retourne l'identifiant de l'utilisateur créé ou mis à jour.

# **Méthode is_user_known(self, user_id) :**

# - Cette méthode permet de vérifier si un utilisateur est connu.
# - Elle prend en paramètre un `user_id` et retourne `True` si l'utilisateur est connu (c'est-à-dire si `user_id` est différent de `None` et inférieur ou égal à `max_user_identifier`), sinon elle retourne `False`.

# **Méthode get_movie(self, movie_id) :**

# - Cette méthode permet d'obtenir un film.
# - Elle prend en paramètre un `movie_id` facultatif pour spécifier l'identifiant du film. Si `movie_id` est `None`, la méthode retourne un échantillon aléatoire d'un film à partir du dataframe `best_movies_df`. Sinon, elle filtre le dataframe `movies_df` pour obtenir le film correspondant à `movie_id`.
# - La méthode retourne un dataframe contenant les informations du film (colonne "movieId" et "title").

# **Méthode get_ratings_for_user(self, user_id) :**

# - Cette méthode permet d'obtenir les évaluations d'un utilisateur.
# - Elle prend en paramètre un `user_id` et filtre le dataframe `ratings_df` pour obtenir les évaluations correspondantes à l'utilisateur.
# - La méthode retourne un dataframe contenant les évaluations de l'utilisateur (colonnes "movieId", "userId" et "rating").

# **Méthode add_ratings(self, user_id, ratings) :**

# - Cette méthode permet d'ajouter de nouvelles évaluations au modèle et de re-entraîner le modèle.
# - Elle prend en paramètres un `user_id` et une liste de `ratings` contenant les nouvelles évaluations.
# - La méthode crée un nouveau dataframe `new_ratings_df` à partir de la liste de `ratings` et l'ajoute au dataframe existant `ratings_df` en utilisant l'opération `union()`.
# - Ensuite, les données sont divisées en ensembles d'entraînement (`training`) et de test (`test`) en utilisant la méthode `randomSplit()`.
# - Enfin, la méthode privée `__train_model()` est appelée pour re-entraîner le modèle.

# **Méthode predict_rating(self, user_id, movie_id) :**

# - Cette méthode permet de prédire une évaluation pour un utilisateur et un film donnés.
# - Elle prend en paramètres un `user_id` et un `movie_id`.
# - La méthode crée un dataframe `rating_df` à partir des données (`user_id`, `movie_id`) et le transforme en utilisant le modèle pour obtenir les prédictions.
# - Si le dataframe de prédiction est vide, la méthode retourne `-1`, sinon elle retourne la valeur de prédiction.

# **Méthode recommend_for_user(self, user_id, nb_movies) :**

# - Cette méthode permet d'obtenir les meilleures recommandations pour un utilisateur donné.
# - Elle prend en paramètres un `user_id` et un nombre de films `nb_movies` à recommander.
# - La méthode crée un dataframe `user_df` contenant l'identifiant de l'utilisateur et utilise la méthode `recommendForUserSubset()` du modèle pour obtenir les recommandations pour cet utilisateur.
# - Les recommandations sont ensuite jointes avec le dataframe `movies_df` pour obtenir les détails des films recommandés.
# - Le dataframe résultant est retourné avec les colonnes "title" et d'autres colonnes du dataframe `movies_df`.

# **Méthode __train_model(self) :**

# - Cette méthode privée permet d'entraîner le modèle avec l'algorithme ALS (Alternating Least Squares).
# - Elle utilise les paramètres `maxIter` et `regParam` définis dans l'initialisation de la classe pour créer une instance de l'algorithme ALS.
# - Ensuite, le modèle est entraîné en utilisant le dataframe `training`.
# - La méthode privée `__evaluate()` est appelée pour évaluer les performances du modèle.

# **Méthode __evaluate(self) :**

# - Cette méthode privée permet d'évaluer le modèle en calculant l'erreur quadratique moyenne (RMSE - Root-mean-square error).
# - Elle utilise le modèle pour prédire les évaluations sur le dataframe `test`.
# - Ensuite, elle utilise l'évaluateur de régression pour calculer le RMSE en comparant les prédictions avec les vraies évaluations.
# - La valeur de RMSE est stockée dans la variable `rmse` de la classe et affichée à l'écran.

# **Méthode **init**(self, sc, movies_set_path, ratings_set_path) :**

# - Cette méthode d'initialisation est appelée lors de la création d'une instance de la classe RecommendationEngine.
# - Elle prend en paramètres le contexte Spark (`sc`), le chemin vers l'ensemble de données de films (`movies_set_path`) et le chemin vers l'ensemble de données d'évaluations (`ratings_set_path`).
# - La méthode initialise le contexte SQL à partir du contexte Spark, charge les données des ensembles de films et d'évaluations à partir des fichiers CSV spécifiés, définit le schéma des données, effectue diverses opérations de traitement des données et entraîne le modèle en utilisant la méthode privée `__train_model()`.

class RecommendationEngine:
    def __init__(self,sc, movies_set_path, ratings_set_path):
        # Méthode d'initialisation pour charger les ensembles de données et entraîner le modèle
        self.spark = SQLContext(sc).sparkSession
        self.movies_set_path_ = movies_set_path
        self.ratings_set_path = ratings_set_path

        self.movies_, self.ratings_ = self.__load_data()

        self.ratings_train_, self.ratings_test_ = self.ratings_.randomSplit([0.8, 0.2], seed=42)
        self.max_iter_ = 5
        self.reg_param_ = 0.05

        self.model_ = None

    def create_user(self, user_id):
        # Méthode pour créer un nouvel utilisateur
        if self.is_user_known(user_id):
            print(f'User {user_id} already exists')
            return False
        elif not isinstance(user_id, int):
            print(f'user_id must be an integer')
            return False
        else :
            self.new_user_id_ = user_id

            new_row = self.spark.createDataFrame([(user_id, None, None, None)], self.ratings_.columns)

            self.ratings_.union(new_row)
            return True

    def is_user_known(self, user_id):
        # Méthode pour vérifier si un utilisateur est connu
        users = self.get_ratings_for_user(user_id)

        return (users.count()>0)

    def get_movie(self, movie_id):
        # Méthode pour obtenir un film
        movie = self.movies_.filter(self.movies_.movieId==movie_id)

        if (movie.count()==0):
            print(f'No movie found corresponding to id {movie_id}')
            return None

        print(f'found movie : {movie.first()["title"]}')
        return movie.first()


    def get_ratings_for_user(self, user_id):
        # Méthode pour obtenir les évaluations d'un utilisateur
        return self.ratings_.filter( self.ratings_['userId'] == user_id)

    def add_ratings(self, user_id, movie_ids, ratings, timestamps):
        # Méthode pour ajouter de nouvelles évaluations et re-entraîner le modèle
        new_rows = zip([user_id]*len(movie_ids), movie_ids,ratings,timestamps)
            
        new_rows_df = self.spark.createDataFrame(new_rows, self.ratings_.columns)

        self.ratings_.union(new_rows_df)


    def predict_rating(self, user_id, movie_id):

        if self.model_ == None:
            return None

        rating_df = self.spark.createDataFrame([(user_id, movie_id)],StructType(
            [StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True)]))

        prediction = self.model_.transform(rating_df)
        if (prediction.count() == 0):
            return None
        return prediction.collect()[0].asDict()["prediction"]

    def recommend_for_user(self, user_id, nb_movies):
        # Méthode pour obtenir les meilleures recommandations pour un utilisateur donné
        if self.model_ == None:
            return None
        
        user_df = self.spark.createDataFrame([user_id], "userCol")
        ratings = self.model_.recommendForUserSubset(user_df, nb_movies)
        user_recommandations = ratings.select(
             explode(col("recommendations").movieId).alias("movieId")
        )
        return user_recommandations.join(self.movies_df, "movieId").drop("genres").drop("movieId")

    def __train_model(self):
        # Méthode privée pour entraîner le modèle avec ALS
        als = ALS(maxIter=self.max_iter,
            regParam=self.reg_param,
            implicitPrefs=False,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop")

        self.model_ = als.fit(self.ratings_train_)
        self.__evaluate()

    def __evaluate(self):
        # Méthode privée pour évaluer le modèle en calculant l'erreur quadratique moyenne
        predictions = self.model_.transform(self.ratings_test_)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        self.rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(self.rmse))

    def __load_data(self):
        
        movies_df = self.spark.read.parquet(self.movies_set_path_)
        """
        root
        |-- movieId: long (nullable = true)
        |-- title: string (nullable = true)
        |-- genres: string (nullable = true)
        """
        ratings_df = self.spark.read.parquet(self.ratings_set_path)
        """
        root
        |-- userId: long (nullable = true)
        |-- movieId: long (nullable = true)
        |-- rating: double (nullable = true)
        |-- timestamp: long (nullable = true)
        """
        movies_df.cache()
        ratings_df.cache()

        return movies_df, ratings_df


# Création d'une instance de la classe RecommendationEngine
if __name__=="__main__":
    sc = SparkSession \
        .builder \
        .appName("Movies recommandations") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
        
    engine = RecommendationEngine(sc, "app/Data/movies.parquet", "app/Data/ratings.parquet")

    # Exemple d'utilisation des méthodes de la classe RecommendationEngine
    if engine.is_user_known(9999999999):
        print('user exist')
    else:
        print('user does not exist')

    user_id = 1
    engine.create_user(user_id)
    if engine.is_user_known(user_id):
        movie = engine.get_movie(1)
        ratings = engine.get_ratings_for_user(user_id)
        # engine.add_ratings(user_id, [],[],[])
        if movie is not None:
            prediction = engine.predict_rating(user_id, movie.movieId)
        recommendations = engine.recommend_for_user(user_id, 10)
