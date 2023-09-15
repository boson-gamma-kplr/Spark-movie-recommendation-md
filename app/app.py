from flask import Blueprint, render_template, Flask, request
main = Blueprint('main', __name__)

from pyspark.sql import SparkSession
from engine import RecommendationEngine

@main.route('/', methods=["GET", "POST", "PUT"])
def index():
    return render_template("index.html")



def create_app(spark_context, movies_set_path, ratings_set_path):
	global recommendation_engine

	recommendation_engine = RecommendationEngine(spark_context, movies_set_path, ratings_set_path)
	app = Flask(__name__)
	app.register_blueprint(main)
	app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
	app.config['TEMPLATES_AUTO_RELOAD'] = True
	return app