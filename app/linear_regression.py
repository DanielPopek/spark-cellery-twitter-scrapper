"""Linear regression pipeline."""

from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import SQLTransformer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# DATA PREPARATION
spark = SparkSession \
    .builder \
    .master('local') \
    .config('spark.mongodb.input.uri',
            'mongodb://my_mongo:27017/tweetmldb.newtweets') \
    .config('spark.mongodb.output.uri',
            'mongodb://my_mongo:27017/tweetmldb.newtweets') \
    .getOrCreate()

tweets_df = spark.read \
    .format("com.mongodb.spark.sql.DefaultSource") \
    .option("database", "tweetmldb") \
    .option("collection", "newtweets") \
    .load()

print(f"tutaj{tweets_df.count}")

data_projected = tweets_df.select(tweets_df.embedding.alias("embedding"),
                                  tweets_df.likes.alias("label"))

train, test = data_projected.randomSplit([0.9, 0.1], seed=12345)

# PIPELINE
vec_embedding = spark.udf.register(
    "vec_embedding",
    lambda a: Vectors.dense(a), VectorUDT()
)

parse_vector_udf_transformer = SQLTransformer(
    statement="SELECT label,"
              "vec_embedding(embedding) features FROM __THIS__")

lr = LinearRegression()

pipeline = Pipeline(stages=[parse_vector_udf_transformer, lr])

# Training and prediction
model = pipeline.fit(train)
prediction_train = model.transform(train)
prediction_test = model.transform(test)

# Evaluation
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse_train = evaluator.evaluate(prediction_train)
rmse_test = evaluator.evaluate(prediction_test)

# Printouts
prediction_test.show()
print("Root Mean Squared Error (RMSE) on train data = %g" % rmse_train)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_test)
