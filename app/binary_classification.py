"""Binary classification pipeline."""

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer, SQLTransformer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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

data_projected = tweets_df.select(tweets_df.embedding.alias("embedding"),
                                  tweets_df.likes.alias("likes"),
                                  tweets_df.retweets.alias("retweets"),
                                  tweets_df.replies.alias("replies"),
                                  tweets_df.main_tag.alias('main_tag'))

train, test = data_projected.randomSplit([0.9, 0.1], seed=12345)

# PIPELINE
vec_embedding = spark.udf.register(
    "vec_embedding",
    lambda a: Vectors.dense(a), VectorUDT()
)

parse_vector_udf_transformer = SQLTransformer(
    statement="SELECT main_tag,likes,retweets,"
              "replies, vec_embedding(embedding) vec_embedding FROM __THIS__")

assembler = VectorAssembler(
    inputCols=["likes", "retweets", "replies", "vec_embedding"],
    outputCol="features")

label_stringIdx = StringIndexer(inputCol="main_tag", outputCol="label")

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

pipeline = Pipeline(stages=[parse_vector_udf_transformer,
                            assembler, label_stringIdx, lr])

# Training and prediction
model = pipeline.fit(train)
prediction = model.transform(test)

# Evaluation
evaluator = MulticlassClassificationEvaluator(metricName="f1")

# Printouts
print(f"TWEETS DB SCHEMA: \n")
tweets_df.printSchema()
print(f"PROJECTION: \n")
data_projected.show()
print(f"PREDICTIONS: \n")
prediction.show()
print(f"F1_SCORE: {evaluator.evaluate(prediction)}")
