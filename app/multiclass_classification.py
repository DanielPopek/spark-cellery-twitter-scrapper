"""Linear regression pipeline."""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Bucketizer

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

data_projected = tweets_df.select(tweets_df.likes.alias("likes"),
                                  tweets_df.retweets.alias("retweets"),
                                  tweets_df.replies.alias("replies"))

train, test = data_projected.randomSplit([0.9, 0.1], seed=12345)

# PIPELINE

splits = [-float("inf"), 0.0, 1, 2, 4, 5, float("inf")]

assembler = VectorAssembler(
    inputCols=["likes", "replies"],
    outputCol="features")

bucketizer = Bucketizer(splits=splits, inputCol="retweets", outputCol="label")

# label_stringIdx = StringIndexer(inputCol="main_tag", outputCol="label")

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

pipeline = Pipeline(stages=[assembler, bucketizer, lr])

# Training and prediction
model = pipeline.fit(train)
prediction = model.transform(test)

# Evaluation
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(prediction)

evaluatorf1 = MulticlassClassificationEvaluator(metricName="f1")
f1 = evaluatorf1.evaluate(prediction)

evaluatorwp = MulticlassClassificationEvaluator(metricName="weightedPrecision")
wp = evaluatorwp.evaluate(prediction)

evaluatorwr = MulticlassClassificationEvaluator(metricName="weightedRecall")
wr = evaluatorwr.evaluate(prediction)

# Printouts
print(f"TWEETS DB SCHEMA: \n")
tweets_df.printSchema()
print(f"PROJECTION: \n")
data_projected.show()
print(f"PREDICTIONS: \n")
prediction.show()
print("Accuracy = %g" % accuracy)
print("f1 = %g" % f1)
print("weightedPrecision = %g" % wp)
print("weightedRecall = %g" % wr)
