import os
import sys
import json
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StringType, FloatType, StructType, StructField, IntegerType
from pyspark.sql.functions import explode, lower, array_contains, collect_list, desc, rank
from pyspark.sql.types import ArrayType
from pyspark.ml.feature import NGram
from pyspark.ml.feature import Tokenizer as MLTokenizer
import string
from pyspark.sql.window import Window

import sparknlp
from sparknlp.base import *
from sparknlp.annotator import SentenceDetector, NerDLModel
from pyspark.ml import Pipeline
from sparknlp.annotator import Tokenizer as NLPTokenizer

#to remove extended stopwords
def remove_extended_stopwords(words):
    extended_stopwords = {'br', 'st', 'gonna', 'gotta', 'btw', 'fyi', 'imo', 'yall', 'wanna', 'sus'}
    return [word for word in words if word.lower() not in extended_stopwords]

#generate n-grams
def generate_ngrams(df, n):
    ngram = NGram(n=n, inputCol="words", outputCol="ngrams")
    return ngram.transform(df)

# Save DataFrames to CSV files
def save_df_to_csv(df, path):
    df.write.mode('overwrite').csv(path, header=True)

def main(file_path, output_path):
    review_schema = StructType([
        StructField("city", StringType(), True),
        StructField("listing_id", StringType(), True),
        StructField("sentiment_category", StringType(), True),
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("reviewer_id", StringType(), True),
        StructField("reviewer_name", StringType(), True),
        StructField("comments", StringType(), True),
        StructField("comments_processed", StringType(), True),
        StructField("language", StringType(), True),
    ])

    reviews = spark.read.schema(review_schema).json(file_path).repartition(50)

    remove_extended_stopwords_udf = udf(remove_extended_stopwords, ArrayType(StringType()))

    reviews_df = reviews.withColumn("comments_processed", remove_extended_stopwords_udf(col("comments_processed")))

    reviews = reviews.filter(col("language") == "en")
    reviews.cache()

    tokenizer = MLTokenizer(inputCol="comments_processed", outputCol="words")
    reviews_df = tokenizer.transform(reviews)

    #process each sentiment category
    sentiments = ['positive', 'negative']
    n_values = [2, 3]  # Bigrams and Trigrams

    for sentiment in sentiments:
        print(f"Processing {sentiment} reviews")
        sentiment_df = reviews_df.filter(col("sentiment_category") == sentiment)

        for n in n_values:
            print(f"Generating {n}-grams")
            ngram_df = generate_ngrams(sentiment_df, n)
            # Explode n-grams to count
            exploded_df = ngram_df.withColumn("ngram", explode(col("ngrams")))
            #Aggregate n-gram counts by listing_id
            ngram_counts = exploded_df.groupby("listing_id", "ngram", "sentiment_category").count()
            #top n-grams for each listing
            windowSpec = Window.partitionBy("listing_id").orderBy(desc("count"))
            top_ngrams_df = ngram_counts.select("listing_id", "ngram", "sentiment_category")
            
            # Save the DataFrame to the CSV file
            csv_filename = os.path.join(output_path, f"top_ngrams_{sentiment}.csv")
            save_df_to_csv(top_ngrams_df, csv_filename)

if __name__ == '__main__':

    file_path = sys.argv[1]
    output_path = sys.argv[2]

    spark = SparkSession.builder.appName('sentiment analysis').getOrCreate()

    spark.sparkContext.setLogLevel('WARN')  # Set log level on SparkContext

    main(file_path, output_path)