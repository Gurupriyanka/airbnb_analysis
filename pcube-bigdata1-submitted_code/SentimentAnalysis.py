import sys
import json
assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
import os
import re
import string
import nltk
from nltk import pos_tag, download
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.functions import udf, col, lower, regexp_replace
from pyspark.sql.types import StringType
from pyspark.sql.types import StringType, FloatType, StructType, StructField, IntegerType
from textblob import TextBlob
from langdetect import detect
from pyspark.sql import DataFrame

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def detect_language(text):
    try:
        return detect(text)
    except:
        return None

def remove_stopwords(text):
    if not isinstance(text, str):
        return text
    words = text.split()
    #get set of English stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def lemmatize_words(data_str):
    wordlist = 0
    cleaned_text = ''
    x = WordNetLemmatizer()
    text = data_str.split()
    tagged_words = pos_tag(text)
    for word in tagged_words:
        if 'v' in word[1].lower():
            y = x.lemmatize(word[0], pos='v')
        else:
            y = x.lemmatize(word[0], pos='n')
        if wordlist == 0:
            cleaned_text = y
        else:
            cleaned_text = cleaned_text + ' ' + y
        wordlist += 1
    return cleaned_text

def sentiment_score(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return None

def categorize_sentiment(score):
    if score > 0.5:
        return 'positive'
    elif score < -0.5:
        return 'negative'
    else:
        return 'neutral'

def process_text(text):
    if text is None:
        return None

    text = text.lower()
    
    text = re.sub(r'https?://\S+', '', text)  #remove URLs
    text = re.sub(r'@\w+', '', text)  #remove mentions
    text = re.sub(r'\d+', '', text)  #remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  #remove punctuation
    
    text = re.sub(r'\b(thats|ive|im|ya|cant|dont|wont|id|r|u|k)\b', 
                  lambda match: {'thats': 'that is', 'ive': 'i have', 'im': 'i am', 'ya': 'yeah', 'cant': 'can not',
                                 'dont': 'do not', 'wont': 'will not', 'id': 'i would', 'r': 'are', 'u': 'you', 
                                 'k': 'OK'}[match.group()], text)
    return text


spark = SparkSession.builder.appName('sentiment analysis').getOrCreate()
detect_language_udf = udf(detect_language, StringType())
sentiment_score_udf = udf(sentiment_score, FloatType())
categorize_sentiment_udf = udf(categorize_sentiment, StringType())
process_text_udf = udf(process_text, StringType())
remove_stopwords_udf = udf(remove_stopwords, StringType())
lemmatize_udf = udf(lemmatize_words, StringType())

#Processing function
def process_reviews(file_path):
    
    review_schema = StructType([
        StructField("listing_id", StringType(), True),
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("reviewer_id", StringType(), True),
        StructField("reviewer_name", StringType(), True),
        StructField("comments", StringType(), True),
        StructField("city", StringType(), True),
    ])
    
    # df = spark.read.schema(review_schema).json(file_path).repartition(20)
    df = spark.read.format("s3selectJSON").schema(review_schema).load(file_path).repartition(50)
    
    #Data preprocessing
    df = df.dropna(subset=['comments'])  # Drop rows where 'comments' is null
    df = df.withColumn('comments_processed', process_text_udf(col('comments')))
    df = df.withColumn('comments_processed', remove_stopwords_udf(col('comments_processed')))
    df = df.withColumn('comments_processed', lemmatize_udf(col('comments_processed')))
    df = df.withColumn('language', detect_language_udf(col('comments_processed')))
    df = df.filter(col('language') == 'en')

    df.cache()

    #Sentiment analysis
    df = df.withColumn('sentiment_score', sentiment_score_udf(col('comments_processed')))
    df = df.withColumn('sentiment_category', categorize_sentiment_udf(col('sentiment_score')))
    
    #group by 'sentiment_category' and count the rows in each group
    sentiment_counts = df.groupBy("sentiment_category").count()
    sentiment_counts.show()
    
    df = df.filter(col('sentiment_category').isin("positive", "neutral", "negative"))    
    df.show()
    return df

if __name__ == '__main__':
    # folder_path = "Documents/ML/Project/reviews/"
    # output_path = "Documents/ML/Project/output"
    folder_path = sys.argv[1]
    output_path = sys.argv[2]
    
    processed_df = process_reviews(folder_path)
    if processed_df:
        processed_df.write.mode('overwrite').csv(output_path, header=True)



