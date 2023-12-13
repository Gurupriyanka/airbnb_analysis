from pyspark.sql import SparkSession
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def main(input_file1, input_file2):

    positive_df = spark.read.csv(input_file1, header=True, inferSchema=True)
    negative_df = spark.read.csv(input_file2, header=True, inferSchema=True)

    positive_ngram_df = positive_df.select('ngram').toPandas()
    negative_ngram_df = negative_df.select('ngram').toPandas()
    
    for df in [positive_df, negative_df]:
        #concatenate all ngrams into a string
        text = df.select(concat_ws(" ", collect_list("ngram"))).first()[0]

        #generate word cloud image
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_font_size=150, relative_scaling=0.5).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

if __name__ == '__main__':
    
#     input_file1 = 'Documents/ML/Project/collocation-output/positive/'
#     input_file2 = 'Documents/ML/Project/collocation-output/negative/'
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    
    spark = SparkSession.builder.appName('sentiment analysis').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')  # Set log level on SparkContext

    main(input_file1, input_file2)
