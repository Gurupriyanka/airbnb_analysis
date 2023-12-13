from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
import sys
assert sys.version_info >= (3, 5)
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, split, lit, col, udf, regexp_extract, length, current_date, datediff, year, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType, TimestampType
from datetime import datetime

from pyspark.sql.functions import when, count, sum, expr, regexp_replace, radians, sin, cos, atan2, sqrt
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, ArrayType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

# Function to compute correlation and return the output as a string
def compute_correlation(data, column1, column2):
    selected_data = data.withColumn("distance_to_city_center_km", data[column1].cast(FloatType()))
    selected_data = data.withColumn(column1, data[column1].cast(FloatType()))

    correlation = selected_data.stat.corr(column1, column2)
    output_text = f"Correlation between {column1} and {column2} of listing location: {correlation}"
    return output_text

def main(input1, input2, output1, output2, output3):
    airbnb_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("listing_url", StringType(), True),
    StructField("scrape_id", LongType(), True),
    StructField("last_scraped", StringType(), True),
    StructField("source", StringType(), True),
    StructField("name", StringType(), True),
    StructField("description", StringType(), True),
    StructField("neighborhood_overview", StringType(), True),
    StructField("picture_url", StringType(), True),
    StructField("host_id", LongType(), True),
    StructField("host_url", StringType(), True),
    StructField("host_name", StringType(), True),
    StructField("host_since", StringType(), True),
    StructField("host_location", StringType(), True),
    StructField("host_about", StringType(), True),
    StructField("host_response_time", StringType(), True),
    StructField("host_response_rate", StringType(), True),
    StructField("host_acceptance_rate", StringType(), True),
    StructField("host_is_superhost", StringType(), True),
    StructField("host_thumbnail_url", StringType(), True),
    StructField("host_picture_url", StringType(), True),
    StructField("host_neighbourhood", StringType(), True),
    StructField("host_listings_count", IntegerType(), True),
    StructField("host_total_listings_count", IntegerType(), True),
    StructField("host_verifications", StringType(), True),
    StructField("host_has_profile_pic", StringType(), True),
    StructField("host_identity_verified", StringType(), True),
    StructField("neighbourhood", StringType(), True),
    StructField("neighbourhood_cleansed", StringType(), True),
    StructField("neighbourhood_group_cleansed", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("property_type", StringType(), True),
    StructField("room_type", StringType(), True),
    StructField("accommodates", IntegerType(), True),
    StructField("bathrooms", StringType(), True),
    StructField("bathrooms_text", StringType(), True),
    StructField("bedrooms", DoubleType(), True),
    StructField("beds", DoubleType(), True),
    StructField("amenities", StringType(), True),
    StructField("price", StringType(), True),
    StructField("minimum_nights", IntegerType(), True),
    StructField("maximum_nights", IntegerType(), True),
    StructField("minimum_minimum_nights", IntegerType(), True),
    StructField("maximum_minimum_nights", IntegerType(), True),
    StructField("minimum_maximum_nights", IntegerType(), True),
    StructField("maximum_maximum_nights", IntegerType(), True),
    StructField("minimum_nights_avg_ntm", DoubleType(), True),
    StructField("maximum_nights_avg_ntm", DoubleType(), True),
    StructField("calendar_updated", StringType(), True),
    StructField("has_availability", StringType(), True),
    StructField("availability_30", IntegerType(), True),
    StructField("availability_60", IntegerType(), True),
    StructField("availability_90", IntegerType(), True),
    StructField("availability_365", IntegerType(), True),
    StructField("calendar_last_scraped", StringType(), True),
    StructField("number_of_reviews", IntegerType(), True),
    StructField("number_of_reviews_ltm", IntegerType(), True),
    StructField("number_of_reviews_l30d", IntegerType(), True),
    StructField("first_review", StringType(), True),
    StructField("last_review", StringType(), True),
    StructField("review_scores_rating", DoubleType(), True),
    StructField("review_scores_accuracy", DoubleType(), True),
    StructField("review_scores_cleanliness", DoubleType(), True),
    StructField("review_scores_checkin", DoubleType(), True),
    StructField("review_scores_communication", DoubleType(), True),
    StructField("review_scores_location", DoubleType(), True),
    StructField("review_scores_value", DoubleType(), True),
    StructField("license", StringType(), True),
    StructField("instant_bookable", StringType(), True),
    StructField("calculated_host_listings_count", IntegerType(), True),
    StructField("calculated_host_listings_count_entire_homes", IntegerType(), True),
    StructField("calculated_host_listings_count_private_rooms", IntegerType(), True),
    StructField("calculated_host_listings_count_shared_rooms", IntegerType(), True),
    StructField("reviews_per_month", DoubleType(), True),
    StructField("city", StringType(), True)
])

    data_c = spark.read.format("s3selectJSON").schema(airbnb_schema).load(input1).repartition(50)

    data = data_c.select(
    "city", "description", "latitude", "longitude", "neighborhood_overview", "host_since",
    "host_response_time", "host_response_rate", "host_acceptance_rate",
    "host_is_superhost", "host_total_listings_count",
    "host_has_profile_pic", "host_identity_verified", "property_type",
    "room_type", "accommodates", "bathrooms_text", "bedrooms", "beds",
    "amenities", "price", "number_of_reviews", "first_review", "last_review",
    "instant_bookable",
    "calculated_host_listings_count", "reviews_per_month", "review_scores_rating"
)
    # Read the city coordinates file into a DataFrame (replace 'worldcities.csv' with the actual file path)

    # Read the city coordinates file into a DataFrame
    city_schema = StructType([
    StructField("city", StringType(), True),
    StructField("city_ascii", StringType(), True),
    StructField("lat", DoubleType(), True),
    StructField("lng", DoubleType(), True),
    StructField("country", StringType(), True),
    StructField("iso2", StringType(), True),
    StructField("iso3", StringType(), True),
    StructField("admin_name", StringType(), True),
    StructField("capital", StringType(), True),
    StructField("population", IntegerType(), True),
    StructField("id", StringType(), True)
])
    city_coordinates_df = spark.read.format("s3selectCSV").schema(city_schema).load(input2)

    # Join the two DataFrames based on the 'city' column
    data = data.join(city_coordinates_df.select('city', 'lat', 'lng'), on='city', how='left')

    # Rename columns for clarity
    data = data.withColumnRenamed('lat', 'city_latitude').withColumnRenamed('lng', 'city_longitude')

    #data.printSchema()

    data = data.withColumn(
    "distance_to_city_center_km",
    haversine_distance(col("latitude"), col("longitude"), col("city_latitude"), col("city_longitude"))
).drop("latitude", "longitude", "city_latitude", "city_longitude")

    data = data.withColumn('desc_length', length(data['description']).cast(IntegerType()))
    data = data.withColumn('neighborhood_overview_length', length(data['neighborhood_overview']).cast(IntegerType()))
    data = data.drop('description', 'neighborhood_overview')

    data = data.withColumn('host_since', to_timestamp(data['host_since'], 'yyyy-MM-dd'))

    current_date_col = lit(datetime.now().date())
    data = data.withColumn('current_date', current_date_col)

    data = data.withColumn('host_period_days', datediff(data['current_date'], data['host_since']))
    data = data.withColumn('host_years', (datediff(data['current_date'], data['host_since']) / 365).cast(IntegerType()))
    data = data.drop('host_since', 'current_date')

    data = data.withColumn('num_bathrooms', regexp_extract(data['bathrooms_text'], r'(\d*\.?\d+)', 1).cast('float'))
    data = data.withColumn('is_private_bath', when(data['bathrooms_text'].contains('private'), 1).otherwise(0))
    data = data.withColumn('is_shared_bath', when(data['bathrooms_text'].contains('shared'), 1).otherwise(0))
    data = data.fillna(1, subset=['num_bathrooms'])
    data = data.drop('bathrooms_text')

    data = data.withColumn('is_entire_home', when(col('property_type').contains('Entire home'), 1).otherwise(0))
    data = data.withColumn('is_private_room', when(col('property_type').contains('Private room'), 1).otherwise(0))
    data = data.withColumn('is_shared_room', when(col('property_type').contains('Shared room'), 1).otherwise(0))
    data = data.withColumn('property_type_count', count('property_type').over(Window.partitionBy('property_type')))

    columns_to_drop = ['property_type', 'room_type', 'amenities','id','city']
    data = data.drop(*columns_to_drop)

    # Create binary columns for each response time category
    data = (data
    .withColumn('is_response_time_N/A', when(col('host_response_time') == 'N/A', 1).otherwise(0))
    .withColumn('is_response_time_within_an_hour', when(col('host_response_time') == 'within an hour', 1).otherwise(0))
    .withColumn('is_response_time_within_a_few_hours', when(col('host_response_time') == 'within a few hours', 1).otherwise(0))
    .withColumn('is_response_time_within_a_day', when(col('host_response_time') == 'within a day', 1).otherwise(0))
    .withColumn('is_response_time_within_a_few_days_or_more', when(col('host_response_time') == 'a few days or more', 1).otherwise(0))
    .drop('host_response_time')
)
    # Create binary columns for host response rate and acceptance rate in data
    data = (data
    .withColumn('is_response_rate_N/A', when(col('host_response_rate') == 'N/A', 1).otherwise(0))
    .withColumn('is_high_response_rate', when(col('host_response_rate').cast('float') > 90, 1).otherwise(0))
    .drop('host_response_rate')
)

    # Convert acceptance rate columns
    data = (data
    .withColumn('is_acceptance_rate_N/A', when(col('host_acceptance_rate') == 'N/A', 1).otherwise(0))
    .withColumn('is_high_acceptance_rate', when(col('host_acceptance_rate').cast('float') > 90, 1).otherwise(0))
    .drop('host_acceptance_rate')
)
    # Convert 't' to True and 'f' to False
    data = data.withColumn('host_has_profile_pic', when(col('host_has_profile_pic') == 't', True).otherwise(False))

    # Convert 't' to True and 'f' to False
    data = data.withColumn('host_identity_verified', when(col('host_identity_verified') == 't', True).otherwise(False))

    # Convert 't' to True and 'f' to False
    data = data.withColumn('host_is_superhost', when(col('host_is_superhost') == 't', True).otherwise(False))

    # Convert 't' to True and 'f' to False
    data = data.withColumn('instant_bookable', when(col('instant_bookable') == 't', True).otherwise(False))

    # Handle blank values (assuming blank means False)
    data = data.na.fill({'host_has_profile_pic': False, 'host_identity_verified': False, 'host_is_superhost': False, 'instant_bookable': False})

    data = data.withColumn('first_review', col('first_review').cast('date'))
    data = data.withColumn('last_review', col('last_review').cast('date'))

    # Calculate days since last review
    data = data.withColumn('days_since_last_review', datediff(current_date(), col('last_review')))
    # Replace null values in 'days_since_last_review' with 0
    data = data.na.fill({'days_since_last_review': 0})

    # Drop columns
    data = data.drop('first_review', 'last_review')

    # Convert 'price' column to PySpark StringType, remove '$' and ',' and convert to PySpark FloatType
    data = data.withColumn('price', col('price').cast('string'))
    data = data.withColumn('price', regexp_replace('price', '\$', ''))
    data = data.withColumn('price', regexp_replace('price', ',', '').cast('float'))

    # Fill null values with 0
    data = data.na.fill(0)

   # Assemble features into a vector
    feature_columns = [col for col in data.columns if col != 'price']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")

    # Standardize the features
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
   # Assemble features into a vector
    feature_columns = [col for col in data.columns if col != 'price']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")

    # Standardize the features
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)

    # Random Forest Regressor
    rf = RandomForestRegressor(labelCol="price", featuresCol="scaled_features")
    pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
    model_rf = pipeline_rf.fit(data)
    rf_model = model_rf.stages[-1]

    input_feature_names = assembler.getInputCols()
    feature_importances = rf_model.featureImportances

    # Create a list of Row objects with feature names and their importances
    rows = [Row(Feature_Name=input_feature_names[i], Importance=round(float(importance),4)) for i, importance in enumerate(feature_importances)]

    # Create a Spark DataFrame from the list of Row objects

    feature_importances_df = spark.createDataFrame(rows)
    feature_importances_df = feature_importances_df.orderBy('Importance', ascending=False)

    # Show the DataFrame
    print("Feature Importance Price:")
    feature_importances_df.show()
    feature_importances_df.write.csv(output1, mode='overwrite', header=True)

    # Assemble features into a vector
    feature_columns = [col for col in data.columns if col != 'review_scores_rating']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")

    # Standardize the features
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
   # Assemble features into a vector
    feature_columns = [col for col in data.columns if col != 'review_scores_rating']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")

    # Standardize the features
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)

    # Random Forest Regressor
    rf = RandomForestRegressor(labelCol="review_scores_rating", featuresCol="scaled_features")
    pipeline_rf = Pipeline(stages=[assembler, scaler, rf])
    model_rf = pipeline_rf.fit(data)
    rf_model = model_rf.stages[-1]

    input_feature_names = assembler.getInputCols()
    feature_importances = rf_model.featureImportances

    # Create a list of Row objects with feature names and their importances
    rows = [Row(Feature_Name=input_feature_names[i], Importance=round(float(importance),4)) for i, importance in enumerate(feature_importances)]

    # Create a Spark DataFrame from the list of Row objects

    feature_importances_df = spark.createDataFrame(rows)
    feature_importances_df = feature_importances_df.orderBy('Importance', ascending=False)

    # Show the DataFrame
    print("Feature Importance Ratings:")
    feature_importances_df.show()
    feature_importances_df.write.csv(output2, mode='overwrite', header=True)

    ###################################################################################################################

    # Calculate correlations
    correlation_price_distance = compute_correlation(data, "price", "distance_to_city_center_km")
    correlation_review_distance = compute_correlation(data, "review_scores_rating", "distance_to_city_center_km")

    # Store all output in a list
    output = [correlation_price_distance, correlation_review_distance]

    # Save the output to a text file
    output_rdd = spark.sparkContext.parallelize(output).coalesce(1)
    output_rdd.saveAsTextFile(output3)

if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    output1 = sys.argv[3]
    output2 = sys.argv[4]
    output3 = sys.argv[5]

    spark = SparkSession.builder.appName('airbnb feature analysis').getOrCreate()
    assert spark.version >= '3.0'
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    main(input1, input2, output1, output2, output3)
