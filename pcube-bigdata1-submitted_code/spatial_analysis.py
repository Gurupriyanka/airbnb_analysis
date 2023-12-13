import os
from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys
assert sys.version_info >= (3, 5)
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, split, lit, col, udf, regexp_extract, length, current_date, datediff, year, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType, TimestampType
from datetime import datetime
from pyspark.sql.functions import when, count, sum, expr, regexp_replace, radians, sin, cos, atan2, sqrt
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
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

def main(input1, input2, input3, output):

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
  
    data = data_c.select("id","city", "latitude", "longitude","price","review_scores_rating")
    
     # Convert 'price' column to PySpark StringType, remove '$' and ',' and convert to PySpark FloatType
    data = data.withColumn('price', col('price').cast('string'))
    data = data.withColumn('price', regexp_replace('price', '\$', ''))
    data = data.withColumn('price', regexp_replace('price', ',', '').cast('float'))

    # Fill null values with 0
    data_prep = data.na.fill(0)
    data = data.na.fill(0)

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

    data = data.withColumn(
    "distance_to_city_center_km",
    haversine_distance(col("latitude"), col("longitude"), col("city_latitude"), col("city_longitude"))
).drop("latitude", "longitude", "city_latitude", "city_longitude")

     #Visualize how distance to city center affects the price of the listing
    data = data.withColumn("distance_to_city_center_km", data["distance_to_city_center_km"].cast(FloatType()))
    data = data.withColumn("price", data["price"].cast(FloatType()))
    
    #Compute the correlation between the independent and dependent variables
    correlation_price = data.stat.corr("price", "distance_to_city_center_km")
    print(f"Correlation between city center and price of listing location: {correlation_price}")

    #Visualize how distance to city center affects the review_scores_rating of the listing
    data = data.withColumn("review_scores_rating", data["review_scores_rating"].cast(FloatType()))

    # #Compute the correlation between the independent and dependent variables
    correlation_ratings = data.stat.corr("review_scores_rating", "distance_to_city_center_km")
    print(f"Correlation between city center and review_scores_rating of listing location: {correlation_ratings}")

    correlation_ratings_price = data.stat.corr("price", "review_scores_rating")
    print(f"Correlation between city center and price and review scores: {correlation_ratings_price}")

    colleges_schema = StructType([
    StructField("college_name", StringType(), True),
    StructField("province", StringType(), True),
    StructField("city", StringType(), True),
    StructField("col_latitude", FloatType(), True),
    StructField("col_lng", FloatType(), True),
    StructField("country", StringType(), True)
])

    # Read college data from a JSON file and cache the DataFrame
    college_df = spark.read.json(input3, schema=colleges_schema)

    college_df = college_df.na.drop(subset=['col_latitude','col_lng'])

    # Perform the cross join
    result_df = data_prep.join(college_df.hint('broadcast'), on = ['city'], how = "left_outer")
    result_df = result_df.filter(col("city").isNotNull())

    # Calculate distance using Haversine formula
    result_df = result_df.withColumn(
    "distance_to_colleges_km",
    haversine_distance(col("latitude"), col("longitude"), col("col_latitude"), col("col_lng"))
).drop("latitude", "longitude", "col_latitude", "col_lng")

    window_spec = Window.partitionBy("id")

  # Count the number of colleges within 25km using the window function
    result = result_df \
    .withColumn("num_colleges_within_25km", sum((col("distance_to_colleges_km") <= 25).cast(IntegerType())).over(window_spec))
    #Visualize how distance to city center affects the price of the listing
    result = result.withColumn("distance_to_colleges_km", result["distance_to_colleges_km"].cast(FloatType()))
    result = result.withColumn("price", result["price"].cast(FloatType()))

    #Compute the correlation between the independent and dependent variables
    # Calculate correlations
    correlation_price_coll = result.stat.corr("price", "distance_to_colleges_km")
    correlation_ratings_coll = result.stat.corr("review_scores_rating", "distance_to_colleges_km")
    correlation_price_num_coll = result.stat.corr("price", "num_colleges_within_25km")
    correlation_ratings_num_coll = result.stat.corr("review_scores_rating", "num_colleges_within_25km")

    # Print statements
    print(f"Correlation between distance of colleges and price of listing location: {correlation_price_coll}")
    print(f"Correlation between distance of colleges and review_scores_rating of listing location: {correlation_ratings_coll}")
    print(f"Correlation between no. of colleges nearby and price of listing location: {correlation_price_num_coll}")
    print(f"Correlation between no. of colleges nearby and review_scores_rating of listing location: {correlation_ratings_num_coll}")

    # Create an RDD with all print statements
    output_rdd = spark.sparkContext.parallelize([
        f"Correlation between distance of colleges and price of listing location: {correlation_price_coll}",
        f"Correlation between distance of colleges and review_scores_rating of listing location: {correlation_ratings_coll}",
        f"Correlation between no. of colleges nearby and price of listing location: {correlation_price_num_coll}",
        f"Correlation between no. of colleges nearby and review_scores_rating of listing location: {correlation_ratings_num_coll}"
    ]).coalesce(1)

    # Save the RDD content to a single text file
    output_rdd.saveAsTextFile(output)

if __name__ == '__main__':
    input1 = sys.argv[1]
    input2 = sys.argv[2]
    input3 = sys.argv[3]
    output = sys.argv[4]

    spark = SparkSession.builder.appName('airbnb spatial analysis').getOrCreate()
    assert spark.version >= '3.0'
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext

    main(input1, input2, input3, output)
