
from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys
assert sys.version_info >= (3, 5)
from pyspark.sql import SparkSession
from pyspark.sql.window import Window as W
from pyspark.sql.functions import input_file_name, split, lit, col, udf, regexp_extract,when, col, months_between, current_date
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType, TimestampType
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
from pyspark.sql.functions import col, expr
from pyspark.sql import functions as F
from pyspark.sql.functions import col, format_number, avg, when, min, max
from pyspark.sql.types import FloatType, ArrayType, LongType


def main(inputs, input2):
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
    StructField("city", StringType(), True)])

  # Create a Spark session
  df = spark.read.json(inputs, schema=airbnb_schema).repartition(50)  # df.show(5)
  df = df.dropna(how='all')
  selected_columns = [
      "id",
      "listing_url",
      "name",
      "description",
      "city",
      "neighborhood_overview",
      "picture_url",
      "host_id",
      "host_url",
      "host_name",
      "host_since",
      "host_location",
      "host_about",
      "host_response_time",
      "host_response_rate",
      "host_acceptance_rate",
      "host_is_superhost",
      "host_thumbnail_url",
      "host_picture_url",
      "host_neighbourhood",
      "host_listings_count",
      "host_total_listings_count",
      "host_verifications",
      "host_has_profile_pic",
      "host_identity_verified",
      "neighbourhood",
      "neighbourhood_cleansed",
      "neighbourhood_group_cleansed",
      "latitude",
      "longitude",
      "property_type",
      "room_type",
      "accommodates",
      "bathrooms",
      "bathrooms_text",
      "bedrooms",
      "beds",
      "amenities",
      "price",
      "minimum_nights",
      "maximum_nights",
      "has_availability",
      "availability_30",
      "availability_60",
      "availability_90",
      "availability_365",
      "number_of_reviews",
      "number_of_reviews_ltm",
      "number_of_reviews_l30d",
      "first_review",
      "last_review",
      "review_scores_rating",
      "review_scores_accuracy",
      "review_scores_cleanliness",
      "review_scores_checkin",
      "review_scores_communication",
      "review_scores_location",
      "review_scores_value",
      "license",
      "instant_bookable",
      "calculated_host_listings_count",
      "calculated_host_listings_count_entire_homes",
      "calculated_host_listings_count_private_rooms",
      "calculated_host_listings_count_shared_rooms",
      "reviews_per_month"
  ]
  df = df.drop("_corrupt_record")

  df = df.select(*selected_columns)
  cleaned_df = df.dropna(how="all", subset=selected_columns)
  # cleaned_df.show(5)

  """##### Price Parsing Transformation"""

  # df.filter(col("price") < 0).show()
  cleaned_df = cleaned_df.withColumn("price", regexp_replace(col("price"), "\\$", "").cast("double"))
  cleaned_df = cleaned_df.filter(col("price").isNotNull() & (col("price") >= 0))
  # cleaned_df.select("price").show(10)

  """##### Feature Engineering
  ##### This tranformation creates new columns based on existing features such as  1. average review score 2.ratio of minimum nights to maximum nights
  """

  feature_engg_df = cleaned_df.withColumn("ratio_min_max_nights", col("minimum_nights") / col("maximum_nights"))
  feature_engg_df = feature_engg_df.withColumn("ratio_min_max_nights", F.round(feature_engg_df["ratio_min_max_nights"], 5))
  feature_engg_df = feature_engg_df.withColumn("average_review_score", F.round(expr("(review_scores_rating + review_scores_accuracy + review_scores_cleanliness + review_scores_checkin + review_scores_communication + review_scores_location + review_scores_value) / 7"), 5))

  feature_engg_df = feature_engg_df.filter(col("ratio_min_max_nights").isNotNull() & col("average_review_score").isNotNull())

  # feature_engg_df.select("average_review_score","minimum_nights","maximum_nights","ratio_min_max_nights").show(5)

  """##### Data Filtering

  #### Filter out rows with missing data or based on specific criteria

  ###### 1.remove columns AR-AW
  ###### 2.List Airbnbs without license.
  ###### 3. Create a new Expires License column
  ###### 4.remove columns without Description and without city names.
  """

  columns_to_drop = [
      "minimum_minimum_nights",
      "maximum_minimum_nights",
      "minimum_maximum_nights",
      "maximum_maximum_nights",
      "minimum_nights_avg_ntm",
      "maximum_nights_avg_ntm"
  ]
  dropped_col_df = feature_engg_df.drop(*columns_to_drop)
  filter_df = dropped_col_df.filter(col("description").isNotNull() & col("city").isNotNull()).cache()
  filter_df = filter_df.withColumn(
      'license_expiry_date',
      when(col('license').isNotNull(),
          regexp_extract(col('license'), r'expires: (\d{4}-\d{2}-\d{2})', 1)
      )
  )
  filter_df = filter_df.withColumn(
      'months_until_expiry',
      months_between('license_expiry_date', current_date())
  )
  filter_df = filter_df.withColumn(
      'License_Expiry_Status',
      when(col('months_until_expiry') <= 6, 'Expires in less than 6 months')
      .when(col('months_until_expiry') <= 12, 'Expires in less than 1 year')
      .when(col('months_until_expiry') <= 24, 'Expires in less than 2 years')
      .otherwise('Status Not Available')
  )
  filter_df = filter_df.withColumn(
      'License_Number',
      when(col('license').isNotNull(),
          regexp_extract(col('license'), r'^([^,]+)', 1)
      )
  )

  filter_df = filter_df.withColumn(
      'License Present?',
      when(col('license').isNotNull(), 'Yes').otherwise('No')
  )


  """##### Calculate the average AIRBNB price for each city"""

  # Filter out rows where the "price" column is null
  filter_df = filter_df.filter(col("price").isNotNull())

  # Group by the "city" column and calculate the average price
  grouped_df = filter_df.groupBy("city").agg(F.format_number(expr("avg(price)"), 2).alias("Average Price in $"))
  # grouped_df.show()

  """##### Property Type Analysis
  ##### Investigate the popularity of different property types among guests.
  ###### did not include room type as it was not adding value to our result"""


  result_property_type_analysis_df = filter_df.groupBy("property_type").agg(
      F.count("property_type").alias("Listing Count for each property"),
      F.format_number(expr("avg(price)"), 2).alias("Average price of each property type in $")
  )

  result_most_popular_property_type_df = result_property_type_analysis_df.orderBy(F.desc("Listing Count for each property")) \
      .select("property_type") \
      .first()
  most_popular_property_type_value = result_most_popular_property_type_df["property_type"]  # calculates the most popular airbnb type 

  # Print the result
  #print(f"The most popular property type among guests is: {most_popular_property_type_value}")

  """##### Determine which city has the most popular types of Airbnb property
  ###### the one having maximum booking count
  """

  # Group by "city", "property_type", and "room_type" and calculate booking counts
  group_city_df = filter_df.groupBy("city", "property_type").agg(F.count("property_type").alias("Listing Count for each city"))

  # Find the most popular property_type for each city
  window_spec = W.partitionBy("city").orderBy(F.desc("Listing Count for each city"))
  most_popular_city_property_type_df = group_city_df.withColumn("rank", F.rank().over(window_spec))
  most_popular_city_property_type_df = most_popular_city_property_type_df.filter(col("rank") == 1)

  # Select relevant columns
  result_most_popular_city_property_type_df = most_popular_city_property_type_df.select(
      "city",
      col("property_type").alias("Popular Property"),
      "Listing Count for each city"
  )
  # Show the result
  result_most_popular_city_property_type_df = result_most_popular_city_property_type_df.orderBy(F.desc("Listing Count for each city"))
  # result_most_popular_city_property_type_df.show(truncate=False)

  """##### Listings blocked by Host
  # How many listings have been blocked by the host
  ##### If "has availability" if true but has 0's in any/all 4(30/60/90/365) columns that means it has been blocked by host for that any/all days.
  ##### If "has availability" if false that means it has been blocked by guest or hosts.
  """

  def availability_message(has_availability, availability_365, availability_90, availability_60, availability_30):
      if has_availability == "f":
          return "Not available"
      elif has_availability == "t" and availability_365 == 0:
          return "Not available for next 365 days"
      elif has_availability == "t" and availability_90 == 0:
          return "Not available for next 90 days"
      elif has_availability == "t" and availability_60 == 0:
          return "Not available for next 60 days"
      elif has_availability == "t" and availability_30 == 0:
          return "Not available for next 30 days"
      else:
          return "Available"


  # Define the UDF
  availability_message_udf = udf(availability_message, StringType())

  # Apply the UDF to create a new column with availability message
  filter_df = filter_df.withColumn("availability_message", availability_message_udf(
      col("has_availability"), col("availability_365"), col("availability_90"), col("availability_60"), col("availability_30")
  ))

  # Select only the required columns
  result_df = filter_df.select("id", col("has_availability"), col("availability_365"), col("availability_90"), col("availability_60"), col("availability_30"),"availability_message")
  # result_df.show(truncate=False)

  """##### Create a new category column stating  if host has responded within an hour means "Highly responsive host", if they respond within few hours or within a day means then  "Respons by the EOD" and if they respond in few days or more then "Least Responsive" else "Data Not Available"

  """

  # Create a new category column based on host_response_time and host_response_rate
  filter_df = filter_df.withColumn(
      "response_category",
      when(col("host_response_time") == "within an hour", "Highly Responsive host")
      .when((col("host_response_time") == "within a few hours") | (col("host_response_time") == "within a day"), "Typically Responds by the EOD")
      .when(col("host_response_time") == "a few days or more", "Least Responsive Host")
      .otherwise("Data Not Available")
  )

  # Show the resulting DataFrame with the new category column
  # filter_df.select("id","host_response_time", "host_response_rate", "response_category").show(truncate=False)

  """#### To find the top 5 superhosts in each city with a 100% acceptance rate and who respond within an hour"""

  # Define the window specification
  window_spec = W.partitionBy("City").orderBy(col("host_response_rate").desc(), col("host_response_time"))

  # Filter superhosts with 100% acceptance rate and response within an hour
  superhosts_df = filter_df.filter(
      (col("host_is_superhost") == "t") &
      (col("host_response_rate") == "100%") &
      (col("host_response_time") == "within an hour")
  )

  # Add a rank column to identify the top 5 superhosts in each city
  result_df = superhosts_df.withColumn("rank", F.rank().over(window_spec)).filter(col("rank") <= 5)

  # Show the result
  # result_df.select("city", "host_id", "host_name", "host_response_rate", "host_response_time", "rank").show(truncate=False)

  """#### Add a new rating category column based on reviews - “Excellent, Very Good, Average..”"""

  conditions = [
      (col("review_scores_rating") >= 4.8, "Excellent"),
      (col("review_scores_rating") >= 4.0, "Very Good"),
      (col("review_scores_rating") >= 3.0, "Good"),
      (col("review_scores_rating") >= 2.0, "Average"),
      (col("review_scores_rating") >= 1.0, "Below Average"),
  ]

  # Add a new column 'rating_category' based on conditions
  filter_df = filter_df.withColumn(
      "Rating_category",
      when(col("review_scores_rating").isNotNull(),
          when(col("review_scores_rating") == 5.0, "Perfect")
          .when(col("review_scores_rating") > 4.8, "Excellent")
          .when(col("review_scores_rating") > 4.0, "Very Good")
          .when(col("review_scores_rating") > 3.0, "Good")
          .when(col("review_scores_rating") > 2.0, "Average")
          .when(col("review_scores_rating") > 1.0, "Below Average")
          .otherwise("No Rating")
      ).otherwise("No Rating")
  )

  # Show the DataFrame with the new 'Rating_category' column
  # result_rating_category_df.select("id","review_scores_rating", "Rating_category").show(truncate=False)

  """#### List how old/recent each review is"""

  # Get the current year
  current_year = F.year(F.current_date())

  # Define conditions for review categories
  conditions = [
      (col("last_review_year") >= current_year - 2, "Recently reviewed"),
      (col("last_review_year") >= current_year - 5, "Not Recently reviewed"),
      (col("last_review_year") < current_year - 5, "Oldest reviews"),
  ]
  filter_df = filter_df.withColumn("last_review_year", F.year("last_review"))

  # Add a new column 'review_category' based on conditions
  filter_df = filter_df.withColumn(
      "review_category",
      when(
          col("last_review").isNotNull(),
          when(
              col("last_review").isNotNull(),
              when(
                  col("last_review_year") >= current_year - 2,
                  "Recently reviewed"
              )
              .when(
                  col("last_review_year") >= current_year - 5,
                  "Not Recently reviewed"
              )
              .when(
                  col("last_review_year") < current_year - 5,
                  "Oldest reviews"
              )
              .otherwise("Not recently reviewed")
          ).otherwise("Not recently reviewed")
      ).otherwise("No reviews available")
  )

  # Filter records where 'last_review' is NOT NULL
  result_review_df = filter_df.filter(col("last_review").isNotNull())

  # Show the DataFrame with the new 'review_category' column and filtered records
  # result_review_df.select("last_review", "last_review_year", "review_category").show(truncate=False)

  """#### Host Profile completeness (Y/N)?
  #####If all the details of the host are available, then the profile is complete else it is not
  """

  # Define conditions for checking completeness
  profile_completeness_conditions = [
      col("host_verifications").isNotNull(),
      col("host_about").isNotNull(),
      col("host_has_profile_pic").isNotNull(),
      col("host_identity_verified").isNotNull(),
      col("host_thumbnail_url").isNotNull(),
      col("host_picture_url").isNotNull(),
      col("host_neighbourhood").isNotNull(),
      col("host_listings_count") != "0",
  ]

  # Calculate completeness score (number of non-null/blank fields)
  completeness_score = sum(when(condition, 1).otherwise(0) for condition in profile_completeness_conditions)

  # Add a new column 'host_profile_completeness' based on completeness score
  filter_df = filter_df.withColumn(
      "Host Profile Completeness",
      when(completeness_score == len(profile_completeness_conditions), "Yes").otherwise("No")
  )

  # Show the DataFrame with the new 'host_profile_completeness' column
  # filter_df.select("id","host_name","Host Profile Completeness").show(truncate=False)

  """#### Analyze description column
  ##### Categorize the listing into wheelchair accessible, Kids/Family Friendly, Bachelor Friendly
  """

  # Define keywords for special categories
  wheelchair_keywords = ["wheelchair", "accessible", "accessibility", "mobility", "disabled", "ramp", "elevator", "handicap", "barrier-free", "ADA compliant", "step-free", "inclusive", "ramp access", "easy access", "special needs"]
  kids_Fam_keywords = ["kids", "children", "family", "family-friendly", "child-friendly", "playground", "toys", "safe", "baby", "childcare", "kid-friendly", "games", "children's books", "high chair", "baby monitor"]
  bachelor_keywords = ["bachelor", "single", "solo", "individual", "private", "personal", "solitude", "alone", "independent", "unattached", "self-contained", "self-sufficient", "individual retreat", "quiet", "introvert"]

  # Create new categorical columns based on keyword presence
  filter_df = filter_df.withColumn(
      "Listing_special_category",
      when(
          F.lower(col("description")).rlike("|".join(wheelchair_keywords)),
          "wheelchair_accessible"
      ).when(
          F.lower(col("description")).rlike("|".join(kids_Fam_keywords)),
          "kids/Family Friendly"
      ).when(
          F.lower(col("description")).rlike("|".join(bachelor_keywords)),
          "Bachelor Friendly"
      ).otherwise("not_specified")
  )

  # Show the DataFrame with the new 'special_category' column
  # filter_df.select("description", "Listing_special_category").show(truncate = True)

  """#### Take 3Q from the price range of each city and categorize as Create price categories (e.g., budget, mid-range, luxury) and analyze how review scores vary among these categories.
  ###### At the end of this code, also calculated the average review score based on the city and price category. For eg. Vancouver's luxury airbnbs have comparatively high reviews than Vancouver's budget airbnbs
  """

  # Define a window specification based on the 'neighbourhood' column
  price_window = W().partitionBy("City").orderBy("price")

  # Add new columns for min and max prices in each City
  filter_df = filter_df.withColumn("min_price", min("price").over(price_window))
  filter_df = filter_df.withColumn("max_price", max("price").over(price_window))

  # Add a new column 'price_category' based on the specified price ranges within each City
  filter_df = filter_df.withColumn(
      "price_category",
      when(col("price") <= col("min_price") + (col("max_price") - col("min_price")) / 3, "Budget")
      .when((col("price") > col("min_price") + (col("max_price") - col("min_price")) / 3) & (col("price") <= col("min_price") + 2 * (col("max_price") - col("min_price")) / 3), "Mid-range")
      .when(col("price") > col("min_price") + 2 * (col("max_price") - col("min_price")) / 3, "Luxury")
      .otherwise("Unknown")
  )

  # Show the DataFrame with the new columns and average review scores
  # filter_df.select("id", "City", "price", "price_category", "review_scores_rating").show()

  """#### Calculate the price_category of each city"""

  # Group by 'City' and 'price_category' and calculate average review scores
  average_review_scores_df = filter_df.groupBy("City", "price_category").agg(
      format_number(F.avg("review_scores_rating"), 2).alias("average_review_score")
  )
  # average_review_scores_df.show()

  """####  Listing Completeness: if all the fields of a particular listing are avialable then the listing is said to be complete"""

    # Define the columns to check for completeness
  required_columns = [
      "name",
      "description",
      "picture_url",
      "host_id",
      "host_name",
      "host_since",
      "host_location",
      "host_about",
      "host_thumbnail_url",
      "host_picture_url",
      "host_neighbourhood",
      "host_listings_count",
      "host_total_listings_count",
      "host_verifications",
  ]

  # Create a list of conditions for completeness
  completeness_conditions = [col(col_name).isNotNull() for col_name in required_columns]

  # Calculate completeness score (number of non-null/blank fields)
  completeness_score = sum(when(condition, 1).otherwise(0) for condition in completeness_conditions)

  # Add a new column 'listing_completeness' based on completeness score
  filter_df = filter_df.withColumn(
      "Listing_Completeness",
      when(completeness_score == len(completeness_conditions), "Yes").otherwise("No")
  )

  # Show the DataFrame with the new 'listing_completeness' column
  # filter_df.select("id", "listing_completeness").show(truncate=False)

  """#### Generate .CSV file from the result"""

#   selected_columns = [
#       "id", "listing_url", "city", "price", "latitude", "longitude", "property_type",
#       "has_availability", "availability_30", "availability_60", "availability_90", "availability_365",
#       "License Present?", "License_Number", "License_Expiry_Status", "availability_message",
#       "response_category", "Host Profile Completeness", "Listing_special_category",
#       "price_category", "Listing_Completeness"
#   ]

  filter_df = filter_df.select(
      "id", "listing_url", "city", "price", "latitude", "longitude", "property_type",
      "Rating_category","review_scores_rating","review_category","has_availability", "availability_30", "availability_60", "availability_90", "availability_365",
      "License Present?", "License_Expiry_Status", "availability_message",
      "response_category","host_is_superhost","host_response_rate", "host_response_time","Host Profile Completeness", "Listing_special_category",
      "price_category", "Listing_Completeness"
  )
  filter_df.write.csv(output,mode='overwrite', header = True)

#   filter_df.write.json(output, compression='gzip', mode='overwrite')

if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    #in_try = pd.read_csv(input1)
    spark = SparkSession.builder.appName('Airbnb Analysis').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')  # Set log level on SparkContext


    main(inputs, output)

