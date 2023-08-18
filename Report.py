# Databricks notebook source
# MAGIC %md
# MAGIC ##### Libraries

# COMMAND ----------

# Load functions
from pyspark.sql.functions import *
from pyspark.sql.functions import col,isnan,when,count
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
# from pyspark.ml import Pipelinefrom pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Import Data

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/"
file_type = "csv"

# Importing all the tables 
circuits = spark.read.csv(file_location + "circuits.csv", header=True, inferSchema=True)
constructorResults = spark.read.csv(file_location + "constructorResults.csv", header=True, inferSchema=True)
constructors = spark.read.csv(file_location + "constructors.csv", header=True, inferSchema=True)
drivers = spark.read.csv(file_location + "drivers.csv", header=True, inferSchema=True)
lapTimes = spark.read.csv(file_location + "lapTimes.csv", header=True, inferSchema=True)
pitStops = spark.read.csv(file_location + "pitStops.csv", header=True, inferSchema=True)
races = spark.read.csv(file_location + "races.csv", header=True, inferSchema=True)
results = spark.read.csv(file_location + "results.csv", header=True, inferSchema=True)
status = spark.read.csv(file_location + "status.csv", header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Data Cleaning and Pre-processing

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1 Table: Circuits

# COMMAND ----------

# Inspecting "circuits" dataframe
circuits.show(5, truncate = False)
circuits.describe().show()
circuits.printSchema()

# COMMAND ----------

# Renaming circuit name to remove ambiguity 
circuits = circuits.withColumnRenamed("name", "circuit_name")

# COMMAND ----------

# Dropping latitude and longitude columns
circuits = circuits.drop("lat","lng")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2 Table: Constructors

# COMMAND ----------

# Inspecting "constructors" dataframe
constructors.show(5, truncate = False)
constructors.describe().show()
constructors.printSchema()

# COMMAND ----------

# Renaming name and nationality columns
constructors = constructors.withColumnRenamed("name", "constructor_name") \
                            .withColumnRenamed("nationality", "constructor_nationality")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3 Table: Constructor Results

# COMMAND ----------

# Inspecting "constructorResults" dataframe
constructorResults.show(5, truncate = False)
constructorResults.describe().show()
constructorResults.printSchema()

# COMMAND ----------

# 'Double' data type occupies more storage space( bytes). So, converung 'points' column to float data type ( occupies 32 bytes)
constructorResults = constructorResults.withColumn("points", col("points").cast("float"))

# COMMAND ----------

# Columns in above dataframes have the correct data types, and there are no null values present. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Table: Drivers

# COMMAND ----------

# Inspecting "drivers" dataframe
drivers.show(5, truncate = False)
drivers.describe().show()
drivers.printSchema()

# COMMAND ----------

# There is a record with empty date of birth, so dropping that record
drivers = drivers.na.drop(subset=["dob"])

# COMMAND ----------

# Combine "forename" and "surname" columns into a single "driver_name" column
drivers = drivers.withColumn("driver_name", concat(col("forename"),lit(" "), col("surname")))

# Dropping 'forename' , surname' 
drivers = drivers.drop("forename", "surname")

# COMMAND ----------

# dob column is present as string data type, so converting it to date type
drivers = drivers.withColumn("dob_date",
                    when(col("dob").contains("-"), to_date(col("dob"), "yyyy-MM-dd")) # Using two formats becasie we have records with differnet data formats
                   .when(col("dob").contains("/"), to_date(col("dob"), "dd/MM/yyyy"))
                   .otherwise(None)  # Handle invalid formats
)

# Calculate age of drivers in the year 2018 and save it an a new dataframe
drivers_age = drivers.withColumn("age", 2018 - year(col("dob_date")))
drivers_age = drivers_age.groupBy("driverId").agg(F.first("age").alias("age"))

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cell, total number of wins are cacluated for total number of races a driver took part in. A higher win rate is an indication of better performance and consistency. This metric will be helpful in understanding driver's skill, comparing performance, evaluating strategies, and making decisions in racing and competitive contexts.

# COMMAND ----------

# Selecting specific columns from the 'drivers', 'races', 'results' dataframes
drivers = drivers.select("driverId", "dob")
race_dates = races.select("raceId", "date")
driver_standings = results.select("raceId", "driverId", "positionOrder")

# Calculating the total number of wins for each driver
window_spec = Window.partitionBy("raceId").orderBy(F.col("positionOrder"))
driver_standings = driver_standings.withColumn("wins_per_race", F.when(F.col("positionOrder") == 1, 1).otherwise(0))
driver_standings = driver_standings.withColumn("totalWins", F.sum(F.when(F.col("wins_per_race") == 1, 1).otherwise(0)).over(Window.partitionBy("driverid")))

# Adding dates to each race
driver_standings = driver_standings.join(race_dates.select("raceId", "date"), "raceId", "left")

# Convert the "date" column to a date object
driver_standings = driver_standings.withColumn("date", F.to_date(driver_standings["date"]))

# Create a new column 'year' to extract the year from the 'date' column
driver_standings = driver_standings.withColumn("year", F.year(driver_standings["date"]))

# Count the number of races each driver has driven in
num_races_per_driver = driver_standings.groupBy("driverId").agg(F.countDistinct("raceId").alias("totalRaces"))

# Adding totalRaces to each driver
drivers = driver_standings.join(num_races_per_driver, "driverId", "left")

# Calculate win rate and drop totalWins column
drivers = drivers.withColumn("winRate", F.round(F.col("totalWins") / F.col("totalRaces"), 3))
drivers = drivers.groupBy("driverId").agg(
                                    F.first("totalWins").alias("totalWins"),
                                    F.first("totalRaces").alias("totalRaces"),
                                    F.first("winRate").alias("winRate")
                                )

# COMMAND ----------

# Merge drivers age dataframe
drivers = drivers_age.join(drivers, "driverId", "left")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.5 Table: Lap Times

# COMMAND ----------

# Inspecting "lapTimes" dataframe
lapTimes.show(5, truncate = False)
lapTimes.describe().show()
lapTimes.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct

lapTimes.agg(*(countDistinct(col(c)).alias(c) for c in lapTimes.columns)).show()

# COMMAND ----------

# Dropping milliseconds column 
lapTimes = lapTimes.drop("milliseconds")

# COMMAND ----------

# Calculate position change for all drivers in each race per lap, a positive vlaue indicates and improvement improvment in position and viceversa
window_spec_driver = Window.partitionBy("driverId").orderBy("lap")
lapTimes = lapTimes.withColumn("position_change",
                    F.when(F.lag("position", 1).over(window_spec_driver).isNull(), -1)
                    .otherwise(F.lag("position", 1).over(window_spec_driver) - F.col("position")))

# COMMAND ----------

# Calculate average position changes and maximum position
lapTimes = lapTimes.groupBy("raceId", "driverId").agg(round(avg("position_change"), 0).alias("avg_position_change"),
                    max("position").alias("max_position"))

# COMMAND ----------

# MAGIC %md
# MAGIC We will create a performance metric "Performance Change Index" to understand driver's performance not only with average number of positions changed but also the direction of the change. This position chnage metrix values ranges from -1 to 1 where
# MAGIC - Positive values indicate an improvement in average position        .
# MAGIC - Negative values denote a decline in average position and 0 indicates no significant change.
# MAGIC       -

# COMMAND ----------

# Calculate position change index metric
lapTimes = lapTimes.withColumn("position_change_index", round((1 - (abs(col("avg_position_change")) / (col("max_position") - 1))) *
        when(col("avg_position_change") > 0, 1).otherwise(-1)))

# COMMAND ----------


# Check count of null vlaues
lapTimes_Columns=['raceId','driverId','avg_position_change','max_position','position_change_index']
lapTimes.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in lapTimes_Columns]).show()


# COMMAND ----------

# We can see that there are null vlaues un position chnage index column, we will check the details of these null vlaues 
lapTimes.filter(lapTimes.position_change_index.isNull()).display()

# COMMAND ----------

# MAGIC %md
# MAGIC From above output we can see that drivers with 'max_position' as 1 have null values for 'position_change_index'.This means that drivers maintained the same position throughout the race (max position is the same as the initial position), this is an indication of good performance. So, replacing tnull vlaues highest performance index metric, which is 1

# COMMAND ----------

# Replacing null values in  "position_change_index" column with 1
lapTimes = lapTimes.fillna({"position_change_index": 1})

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.6 Table: Pit Stops

# COMMAND ----------

# Inspecting "pitStops" dataframe
pitStops.show(5, truncate = False)
pitStops.describe().show()
pitStops.printSchema()

# COMMAND ----------

# Converting the 'duration' column to integer data type  and rename this column
pitStops = pitStops.withColumn("duration", col("duration").cast("int"))
pitStops = pitStops.withColumnRenamed("duration", "pitStop_duration")

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cell, total and average pit stop duration, number of pit stops and average of laps changed between each pit stop (races with one pit stop will have this vlaue as -1) and pit stops efficiency  are calculated. 

# COMMAND ----------

# Group the data frame to calculate above mentioned values
pitStops = pitStops.groupBy('raceId', 'driverId').agg(
                            sum('pitStop_duration').alias('total_pitStop_duration'),   # Calculate the total duration of pit stops for each group
                            round(avg('pitStop_duration'), 0).alias('avg_pitStop_duration'),     # Calculate and round the average duration of pit stops for each group
                            count('lap').alias('total_pitStops'),                      # Count the total number of pit stops for each group
                            when(count('lap') > 1, round(((max('lap') - min('lap')) / (count('lap') - 1)), 0)).otherwise(-1).alias('avg_laps_between_pitstops'),  # Calculate and round the average laps between pit stops for each group
                            round((col('avg_pitStop_duration') / col('avg_laps_between_pitstops')), 2).alias('pitStop_efficiency')  # Calculate Pit Stop Efficiency by dividing average pit stop duration by average laps between pit stops
                        )

# COMMAND ----------

# Check the average of total pit stops 
pitStops.groupBy().avg("total_pitStops").collect()[0][0]

# COMMAND ----------

# Create a  column for pit stop frequency labels based on total_pitStops (Here rounded value of 2 is taken)
pitStops = pitStops.withColumn("pitStopFrequencyLabel",
                   when(col("total_pitStops") > 2, "High Pit Stop Frequency")
                  .when(col("total_pitStops") <= 2, "Low Pit Stop Frequency")
                  .otherwise("Normal Pit Stop Frequency"))

# COMMAND ----------

# There are around 9 records with null values, this is very less when compared to total records in dataframe. So, dropping these records.
pitStops = pitStops.na.drop("any")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.7 Table: Races

# COMMAND ----------

# Inspecting "races" dataframe
races.show(5, truncate = False)
races.describe().show()
races.printSchema()

# COMMAND ----------

# Extract Month from Date
races = races.withColumn("month", month(races.date))

# Extract season name from above new month column
season_conditions = [
    (col("month").between(3, 5), "Spring"),
    (col("month").between(6, 8), "Summer"),
    (col("month").between(9, 11), "Fall"),
    (col("month").isin(12, 1, 2), "Winter")
]

# Create 'season_name' column using the above conditions
races = races.withColumn("season_name", when(col("month").between(3, 5), "Spring")
                                                        .when(col("month").between(6, 8), "Summer")
                                                        .when(col("month").between(9, 11), "Fall")
                                                        .when(col("month").isin(12, 1, 2), "Winter")
                                                        .otherwise("Unknown"))

# Drop the newly created 'month' column 
races = races.drop('month')

# COMMAND ----------

# Extract time of day
time_conditions = [
    (col("time").between("06:00:00", "11:59:59"), "Morning"),
    (col("time").between("12:00:00", "17:59:59"), "Afternoon"),
    (col("time").between("18:00:00", "20:59:59"), "Evening"),
    (col("time").between("21:00:00", "05:59:59"), "Night")
]

# Using above condition and 'time' column created a new column with time of day
races = races.withColumn("time_of_day", when(col("time").between("06:00:00", "11:59:59"), "Morning")
                                                        .when(col("time").between("12:00:00", "17:59:59"), "Afternoon")
                                                        .when(col("time").between("18:00:00", "20:59:59"), "Evening")
                                                        .when(col("time").between("21:00:00", "05:59:59"), "Night"))

# COMMAND ----------

# Dropping 'month', 'time',  'url' columnd from  races dataframe due to its irrelevance
races = races.drop('month', 'time', 'url')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.8 Table: Results

# COMMAND ----------

# Inspecting "results" dataframe
results.show(5, truncate = False)
results.describe().show()
results.printSchema()

# COMMAND ----------

# Columns 'position', 'positionText' and 'positionOrder' columns contain redundant information representing the same race position, we are dropping the 'position' and positionText' column to eliminate duplication
results = results.drop('time', 'position', 'positionText')

# COMMAND ----------

# Convert "fastestLapTime" to milliseconds
fastest_lap_time_expr = (
    split(col("fastestLapTime"), ":")[0].cast("int") * 60 * 1000 +
    split(col("fastestLapTime"), ":")[1].cast("double") * 1000
)

# Using above expression to convert "fastestLapTime" to milliseconds to create a new column "fastestLapTime_ms"
results = results.withColumn("fastestLapTime_ms", fastest_lap_time_expr)

# COMMAND ----------

# dropping "fastestLapTime"
results = results.drop("fastestLapTime")

# COMMAND ----------

# Converting the few columns to integer data type to ensure consistency
# results = results.withColumn("time", col("time").cast("int"))
results = results.withColumn("milliseconds", col("milliseconds").cast("int"))
results = results.withColumn("fastestLap", col("fastestLap").cast("int"))
results = results.withColumn("rank", col("rank").cast("int"))

# Converting 'fastestLapTime' column to double
results = results.withColumn("fastestLapTime_ms", col("fastestLapTime_ms").cast("float"))

# COMMAND ----------

# Filter the results dataframe to include only races where statusId is equal to 1 which are the races where races are finished
# results_filter = results.filter(results.statusId == 1)

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that there are missing values in 'milliseconds', 'fastestLap' , 'rank', 'fastestLapTime_ms', and 'fastestLapSpeed'. So, we will replace missing values in numerical columns with mean and categorical columns with zero.

# COMMAND ----------

# Replacing missing values of 'fastestLap', 'rank' with zero 
results = results.na.fill(value=0,subset=["fastestLap", "rank"])

# Calculate mean for numerical columns with missing values
numerical_columns = ["milliseconds", "fastestLapSpeed", "fastestLapTime_ms"]
mean_values = results.select(*[mean(col(column)).alias(column) for column in numerical_columns]).first()

# Extract mean values
mean_milliseconds = mean_values["milliseconds"]
mean_fastestLapSpeed = mean_values["fastestLapSpeed"]
mean_fastestLapTime_ms = mean_values["fastestLapTime_ms"]

# Replace missing values in numerical columns with mean
results = results.fillna(
    {"milliseconds": mean_milliseconds, "fastestLapSpeed": mean_fastestLapSpeed, "fastestLapTime_ms": mean_fastestLapTime_ms},
    subset=numerical_columns
)

# COMMAND ----------

# Calculate the Average Lap Time in minutes
results = results.withColumn("avg_lap_time_seconds", col("fastestLapTime_ms") / 1000)

# Drop 'fastestLapTime_ms' column
results = results.drop("fastestLapTime_ms")

# COMMAND ----------

# Categorize the positions into groups like Top 3, Top 10 and Outside Top 10
results = results.withColumn("position_group", when(col("positionOrder") <= 3, "Top 3")
                                              .when(col("positionOrder") <= 10, "Top 10")
                                               .otherwise("Outside Top 10"))

# COMMAND ----------

# Calculate drivers experience in terms of number of races a driver participated 
results = results.withColumn("driver_experience", count("raceId").over(Window.partitionBy("driverId").orderBy("raceId")))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create a new column named 'constructor_success' , this will help in whether constructor's "positionOrder" is within the top 3 positions.  If the position order is less than or equal to 3, a value of 1 is assigned or else 0 is assigned.

# COMMAND ----------

# Calculate constructor success 
results = results.join(results .groupBy("constructorId").agg(max(when(col("positionOrder") <= 3, 1).otherwise(0)).alias("constructor_success"))
                               .select("constructorId", "constructor_success"),"constructorId")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.9 Table: Status

# COMMAND ----------

# Inspecting "status" dataframe
status.show(5, truncate = False)
status.describe().show()
status.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct

status.agg(*(countDistinct(col(c)).alias(c) for c in status.columns)).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Data Exploration

# COMMAND ----------

# Convert all the to Pandasdatafrmaes to visualize 
circuits_pd = circuits.toPandas()
constructorResults_pd = constructorResults.toPandas()
constructors_pd = constructors.toPandas()
drivers_pd = drivers.toPandas()
lapTimes_pd = lapTimes.toPandas()
pitStops_pd = pitStops.toPandas()
races_pd = races.toPandas()
results_pd = results.toPandas()
status_pd = status.toPandas()

# Merge DataFrames using the given logic
results_races_pd = results_pd.merge(races_pd, on='raceId', how='inner')
results_races_drivers_pd = results_races_pd.merge(drivers_pd, on='driverId', how='inner')
results_races_drivers_constructors_pd = results_races_drivers_pd.merge(constructors_pd, on='constructorId', how='inner')
results_races_drivers_constructors_status_pd = results_races_drivers_constructors_pd.merge(status_pd, on='statusId', how='inner')
results_races_drivers_constructors_status_laptimes_pd = results_races_drivers_constructors_status_pd.merge(
    lapTimes_pd, on=['raceId', 'driverId'], how='inner')
basetable_pd = results_races_drivers_constructors_status_laptimes_pd.merge(
    pitStops_pd, on=['driverId', 'raceId'], how='inner')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Competitive Dynamics
# MAGIC We can use the historical performance date to understand competative dynamics. 

# COMMAND ----------

# Plot number of races conducted over a periof of time to understand how F1 racing has gained popularity among audience. 
plt.figure(figsize=(10, 6))
sns.histplot(data=races_pd, x='year', bins=20, kde=True)
plt.xlabel('Year')
plt.ylabel('Number of Races')
plt.title('Distribution of Number of Races Conducted per Year')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Plot Top 10 drivers for the period of 2008 - 2018

# Filter data for the last 10 years (from 2018 to 2008)
start_year = 2018
end_year = 2008
last_10_years = range(start_year, end_year - 1, -1)

# Filter and aggregate data
filtered_data = basetable_pd[basetable_pd['year'].isin(last_10_years)]
driver_wins = filtered_data[filtered_data['positionOrder'] == 1].groupby('driverId')['positionOrder'].count()
driver_rankings = filtered_data.groupby('driverId')['points'].sum()

# Combine wins and rankings
driver_stats = pd.DataFrame({'Wins': driver_wins, 'Points': driver_rankings})

# Get top 10 drivers by wins
top_25_drivers = driver_stats.sort_values(by='Wins', ascending=False).head(10)

# Reading drivers dataframe againto get names
drivers_names = spark.read.csv(file_location + "drivers.csv", header=True, inferSchema=True)

# Combine "forename" and "surname" columns into a single "driver_name" column
drivers_names = drivers_names.withColumn("driver_name", concat(col("forename"),lit(" "), col("surname")))
drivers_names_pd = drivers_names.toPandas()

# Plotting
plt.figure(figsize=(10, 12))
sns.barplot(x='Wins', y=top_25_drivers.index.map(drivers_names_pd.set_index('driverId')['driver_name']), data=top_25_drivers, orient='h')
plt.xlabel('Number of Wins')
plt.ylabel('Driver')
plt.title('Top 25 Drivers by Wins from 2018 to 2008')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Constructor Success and Total Wins

# Grouping by constructorId and counting the number of wins
constructor_wins = results_races_drivers_constructors_pd[results_races_drivers_constructors_pd['positionOrder'] == 1].groupby('constructorId')['positionOrder'].count()

# Get constructor names for the x-axis labels
constructor_names = results_races_drivers_constructors_pd.groupby('constructorId')['constructor_name'].first()

# Creating a new DataFrame with constructorId, constructor_name, and number of wins
constructor_wins_df = pd.DataFrame({'constructorId': constructor_wins.index, 'constructor_name': constructor_names[constructor_wins.index], 'wins': constructor_wins.values})

# Sorting the DataFrame by number of wins
constructor_wins_df = constructor_wins_df.sort_values(by='wins', ascending=False)

# Creating the bar plot using Seaborn
plt.figure(figsize=(10, 10))
sns.barplot(x='wins', y='constructor_name', data=constructor_wins_df, palette='viridis')
plt.xlabel('Number of Wins')
plt.ylabel('Constructor')
plt.title('Number of Wins per Constructor')
plt.show()

# COMMAND ----------

# Bar chart: Distribution of Position Groups
plt.figure(figsize=(10, 6))
position_group_counts = results_pd['position_group'].value_counts().sort_index()
sns.barplot(x=position_group_counts.index, y=position_group_counts.values)
plt.title('Distribution of Position Groups')
plt.xlabel('Position Group')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

# Scatter plot of totalWins vs totalRaces
plt.figure(figsize=(10, 8))
sns.scatterplot(data=results_races_drivers_pd, x='totalRaces', y='totalWins')
plt.title('Total Wins vs Total Races')
plt.xlabel('Total Races')
plt.ylabel('Total Wins')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC From the above scatter plot we can see that a co-relation between total races and total wins. It is evident that the drivers driving experience (in terms of races) is proportional to winning the races. So, it would be a better choice to pick expereinced drivers. 

# COMMAND ----------

# Histogram of winRate
plt.figure(figsize=(10, 8))
sns.histplot(data=results_races_drivers_pd, x='winRate', bins=20, kde=True)
plt.title('Win Rate Distribution')
plt.xlabel('Win Rate')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This win rate distribution solidifies the understanding we derived from the scatter plot of total wins and total races, as most of the drivers have a lower win rate due to participating in fewer races. The win rate increases with the number of races a driver has participated in.

# COMMAND ----------

# Driver Experience vs Position Order
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_races_drivers_pd, x='driver_experience', y='rank', ci=None)
plt.title('Driver Experience vs Position Order')
plt.xlabel('Driver Experience')
plt.ylabel('Position Order')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Impact of weather conditions

# COMMAND ----------

# Impact of seasons on each circuit
plt.figure(figsize=(14, 8))
sns.lineplot(x='name', y='totalWins', hue='season_name', data=results_races_drivers_constructors_status_laptimes_pd)
plt.title('Impact of Season on Winning for Each Season')
plt.xlabel('Circuit Name')
plt.ylabel('Total Wins')
plt.xticks(rotation=90)
plt.legend(title='Season')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Impact of season on average lap time 
plt.figure(figsize=(10, 6))
sns.boxplot(x='season_name', y='avg_lap_time_seconds', data=results_races_drivers_constructors_status_laptimes_pd)
plt.title('Season vs Average Lap time (in Seconds)')
plt.xlabel('Season Name')
plt.ylabel('Average Lap Time')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Impact of season on average number of positions changed 
plt.figure(figsize=(10, 6))
sns.boxplot(x='season_name', y='avg_position_change', data=results_races_drivers_constructors_status_laptimes_pd)
plt.title('Season vs Average number of positions changed')
plt.xlabel('Season Name')
plt.ylabel('Positions Changed')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Impact of season on total number of pit stops
plt.figure(figsize=(10, 6))
sns.barplot(x='season_name', y='total_pitStops', data=basetable_pd)
plt.title('Impact of Weather on Total Pit Stops')
plt.xlabel('Season Name')
plt.ylabel('Total Pit Stops')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Adding time of the day
plt.figure(figsize=(10, 6))
sns.barplot(x='season_name', y='total_pitStops', hue='time_of_day', data=basetable_pd, ci=None)
plt.title('Impact of Season and Time of Day on Total Pit Stops')
plt.xlabel('Season Name')
plt.ylabel('Total Pit Stops')
plt.tight_layout()
plt.legend(title='Time of Day')
plt.show()

# COMMAND ----------

# Impact of seasons on average pit stop duration
plt.figure(figsize=(10, 6))
sns.boxplot(x='season_name', y='avg_pitStop_duration', data=basetable_pd)
plt.title('Impact of Season on Pit Stop Duration')
plt.xlabel('Season Name')
plt.ylabel('Average Pit Stop Duration')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Car safety effecting outcome

# COMMAND ----------

# Group the data by status and count the number of races per status
status_count = results_races_drivers_constructors_status_laptimes_pd.groupby("status")["raceId"].count().reset_index()

# Sort the status_count dataframe in descending order
top_20_status = status_count.sort_values(by='raceId', ascending=False).head(20)

# Create a bar plot to show the number of races for the top 20 statuses
plt.figure(figsize=(12, 8))
sns.barplot(x='status', y='raceId', data=top_20_status)
plt.title('Top 20 Race Statuses by Number of Races')
plt.xlabel('Status')
plt.ylabel('Number of Races')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC Majority of the races were finsihed but there are few races that were not finished due to car specific problemns like gearbox, hydrualics, accidents, brakes, etcetera. SO, concentrating on these will be an advantage and next we will see if fastest lap speed has any realtion to these failures.

# COMMAND ----------

# List of specific statuses to analyze
specific_statuses = ['Engine', 'Spun-off', 'Collision', 'Gearbox-off', 'Accident', 'Hydraulics', 'Suspension', 'Brakes', 'Clutch']

# Filter the data for specific statuses
filtered_data = results_races_drivers_constructors_status_laptimes_pd[
    results_races_drivers_constructors_status_laptimes_pd['status'].isin(specific_statuses)]

# Group by fastestLapSpeed and count the number of races
grouped_data = filtered_data.groupby(['fastestLapSpeed', 'status'])['raceId'].count().reset_index()

# Create a scatter plot to show how the number of races with specific status impacts fastest lap speed
plt.figure(figsize=(10, 8))
sns.scatterplot(x='fastestLapSpeed', y='raceId', hue='status', data=grouped_data)
plt.title('Impact of Number of Races with Specific Status on Fastest Lap Speed')
plt.xlabel('Fastest Lap Speed')
plt.ylabel('Number of Races')
plt.legend(title='Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that most of the these failures happen between 140 to 240. So, it would be better to concentrate on tehse reasons

# COMMAND ----------

# CHeck relationship between fastest lap speed and positon order relationship between fastestLapSpeed and positionOrder
plt.figure(figsize=(12, 8))
sns.lineplot(x='positionOrder', y='fastestLapSpeed', data=results_races_drivers_constructors_status_laptimes_pd)
plt.title('Relationship between Fastest Lap Speed and Position Order')
plt.xlabel('Position Order')
plt.ylabel('Fastest Lap Speed')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Most of the races are not finished when speed ranges between 140 and 240, at the same time there is possibility of closing in top 3 order when speed is maintained 200 and 205. So, making cars secure will help in reaching this speed safely.

# COMMAND ----------

results_races_drivers_constructors_status_laptimes_pd.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Importance of Rapid Pit Stops

# COMMAND ----------

# Compare the realtionship between average pit stop durationa nd pit stop efficiency
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_pitStop_duration', y='pitStop_efficiency', data=basetable_pd)
plt.title('Average Pit Stop Duration vs. Pit Stop Efficiency')
plt.xlabel('Average Pit Stop Duration')
plt.ylabel('Pit Stop Efficiency')
plt.tight_layout()
plt.show()

# COMMAND ----------

plt.figure(figsize=(12, 6))
sns.lineplot(x='totalWins', y='pitStop_efficiency', data=basetable_pd)
plt.title('Pit Stop Efficiency over Races')
plt.xlabel('Total Race lWins')
plt.ylabel('Pit Stop Efficiency')
plt.tight_layout()
plt.show()

# COMMAND ----------

# COmap
plt.figure(figsize=(10, 6))
sns.lineplot(x='avg_pitStop_duration', y='position_change_index', data=basetable_pd)
plt.title('Comparison between Average Pit Stop Duration and Position Change Index')
plt.xlabel('Average Pit Stop Duration')
plt.ylabel('Position Change Index')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Modelling

# COMMAND ----------

# Merge all the relavant dataframes to form a basetable
results_races = results.join(races, "raceId", "inner")
results_races_drivers = results_races.join(drivers, "driverId", "inner") 
results_races_drivers_constructors = results_races_drivers.join(constructors, "constructorId", "inner")
results_races_drivers_constructors_status = results_races_drivers_constructors.join(status, 'statusId', "inner")
results_races_drivers_constructors_status_laptimes = results_races_drivers_constructors_status.join(lapTimes, ['raceId', 'driverId'], "inner")
basetable = results_races_drivers_constructors_status_laptimes.join(pitStops, ['driverid', 'raceid'], "inner")

# COMMAND ----------

basetable.display()

# COMMAND ----------

# Dropping few columns that are not necessary 
basetable = basetable.drop("statusId", "year", "date", "constructorRef", "constructorRef", "name", "constructor_name")

# COMMAND ----------

# Status column has different types of statuses, so repalcing status values other than 'Finished' with 'Not Finished'
basetable = basetable.withColumn("status", when(col("status") == "Finished", "Finished").otherwise("Not Finished"))

# Replace status values with 1 and 0, where 1 indicting 'Finished' and 0 indicating 'Not Finished'
basetable = basetable.withColumn("status", when(col("status") == "Finished", 1).otherwise(0))

# COMMAND ----------

# List of categorical columns that will be encoded 
categorical_columns = [ 'position_group','season_name','time_of_day','constructor_nationality', 'pitStopFrequencyLabel']

# Initialize StringIndexers for each categorical column
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index",  handleInvalid="skip") for col in categorical_columns]

# Create a pipeline to execute the StringIndexers in sequence
pipeline = Pipeline(stages=indexers)

# Fit and transform the pipeline on the DataFrame
basetable = pipeline.fit(basetable).transform(basetable)

# Drop the columns that have been encoded 
basetable = basetable.drop(*categorical_columns)

# COMMAND ----------

# Create a train and test set with a 70% train, 30% test split
basetable_train, basetable_test = basetable.randomSplit([0.7, 0.3],seed=123)

# Print the number of observations in training and test data
print(basetable_train.count())
print(basetable_test.count())

# COMMAND ----------

# Create and fit the RFormula transformer on the training and test data
train = RFormula(formula="status ~ . - driverId").fit(basetable_train).transform(basetable_train)
test = RFormula(formula="status ~ . - driverId").fit(basetable_test).transform(basetable_test)

# Print the number of observations in training and test data
print("train nobs: " + str(train.count()))
print("test nobs: " + str(test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Support Vector Machine

# COMMAND ----------

# Define your classifier
lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.1.1 SVM Model Scoring

# COMMAND ----------

# Compute predictions for test data
predictions_test_svm = lsvcModel.transform(test)

# Initialize evaluators specific to the current model
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# Calculate area under ROC (AUC) for test sets
auc_test_svm = binary_evaluator.evaluate(predictions_test_svm)

# Show the AUC
print("SVM Model Performance Scores are as following:")
print("Test AUC = %g" % (auc_test_svm))

# Calculate accuracy for test sets
accuracy_test_svm = multiclass_evaluator.evaluate(predictions_test_svm)

# Show the accuracy
print("Test Accuracy = %g" % (accuracy_test_svm))

# Calculate F1 score for test sets
f1_test_svm = f1_evaluator.evaluate(predictions_test_svm)

# Show the F1 score
print("Test F1 Score = %g" % (f1_test_svm))

# Calculate precision for test set
precision_test_svm = precision_evaluator.evaluate(predictions_test_svm)

# Show the precision
print("Test Precision = %g" % (precision_test_svm))

# Calculate recall for test set
recall_test_svm = recall_evaluator.evaluate(predictions_test_svm)

# Show the recall
print("Test Recall = %g" % (recall_test_svm))

# COMMAND ----------

# #Complete the #FILL IN# gaps
# from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# # Compute predictions for test data
# predictions = lsvcModel.transform(test)

# # Show the computed predictions and compare with the original labels
# predictions.select("features", "label", "prediction").show(10)

# # Define the evaluator method with the corresponding metric and compute the classification error on test data
# evaluator = MulticlassClassificationEvaluator().setMetricName('accuracy')
# accuracy = evaluator.evaluate(predictions) 

# # Show the accuracy
# print("Test accuracy = %g" % (accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Logistic Regression

# COMMAND ----------

train_lr = RFormula(formula="status ~ . - driverId").fit(basetable_train).transform(basetable_train)
test_lr = RFormula(formula="status ~ . - driverId").fit(basetable_test).transform(basetable_test)

print("train nobs: " + str(train_lr.count()))
print("test nobs: " + str(test_lr.count()))

# COMMAND ----------

# Estimate a logistic regression and fit it on test data
logreg = LogisticRegression().fit(train_lr)

# Compute predictions for test data
predictions_test_logreg = logreg.transform(test_lr)

# COMMAND ----------

# Initialize evaluators specific to the current model
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# Calculate area under ROC (AUC) for test sets
auc_test_logreg = binary_evaluator.evaluate(predictions_test_logreg)

# Show the AUC
print("Logistic Regression Model Performance Scores are as following:")
print("Test AUC = %g" % (auc_test_logreg))

# Calculate accuracy for test sets
accuracy_test_logreg = multiclass_evaluator.evaluate(predictions_test_logreg)

# Show the accuracy
print("Test Accuracy = %g" % (accuracy_test_logreg))

# Calculate F1 score for test sets
f1_test_logreg = f1_evaluator.evaluate(predictions_test_logreg)

# Show the F1 score
print("Test F1 Score = %g" % (f1_test_logreg))

# Calculate precision for test set
precision_test_logreg = precision_evaluator.evaluate(predictions_test_logreg)

# Show the precision
print("Test Precision = %g" % (precision_test_logreg))

# Calculate recall for test set
recall_test_logreg = recall_evaluator.evaluate(predictions_test_logreg)

# Show the recall
print("Test Recall = %g" % (recall_test_logreg))

# COMMAND ----------

# Train a logistic regression model on the training set
# logreg = LogisticRegression().fit(train_lr)

# Generate predictions for the training set
predictions_train_logreg = logreg.transform(train_lr)

# Initialize evaluators specific to the current model
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# Calculate area under ROC (AUC) for training set
auc_train_logreg = binary_evaluator.evaluate(predictions_train_logreg)

# Calculate accuracy for training set
accuracy_train_logreg = multiclass_evaluator.evaluate(predictions_train_logreg)

# Calculate F1 score for training set
f1_train_logreg = f1_evaluator.evaluate(predictions_train_logreg)

# Calculate precision for training set
precision_train_logreg = precision_evaluator.evaluate(predictions_train_logreg)

# Calculate recall for training set
recall_train_logreg = recall_evaluator.evaluate(predictions_train_logreg)

# Display the performance metrics for the training set
print("Logistic Regression Model Performance Scores on Training Set:")
print("Training AUC = %g" % auc_train_logreg)
print("Training Accuracy = %g" % accuracy_train_logreg)
print("Training F1 Score = %g" % f1_train_logreg)
print("Training Precision = %g" % precision_train_logreg)
print("Training Recall = %g" % recall_train_logreg)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3 Decision Tree

# COMMAND ----------

train_dt = RFormula(formula="status ~ . - driverId").fit(basetable_train).transform(basetable_train)
test_dt = RFormula(formula="status ~ . - driverId").fit(basetable_test).transform(basetable_test)

print("train nobs: " + str(train_dt.count()))
print("test nobs: " + str(test_dt.count()))

# COMMAND ----------

# Decision Tree Classifier
dt_model = DecisionTreeClassifier().fit(train_dt)
dt_pred = dt_model.transform(test_dt)

# COMMAND ----------

# Generate predictions for the test set
predictions_test_dt = dt_model.transform(test_dt)

# Initialize evaluators specific to the current model
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")	
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# Calculate area under ROC (AUC) for test set
auc_test_dt = binary_evaluator.evaluate(predictions_test_dt)

# Calculate accuracy for test set
accuracy_test_dt = multiclass_evaluator.evaluate(predictions_test_dt)

# Calculate F1 score for test set
f1_test_dt = f1_evaluator.evaluate(predictions_test_dt)

# Calculate precision for test set
precision_test_dt = precision_evaluator.evaluate(predictions_test_dt)

# Calculate recall for test set
recall_test_dt = recall_evaluator.evaluate(predictions_test_dt)

# Display the performance metrics for the test set
print("Decision Tree Model Performance Scores on test Set:")
print("test AUC = %g" % auc_test_dt)
print("test Accuracy = %g" % accuracy_test_dt)
print("test F1 Score = %g" % f1_test_dt)
print("test Precision = %g" % precision_test_dt)
print("test Recall = %g" % recall_test_dt)

# COMMAND ----------

# Generate predictions for the training set
predictions_train_dt = dt_model.transform(train_dt)

# Initialize evaluators specific to the current model
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# Calculate area under ROC (AUC) for training set
auc_train_dt = binary_evaluator.evaluate(predictions_train_dt)

# Calculate accuracy for training set
accuracy_train_dt = multiclass_evaluator.evaluate(predictions_train_dt)

# Calculate F1 score for training set
f1_train_dt = f1_evaluator.evaluate(predictions_train_dt)

# Calculate precision for training set
precision_train_dt = precision_evaluator.evaluate(predictions_train_dt)

# Calculate recall for training set
recall_train_dt = recall_evaluator.evaluate(predictions_train_dt)

# Display the performance metrics for the training set
print("Decision Tree Model Performance Scores on Training Set:")
print("Training AUC = %g" % auc_train_dt)
print("Training Accuracy = %g" % accuracy_train_dt)
print("Training F1 Score = %g" % f1_train_dt)
print("Training Precision = %g" % precision_train_dt)
print("Training Recall = %g" % recall_train_dt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.4 Feature Importance

# COMMAND ----------

# Get the coefficients (feature importance) from the model
coefficients = logreg.coefficients

# Get the names of the features
feature_names = train_lr.columns

# Get the coefficients (feature importance) from the model
coefficients = logreg.coefficients.tolist()  # Convert numpy array to a list

# Get the names of the features
feature_names = train.columns

# Create a DataFrame with feature names and coefficients
feature_importance_df = spark.createDataFrame(zip(feature_names, coefficients), ["Feature", "Importance"])

# Sort the DataFrame by Importance in descending order
feature_importance_df = feature_importance_df.orderBy(col("Importance").desc())

# Show the sorted feature importance DataFrame
feature_importance_df.display()

# COMMAND ----------

# Re-running the model again with the top 12 features as mentioned in the step above.

# Based on the above feature selection method, we are only using the top 12 variables for modelling.

features_lr =  ["raceId",
                "driverId",
                "winRate",
                "avg_pitStop_duration",
                "laps",
                "avg_lap_time_seconds",
                "fastestLapSpeed",
                "points",
                "age",
                "max_position",
                "avg_laps_between_pitstops",
                "total_pitStop_duration",
                "season_name_index",
                "status"]


# Select only the columns in the features list
Features_for_lr = basetable.select(*[col(feature) for feature in features_lr])
display(Features_for_lr)

#Create a train and test set with a 70% train, 30% test split
basetable_train_lr, basetable_test_lr = Features_for_lr.randomSplit([0.7, 0.3],seed=123)

print(Features_for_lr.count(),basetable_train_lr.count(),basetable_test_lr.count())

# COMMAND ----------

train_best = RFormula(formula="status ~ . - driverId").fit(basetable_train_lr).transform(basetable_train_lr)
test_best = RFormula(formula="status ~ . - driverId").fit(basetable_test_lr).transform(basetable_test_lr)

print("train nobs: " + str(train_best.count()))
print("test nobs: " + str(test_best.count()))

# COMMAND ----------

lr_model_topFeatures = DecisionTreeClassifier().fit(train_best)

lr_model_topFeatures_pred = lr_model_topFeatures.transform(test_best)

# COMMAND ----------

# Initialize evaluators specific to the current model
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

# Calculate area under ROC (AUC) for test sets
auc_test_logreg_topFeatures = binary_evaluator.evaluate(lr_model_topFeatures_pred)

# Show the AUC
print("Top-N features Logistic Regression Model Performance Scores are as following:")
print("Test AUC = %g" % (auc_test_logreg_topFeatures))

# Calculate accuracy for test sets
accuracy_test_logreg_topFeatures = multiclass_evaluator.evaluate(lr_model_topFeatures_pred)

# Show the accuracy
print("Test Accuracy = %g" % (accuracy_test_logreg_topFeatures))

# Calculate F1 score for test sets
f1_test_logreg_topFeatures = f1_evaluator.evaluate(lr_model_topFeatures_pred)

# Show the F1 score
print("Test F1 Score = %g" % (f1_test_logreg_topFeatures))

# Calculate precision for test set
precision_test_logreg_topFeatures = precision_evaluator.evaluate(lr_model_topFeatures_pred)

# Show the precision
print("Test Precision = %g" % (precision_test_logreg_topFeatures))

# Calculate recall for test set
recall_test_logreg_topFeatures = recall_evaluator.evaluate(lr_model_topFeatures_pred)

# Show the recall
print("Test Recall = %g" % (recall_test_logreg_topFeatures))

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.5 Cross-Validation

# COMMAND ----------

# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# from pyspark.ml.evaluation import BinaryClassificationEvaluator

# # Define the LogisticRegression estimator
# logreg_cv = LogisticRegression(labelCol="label", featuresCol="features")

# # Initialize the BinaryClassificationEvaluator for AUC
# evaluator_lr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# # Define the parameter grid for hyperparameter tuning
# param_grid = ParamGridBuilder() \
#     .addGrid(logreg_cv.regParam, [0.01, 0.1, 1.0]) \
#     .addGrid(logreg_cv.elasticNetParam, [0.0, 0.5, 1.0]) \
#     .build()

# # Initialize the CrossValidator with the LogisticRegression estimator
# cross_validator = CrossValidator(estimator=logreg_cv,
#                                  estimatorParamMaps=param_grid,
#                                  evaluator=evaluator_lr,
#                                  numFolds=5)  # Number of cross-validation folds

# # Fit the CrossValidator to the training data
# cv_model = cross_validator.fit(train_lr)

# # Make predictions on the test set
# cv_predictions = cv_model.transform(test_lr)

# # Evaluate the model performance
# cv_auc = evaluator_lr.evaluate(cv_predictions)

# print("Cross-Validation AUC:", cv_auc)
