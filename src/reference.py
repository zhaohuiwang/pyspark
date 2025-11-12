
'''
https://www.palantir.com/docs/foundry/transforms-python-spark/pyspark-overview/

Aparch Spark is a unified analytics engine for large-scale data processing. It is written primarily in Scala and run on the Jave Virtual Machine (JVM). 
PySpark is the Python API for Apache Spark. It allows developers to interact with the Spark engine using Python. In other words, PySpark is a wrapper language that allows you to interface with an Apache Spark backend.

Distributed computing: PySpark runs computations in paraller across cluster.
Resilient distributed dataset (RDD):
Lazy evaluation: The transformations are only computed when an action requires a result to be returned to the driver program (for efficiency propuse).
Every Spark application consists of a drive program that runs the user's main function and executes various parallel operations on a cluster (consists of multiple nodes)
RDD Persistence: methods persist() or cache(), a shorthand for using the default storage level MEMORY_ONLY, can be used to persist any RDD in memory on the nodes (as serizlized Java object for much faster access the next time you query it) or on disk or replicated across multiple nodes by passing a StorageLevel object (e.g. pyspark.storagelevel.StorageLevel.DISK_ONLY). Spark's cache is fault-tolerant - if any partition of an RDD is lost, it will be recomputed automatically using the transformations that originally created it.

MEMORY_ONLY:
MEMORY_AND_DISK:
MEMORY_ONLY_SER(Java and Scala):
MEMORY_AND_DISK_SER (Java and Scala):
DISK_ONLY:
MEMORY_ONLY_2, MEMORY_AND_DISK_2, etc.
OFF_HEAP (experimental):

'''
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F, types as T, Window as W


###### Create a SparkSession
spark = (
    SparkSession.builder
    .appName("PySparkZW")
    .master("local[*]") # All available cores; n Exactly n cores; n% 2 - n cores
    .getOrCreate()      # Creates or returns existing session
    )
# Default True in Spark 4.0 for the following config()
# spark.sql.adaptive.enabled
# spark.sql.execution.arrow.pyspark.enabled
# spark.sql.adaptive.coalescePartitions.enabled

######################################################
###### Create DataFrames
data = [("Alice", 31, "New York", 5.3, "1990-12-15"),
        ("Bob", 42, "London", 3.8, "1979-05-20"),
        ("Charlie", 53, "Paris", 8.4, "1968-09-25"),
        ("David", 32, "New York", 4.6, "1989-04-08")]

columns = ["Name", "Age", "City", "Tenure", "date_of_birth"]

df = spark.createDataFrame(data, schema=columns)

data_id =[("Alice", 1, ),
          ("Bob", 2),
          ("Charlie", 3),
          ("David", 4)]
columns_id = ["Name", "ID"]

df_id = spark.createDataFrame(data_id, schema=columns_id)

######################################################
###### Examining the DataFrame structure and summary 
df.printSchema() # printSchema() is equivalent to df.info() in Pandas

df.columns  # Returns all columns as a Python list
df.dtypes   # Returns all columns names and their data types as a list of tuples


# Using limit to return a new DataFrame with only the first n rows
df.limit(2).show()

# DataFrame.show(n=20, truncate=True, vertical=False)
df.show(2,0) # show top 2 rows and no truncation to the columns values


# Count and shape
df.count()
df_id.select('ID').distinct().count()
print(f"DataFrame shape: ({df.count()}, {len(df.columns)})")

# To get the maximum value in a column, return as a DataFrame by default
df.agg(F.max("Age").alias("MaxAge")).show()
# To get the value itself not a DataFrame.
df.agg(F.max("Age").alias("MaxAge")).collect()[0][0]

# To get summary statistics per group
df.groupBy("City").agg(F.avg("Age").alias("AverageAge")).show()

######################################################
###### Subsetting rows - using filter() or where()
# where is an alias of filter and does exactly the same thing

# Subsetting rows using filter() and where()
df.filter(F.col('Age')==2).show() # select all rows with value 2 in Age column
df.filter((F.col('Age')>=2) & (F.col('Name')=='David')).show() # using '&' for AND, '|' for OR for multiple conditions, each in ().
df.where(df.Age==2).show()

'''
Because PySpark logical operations are applied on whole columns, we can not use 'and' or 'or' Python operators. These operators expect both arguments to already evaluate to single booleans. PySpark is able to interpret the '&' (bitwise-and), | (bitwise-or) operators, and ~ negatition (tilda) sumbols to buid up a SQL query that runs very efficiently on all rows. 
~ : Negation
& : And
| : Or
^ : Exclusive-or, aka XOR, exclusive disjunction
It only returns a "true" value of 1 if the two values are exclusive, i.e. they are both different. In the set theory, it returns all the elements that is "exclusive" to one set (not "shared"). 
A^B = (A | B) - (A & B)
A ⊕ B = (A U B) - (A ∩ B)
(A XOR B) = (A UNION B) EXCEPT (A INTERSECT B) 
'''


# Compare against a list of allowed values
df.filter(F.col('Age').isin([42, 53])).show()

# Sort results
df.orderBy(df.Age.asc()).show()
df.orderBy(df.Age.desc()).show()

######################################################
###### Subsetting Columns using select()
# Columns are managed by the PySpark class: pyspark.sql.Column. F.col("column_name") or F.column("column_name") 

df.select('Name', 'Age')    # passing in column names directly
df.select(F.col('Name'), F.col('Age'))  # or passing column instances

cols = df.columns[0:2]
df.select(*cols)    # passing in an "unpacked" array
df.select(cols)     # passing in an array

df.select(F.col("Name").alias("F_Name"), "Age").show()
df.select(df.ID.try_cast("string")).show()


# Using joining to subset or join two DataFrames
# Match on same columns
df_nid = df.join(df_id, on="Name", how="inner")
df_nid.show()
# Match on different columns (name), both will be retained, here two 'Name' retained 
df.join(df_id, on=(df.Name==df_id.Name), how="inner").show() 

'''
Avoid right join. Switch the order of your dataframes and use left instead.
"join explosion" when there are multiple matches for a key, the rows will be duplicated as many as there were matches. 


'''


'''
Doing a select at the begining of a PySpark transformation, or before returning, is considered good practice. This select statement specifies the contract with both the reader and the code about the expected dataframe schema for inputs and outputs.

Chaining of expressions
Avoid chaining expressions of different behaviors or contexts. For example, if you were to perform selection, new column creation and joining, these three  logical code blocks are better isolated. It is recommend chains of no longer than 3-5 statements.

Avoid using literal strings or integers in filtering conditions, new values of columns, and so on. Instead, extract them into variables, constants, dicts or classes as suitable, to capture their meaning. This makes the code more readable and enforces consistency across the repository.
'''

window_spec = W.partitionBy("City").orderBy(F.desc("Age"))
df.select(
    'Name', 'Age', 'City',
    F.col('date_of_birth').alias('DOB'),
    F.avg('Age').over(window_spec).alias('avg_age'),
    F.rank().over(window_spec).alias('age_rank')
    ).show()

'''

'''
######################################################
###### Column operations

# .withColumnRenamed("old_name", "new_name") method to rename a column
df.withColumnRenamed('Name', 'name').show()

# .withColumn(name, column_expression) method to create or update a column
df.withColumn("name", F.lower(F.col("Name"))).show() # update a column
df.withColumn("ageX2", F.col("Age") * 2).show()
df.withColumn("NA", F.concat(F.col("Name"), F.col("Age").cast("string"))).show()

# .drop(*cols) directly drop columns or select() an indirect way
df.drop('date_of_birth').show()
# you may pass column names directly, or an array, "unpacked" array, column instances. Same as .select()

# cast datatype
df.select(df.Age.cast("string").alias("age")).show()
# or
from pyspark.sql.types import StringType
df.select(df.Age.cast(StringType()).alias("age")).show()

'''
Here are all the DataTypes that exist: NullType, StringType, BinaryType, BooleanType, DateType, TimestampType, DecimalType, DoubleType, FloatType, ByteType, IntegerType, LongType, ShortType, ArrayType, MapType, StructType, StructField
'''
# F.when(condition, value).otherwise(value2)
at_least_40 = F.when(F.col("age") >= 40, True).otherwise(False)
df.select(at_least_40.alias("at_least_21")).show()


df_ = df.select("*") # Create a duplicate

for col in df.columns:
    df_ = df_.withColumnRenamed(col, col.lower().replace(' ', '_').replace('-', '_'))
df_.show()

# In general, for loops are inefficient in Spark. List comprehension is preferred here
df_ = df.select(*[col.lower().replace(' ', '_').replace('-', '_') for col in df.columns])
df_.show()


df.select(*(F.col(c) for c in df.columns if c in df_id.columns)).show()

df.where(df.Age > 40).show()     # explicit, depedent on the dataframe name
df.where(F.col("Age")>40).show() # prefered, implicit column selection

'''
Constants (Literals)
Whenever you compare a column to a constant, or "literal", such as a single hard coded string, date, or number, PySpark actually evaluates this basic Python datatype into a "literal" (same thing as declaring F.lit(value)). A literal is simply a column expression with a static value. For example, 
df.filter(F.col("X") == F.lit("Y"))

'''

# Add a new column 'Country' with a constant value
df_with_country = df.withColumn("Country", F.lit("USA"))
df_with_country.show()
# F.concat() vs F.concat_ws()
df.select(F.concat(df.Name,df.Age).alias("UID"),"Name","Age","City").show()
# SQL function expr(), execute SQL-like expressions
df.withColumn(
    "Level",
    F.expr("CASE WHEN Age >= 50 THEN 'S'" \
    "WHEN Age <= 40 THEN 'Y' ELSE 'M' END")).show()

(
    df_nid
    # forced casting into the same type instead  
    .withColumn("N_ID", F.coalesce(F.col("Name"), df_nid.ID.try_cast("string")))
    .drop(*["Name", "ID"])
    # Pandas can drop a list but not PySpark
    .drop("Name", "ID") # Unpack list elements first before dropping  
    .show()
    )
# Take the first value that is not null
df.withColumn('last_name', F.coalesce(df.Name, df.City, F.lit('N/A'))).show()

# Apply rank function over a window
window_spec = W.partitionBy("City").orderBy(F.desc("Age"))
df.withColumn("rank", F.rank().over(window_spec)).show()

df.fillna({'Name': 'Tom', 'Age': 0,}).show()

# Drop duplicate rows
df.distinct().show()
df.dropDuplicates().show()
df.dropDuplicates(subset=['Name', 'City']).show() # consider only specific columns
df.dropna(how='any', thresh=None, subset=None).show()

# .orderBy(*cols, **kwargs) alias .sort(*cols, **kwargs)
df.sort(F.asc('Age')).show()
df.sort(F.desc('Age')).show()
df.orderBy(F.asc('Age')).show()
df.orderBy(F.desc('Age')).show()

######################################################
###### String operations / functions ######
df.filter(df.Name.contains('o')).show()
df.filter(df.Name.startswith('Al')).show()
df.filter(df.Name.endswith('ice')).show()
df.filter(df.Name.isNull()).show()
df.filter(df.Name.isNotNull()).show()
df.filter(df.Name.like('Al%')).show()   # Wild card
df.filter(df.Name.rlike('[A-Z]*ice$')).show()
df.filter(df.Name.isin('Bob', 'Mike')).show()


'''
Column.like(sql_like)
Column.like('a%')	# Finds any values that starts with "a"
Column.like('%a')	# Finds any values that ends with "a"
Column.like('%or%')	# Finds any values that have "or" in any position
Column.like('_r%')	# Finds any values that have "r" in the second position
Column.like('a_%_%')# Finds any values that starts with "a" and ≥ 3 characters
Column.like('a%o')	# Finds any values that starts with "a" and ends with "o"
Column.rlike(regex)
Column.startswith(string)
Column.endswith(string)
Column.contains(string)
Column.substr(startPos, length) # Substring counting from 1 (not 0 index)
Column.isin(*cols)
Column.between(lowerBound, upperBound) # Inclusive

F.initcap(col)
F.lower(col)
F.upper(col)

F.rtrim(col)
F.rtrim(col)
F.trim(col)
F.lpad(col, len, pad)
F.rpad(col, len, pad)

F.concat(*cols)
F.concat_ws(sep, *cols)

F.regexp_replace(str, pattern, replacement)
F.regexp_extract(str, pattern, idx) # idx - matched group id. Returns empty string if no match
F.regexp_extract_all(str, regexp, idx=None) # Extract all strings in the str that match the Java regex regexp
'''


# Substring - col.substr(startPos, length) (1-based indexing)
df.withColumn('short_name', df.Name.substr(1, 2)).show()

df_id.withColumn('id', F.lpad('ID', 4, '0')).show()

# Trim the spaces
df.withColumn('name', F.trim(df.Name)).show()
df.withColumn('name', F.ltrim('Name')).show()

df.withColumn('name', F.concat('Name', F.lit(' '), 'City')).show()
df.withColumn('name', F.concat_ws('-', 'Name', 'City')).show()

df.withColumn('id', F.regexp_replace("Name", 'Al(.*)', 'AAll')).show()
df.withColumn('id', F.regexp_extract("Name", 'Al*', 0)).show()


######################################################
###### Number operations
df.printSchema()
# Round - F.round(col, scale=0)
df.withColumn('tenure', F.round('Tenure', 0)).show()
# Floor - F.floor(col)
df.withColumn('tenure', F.floor('Tenure')).show()
# Ceiling - F.ceil(col)
df.withColumn('tenure', F.ceil('Tenure')).show()
# Absolute Value - F.abs(col)
df.withColumn('tenure', F.abs('Tenure')).show()
# X raised to power Y – F.pow(x, y)
df.withColumn('exponential_growth', F.pow('Tenure', 'Age')).show()
# Select smallest value out of multiple columns – F.least(*cols)
df.withColumn('least', F.least('Tenure', 'Age')).show()
# Select largest value out of multiple columns – F.greatest(*cols)
df.withColumn('greatest', F.greatest('Tenure', 'Age')).show()

'''
F.ceil(x)
F.round(column, scale=0)
F.floor(column)

F.log(arg1, arg2=None)
F.log10(column)
F.log1p(column)

F.rand(seed=None)   # sample from uniform distribution [0.0, 1.0]
F.randn(seed=None)  # sample from the standard normal distribution

F.cos(x)
F.sin(x)
F.tan(x)
F.acos(x)
F.degrees(column)
F.radians(column)

'''
######################################################
###### Dates and timestamps
df.printSchema()
# Convert a string of known format to a date (excludes time information)
dfd = df.withColumn('date_of_birth', F.to_date('date_of_birth', 'yyyy-MM-dd'))
dfd.printSchema()

df.filter(F.year('date_of_birth') == F.lit('1990')).show()

# Add and subtract days
df.withColumn('three_days_after', F.date_add('date_of_birth', 3)).show()
df.withColumn('three_days_before', F.date_sub('date_of_birth', 3)).show()

# Add and subtract months
df.withColumn('next_month', F.add_months('date_of_birth', 1)).show()
df.withColumn('previous_month', F.add_months('date_of_birth', -1)).show()

# Get number of days between two dates
df.withColumn('days_between', F.datediff('end', 'start'))

# Get number of months between two dates
df.withColumn('months_between', F.months_between('end', 'start'))

# Keep only rows where date_of_birth is between 2017-05-10 and 2018-07-21
df.filter(
    (F.col('date_of_birth') >= F.lit('1990-05-10')) &
    (F.col('date_of_birth') <= F.lit('2005-07-21'))
).show()

''''
F.year(timestamp_col)         # Get year from timestamp_col or literal
F.quarter(timestamp_col)
F.dayofyear(timestamp_col)
F.weekofyear(timestamp_col)
F.month(timestamp_col)
F.monthname(timestamp_co)       # return month like Jan, Feb, ... 
F.dayofmonth(timestamp_col)
F.weekday(timestamp_col)
F.hour(timestamp_col)
F.minute(timestamp_col)
F.second(timestamp_col)

F.date_add(start, days)
F.date_sub(start, days)
F.add_months(start, months)
F.datediff(end, start)          # delta (number of) days
F.months_between(date1, date2)  # delta (number of) months

F.to_date(column, format=None)  # converting from string
F.to_timestamp(column, format=None)
F.to_utc_timestamp(timestamp, tz)

F.date_format(date, format)     # converting to string
F.from_unixtime(timestamp, format='yyyy-MM-dd HH:mm:ss')
F.from_utc_timestamp(timestamp, tz)

'''
df.select(F.weekday('date_of_birth').alias('date')).show()
df.select(F.date_add('date_of_birth', 5).alias('D+5')).show()
df.select(F.monthname('date_of_birth').alias('M+5')).show()
df.select(F.dayofmonth('date_of_birth').alias('DofM')).show()
df.select(F.dayofyear('date_of_birth').alias('DofY')).show()
df.select(F.weekofyear('date_of_birth').alias('WofY')).show()



######################################################
###### Aggregation and pivot tables
# Calculate the average age across all the rows
df.agg(F.avg("Age").alias("AverageAge")).show()
# Calculate the average age by city
df.groupBy("City").agg(F.avg("Age").alias("AverageAge")).show()
df.groupBy(F.col("City")).agg(F.avg("Age").alias("AverageAge")).show()

df.groupBy(F.col("City")).pivot('Name').mean('Age').show()
# pandas.pivot(data, *, columns, index=<no_default>, values=<no_default>)
# pandas.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', ...)



df.agg(F.avg("Age").alias("AverageAge")).collect()[0][0]
# Note: .collect() returns all the records as a list of row. [0][0] return the value in the first row. Here the avage age across all records.

'''
from pyspark.sql import functions as F
F.count(*cols)
F.countDistinct(*cols)
F.sum(*cols)
F.avg(col) / F.mean(col)
F.max(*cols)
F.min(*cols)
F.stddev(col)
F.variance(col)
F.corr(x, y)

F.first(*cols, ignorenulls=False)
F.last()


Aggregate functions operate on a group of rows and calculate a single return value for every group. All these aggregate functions accept input as, Column type or column name as a string and several other arguments based on the function.

F.collect_list()    # Returns all values in an input column with duplicates
F.collect_set()     # ... without duplicate
F.countDistinct()
F.count()



'''


df.groupBy("City").agg(
    F.avg("Age").alias("AverageAge"),
    F.count('Name').alias('num_employee'),
    F.max('Age').alias('oldest_age'),
    F.first('Name', True).alias('poc')
).show()

df.show()

''' Pandas 
df.groupby('A').B.agg(['min', 'max'])
df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
df.groupby("A").agg(
    b_min=pd.NamedAgg(column="B", aggfunc="min"),
    c_sum=pd.NamedAgg(column="C", aggfunc="sum")
)
df.groupby('A').agg(lambda x: sum(x) + 2)
'''

# Apply over window specification instead of groupBy()
window_spec = W.partitionBy("City")

df.select(
    'Name', 'Age', 'City',
    F.col('date_of_birth').alias('DOB'),
    F.avg('Age').over(window_spec).alias('avd_age')
    ).show()

 
# F.collect_list(col) returns all values as an array within each group
# F.collect_set(col) similar as F.collect_list(col) but no duplicates.
df_ = df.groupBy('City').agg(F.collect_set('Name').alias('resident_names'))
df_.printSchema()
df_.show()
df_ = df.groupBy('City').agg(F.collect_list('Name').alias('resident_names'))
df_.printSchema()
df_.show()



# Always clean up
# try:
#     pass
# finally:
#     spark.stop()

# DataFrame.repartition(numPartitions, *cols) Returns a new DataFrame partitioned by the given partitioning expressions. The resulting DataFrame is hash partitioned, Shuffled(a costly operation).

df_ = df.repartition(1)
df_.show()


times_two_udf = F.udf(lambda x: x * 2) # Multiply each row's x value by two
df.withColumn('Agex2', times_two_udf(df.Age)).show()

# Randomly choose a value to use as a row's name
import random

random_name_udf = F.udf(lambda: random.choice(['Bob', 'Tom', 'Amy', 'Jenna']))

df.withColumn('name', random_name_udf()).show() # Caution: It replaces 'Name'?
df.withColumn('Rname', random_name_udf()).show() # 



df.persist(pyspark.storagelevel.StorageLevel.MEMORY_ONLY)

df.unpersist(blocking=True)

logFile = "/opt/spark/spark-4.0.1-bin-hadoop3/README.md"
logData = spark.read.text(logFile).cache()

numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

print("Lines with a: %i, lines with b: %i" % (numAs, numBs))










df.show()

df.collect()    # Returns all the records in the DataFrame as a list of Row
df.take(3)      # Returns the first num rows as a list of Row
df.head(n=3)    # same as take()?
df.toPandas()
df.toArrow()    # pyarrow.Table


spark.stop()
