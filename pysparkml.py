# Read the data frame
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans 
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,  VectorIndexer 
from pyspark.ml.classification import LogisticRegression

schema = StructType([ 
	StructField("Age", DoubleType(), True),
	StructField("Job", StringType(), True),
	StructField("Marital", StringType(), True), 
	StructField("Education", StringType(), True),
	StructField("Default", StringType(), True),
	StructField("Balance", DoubleType(), True),
	StructField("Housing", StringType(), True),
	StructField("Loan", StringType(), True),
	StructField("Contract", StringType(), True),
	StructField("Day", DoubleType(), True),
	StructField("Month", StringType(), True),
	StructField("Duration", DoubleType(), True),
	StructField("Campaign", DoubleType(), True),
	StructField("Pdays", DoubleType(), True),
	StructField("Previous", DoubleType(), True),
	StructField("Poutcome", StringType(), True),
	StructField("y", StringType(), True)])

df = sqlContext.read.format("com.databricks.spark.csv") \
        .options(delimiter=',')\
        .options( inferschema=True)\
        .options( header=False)\
        .option("timestampFormat", "MM/dd/yyyy HH:mm:ss")\
        .load('file:///home/1292B29/bank/3_bank.csv', schema = schema)

df.registerTempTable("SparkTableTest")
#df_test_new = sqlContext.sql("select * from SparkTableTest")
df_test_new = sqlContext.sql("SELECT IF (Balance<0, 0, Balance) Balance_mod, Age, Job, Marital, Education, Default, Housing, Loan, Contract, Day, Month, Duration, Campaign, IF (Pdays<0, 0, Pdays) Pdays_mod, Previous, Poutcome, IF(y='no', 0, 1) Approved from SparkTableTest") 
pre = sqlContext.sql("SELECT IF(y='no', 0, 1) Approved from SparkTableTest") 
#6
df_test_new.registerTempTable("df_test_new")
df_class_imbalance = sqlContext.sql("SELECT COUNT(*) FROM df_test_new GROUP BY Approved") 

#7
numericalVars = ['Age','Balance_mod','Day','Duration','Campaign','Pdays_mod','Previous'] 
catVars = ['Job', 'Marital','Education','Default','Housing','Loan','Contract','Month','Poutcome']

for x in catVars:
	index = StringIndexer(inputCol = x, outputCol = x + "_index")
	df_test_new = index.fit(df_test_new).transform(df_test_new)
	encoder = OneHotEncoder(inputCol = x + "_index", outputCol = x + "_vec")
	df_test_new = encoder.transform(df_test_new)
df_test_new.show()

mod_features = numericalVars + [x + "_vec" for x in catVars]

assembler = VectorAssembler(inputCols = mod_features, outputCol = 'features')
df_model = assembler.transform(df_test_new)

#8.
# label = pre['Approved']

# How to add label to df_model
df_model = df_model.withColumn('label', pre.Approved)

#9 Logistic Regression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="binomial")
lrModel = lr.fit(train)





