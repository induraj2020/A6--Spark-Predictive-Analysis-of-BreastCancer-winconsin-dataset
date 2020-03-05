# Databricks notebook source
# MAGIC %md ##![Spark Logo Tiny](https://s3-us-west-2.amazonaws.com/curriculum-release/images/105/logo_spark_tiny.png) Breast Cancer Dataset Analysis By Induraj Ramamurthy

# COMMAND ----------

# MAGIC %md ##### To check if the spark session already exists, if exists then use the same, else create new session 
# MAGIC * ** checking for the spark context version **
# MAGIC * ** Location of dataset specified (line 6) **

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Breast_Cancer_Pyspark_By_IndurajPR').getOrCreate()
import pyspark.sql.functions as F
from pyspark.sql.types import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import PCA
!pip install mlflow
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# COMMAND ----------

# MAGIC %md #### Load dataset into dataframe 
# MAGIC * ** Print schema **

# COMMAND ----------

print(spark)
print(sc)
print(sc.version)
filelocation = "/FileStore/tables/data.csv"

# COMMAND ----------

df= (spark.read
          .option('header','true')
          .option('delimiter',",")
          .csv(filelocation)
    )
print(df.printSchema())


# COMMAND ----------

# MAGIC %md #### Data Transformations
# MAGIC * ** Covert schema ** 
# MAGIC * ** Converting the Dependent Variable Type**
# MAGIC * ** Dropping Unnecessary column "_c32" **

# COMMAND ----------

from pyspark.sql.types import *

csvSchema = StructType([
  StructField("id", IntegerType(), True),
  StructField("diagnosis", StringType(), True),
  StructField("radius_mean", DoubleType(), True),
  StructField("texture_mean", DoubleType(), True),
  StructField("perimeter_mean", DoubleType(), True),
  StructField("area_mean", DoubleType(), True),
  StructField("smoothness_mean", DoubleType(), True),
  StructField("compactness_mean", DoubleType(), True),
  StructField("concavity_mean", DoubleType(), True),
  StructField("concave points_mean", DoubleType(), True),
  StructField("symmetry_mean", DoubleType(), True),
  StructField("fractal_dimension_mean", DoubleType(), True),
  StructField("radius_se", DoubleType(), True),
  StructField("texture_se", DoubleType(), True),
  StructField("perimeter_se", DoubleType(), True),
  StructField("area_se", DoubleType(), True),
  StructField("smoothness_se", DoubleType(), True),
  StructField("compactness_se", DoubleType(), True),
  StructField("concavity_se", DoubleType(), True),
  StructField("concave points_se", DoubleType(), True),
  StructField("symmetry_se", DoubleType(), True),
  StructField("fractal_dimension_se", DoubleType(), True),
  StructField("radius_worst", DoubleType(), True),
  StructField("texture_worst", DoubleType(), True),
  StructField("perimeter_worst", DoubleType(), True),
  StructField("area_worst", DoubleType(), True),
  StructField("smoothness_worst", DoubleType(), True),   
  StructField("compactness_worst", DoubleType(), True),
  StructField("concavity_worst", DoubleType(), True),
  StructField("concave points_worst", DoubleType(), True),
  StructField("symmetry_worst", DoubleType(), True),
  StructField("fractal_dimension_worst", DoubleType(), True),
  StructField("_c32", StringType(), True)
   ])

# COMMAND ----------

df= (spark.read
          .option('header','true')
          .option('delimiter',",")
          .schema(csvSchema)
          .csv(filelocation)
          .drop('_c32')
    )
display(df)

# COMMAND ----------

# MAGIC %md #### File import performance and Parquet
# MAGIC * Time taken while performing actions with CSV file (without cache and With cache)
# MAGIC * Time taken with Parquet file (without cache and With cache)
# MAGIC * Converting dataframe to parquet 
# MAGIC * Needs the special characters to be removed from column names
# MAGIC * Save the data frame as parquet file

# COMMAND ----------

import time
total_uncached = df.count()
total_cached = (df.cache().count())
start_time = time.time()
print("Uncached scenario- counting {} rows took {} seconds".format(df.count(), time.time()-start_time))
start_time = time.time()
print("Cached scenario- counting {} rows took {} seconds".format(df.cache().count(), time.time()-start_time))

# COMMAND ----------

no_id_null= df.filter("id is null").count()
if no_id_null>0:
  df = df.filter(df.id.isNull())
if df.count() > df.dropDuplicates().count():
  print('Dataset has duplicates')
  df = df.dropDuplicates()
else:
  print('Dataset has no duplicates')
  
print("Total Number of rows of records: {}".format(df.count()))
print("Total Number of Unique patients: {}".format(df.select(['id']).distinct().count()))

# COMMAND ----------

# MAGIC %md #### Checking the driver and Workers
# MAGIC * ** Performing Descriptive Analyysis ** *

# COMMAND ----------

app_name = spark.conf.get('spark.app.name')
num_partitions = spark.conf.get('spark.sql.shuffle.partitions')
print("Name : \t {}".format(app_name))
print("Number of partitions :\t {}".format(num_partitions))
print("The Number of partitions in which dataframe resides: {}".format(df.rdd.getNumPartitions()))

# COMMAND ----------

df_cols_to_describe = [cols for cols in df.columns if cols not in ['id','diagnosis']]
display(df[df_cols_to_describe].describe())

# COMMAND ----------

df = df.withColumn("nlabel",F.when(df.diagnosis=='M',F.lit(1)).otherwise(F.lit(0)))

# COMMAND ----------

# for c in df.columns:
#     df = df.withColumnRenamed(c, c.replace(" ", ""))
# df.write.parquet("breast_cancer01.parquet")

# COMMAND ----------

df = spark.read.parquet("breast_cancer01.parquet")

# COMMAND ----------

import time
total_uncached = df.count()
total_cached = (df.cache().count())
start_time = time.time()
print("Uncached scenario- counting {} rows took {} seconds".format(df.count(), time.time()-start_time))
start_time = time.time()
print("Cached scenario- counting {} rows took {} seconds".format(df.cache().count(), time.time()-start_time))
print("-------------------------------------------------------")


# COMMAND ----------

# MAGIC %md #### Exploratory Data Analysis

# COMMAND ----------

display(df)

# COMMAND ----------

display(df)

# COMMAND ----------

feature_columns= ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concavepoints_mean',
 'symmetry_mean',
 'fractal_dimension_mean']

# COMMAND ----------

# MAGIC %md ####Join plot

# COMMAND ----------

pandas_df1 = df.toPandas()
import numpy as np
fig_dims = (8, 8)
fig, ax = plt.subplots(nrows=10, ncols=1,figsize= fig_dims)
dataMalignant=pandas_df1[pandas_df1['diagnosis'] =='M']
dataBenign=pandas_df1[pandas_df1['diagnosis'] =='B']

for i in range(10):
  plt.subplot(5, 2, i+1)
  sns.distplot(dataMalignant[feature_columns[i]],bins=100,color='red',label='M')
  sns.distplot(dataBenign[feature_columns[i]],bins=100,color='green',label='B')
  plt.legend()
  plt.title(feature_columns[i],fontsize=8)
plt.tight_layout()
display(fig)

# COMMAND ----------

fig_dims = (8, 10)
fig, ax = plt.subplots(nrows=10, ncols=1,figsize= fig_dims)
for i in range(10):
  plt.subplot(5, 2, i+1)
  sns.boxplot(x=pandas_df1['diagnosis'],y=pandas_df1[feature_columns[i]])
  plt.title(feature_columns[i],fontsize=8)
plt.tight_layout()
display(fig)

# COMMAND ----------

def min_max_scaler(data, cols_to_scale):
  for col in cols_to_scale:
    max_v = data.agg({col:'max'}).collect()[0][0]
    min_v = data.agg({col:'min'}).collect()[0][0]
    new_col = 'scaled_'+col
    data = data.withColumn(new_col, (df[col]-min_v)/(max_v-min_v))
  return data

# COMMAND ----------

df1 = df
to_omit= ['diagnosis','id']
cols_to_scale= [x for x in df.columns if x not in to_omit]
df1 =min_max_scaler(df1, cols_to_scale)
df1= df1.drop(*cols_to_scale).drop('id')
display(df1)

# COMMAND ----------

pandas_df1= df1.toPandas()
features = cols_to_scale
x = df1.drop(*to_omit).toPandas()
y=  pandas_df1['diagnosis']

data_violinplot_fin=[]
data_violinplot0 = pd.concat([y,x.iloc[:,1:10]], axis=1)
data_violinplot0 = pd.melt(data_violinplot0, id_vars='diagnosis', var_name='features', value_name='value')
data_violinplot1 = pd.concat([y,x.iloc[:,20:29]], axis=1)
data_violinplot1 = pd.melt(data_violinplot1, id_vars='diagnosis', var_name='features', value_name='value')

data_violinplot_fin.append(data_violinplot0)
data_violinplot_fin.append(data_violinplot1)

fig_dims = (15, 10)
fig, ax = plt.subplots(nrows=1, ncols=2,figsize= fig_dims) 

for i in range(2):
  plt.subplot(1,2,i+1)
  ax = sns.violinplot(x='value',y='features',hue='diagnosis',split=True, inner="quart",data=data_violinplot_fin[i])
plt.tight_layout()
display(fig)

# COMMAND ----------

data_swarmplot1 = pd.concat([y,x.iloc[:,:10]], axis=1)
data_swarmplot1 = pd.melt(data_swarmplot1, id_vars='diagnosis', var_name='features', value_name='value')
fig_dims = (10, 5)
fig, ax = plt.subplots(figsize= fig_dims)
ax = sns.swarmplot(x="features",y="value",hue="diagnosis",data=data_swarmplot1)
plt.xticks(rotation=90)
plt.legend(loc='upper right')
display(fig)


# COMMAND ----------

data_swarmplot2 = pd.concat([y,x.iloc[:,11:20]], axis=1)
data_swarmplot2 = pd.melt(data_swarmplot2, id_vars='diagnosis', var_name='features', value_name='value')
fig_dims = (10, 5)
fig, ax = plt.subplots(figsize= fig_dims)
ax = sns.swarmplot(x="features",y="value",hue="diagnosis",data=data_swarmplot2)
plt.xticks(rotation=90)
plt.legend(loc='upper right')
display(fig)

# COMMAND ----------

from pyspark.mllib.stat import Statistics
to_drop = ['id','diagnosis']
df2 = df1.drop(*to_drop)
col_names = df2.columns
features = df2.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

# COMMAND ----------

fig_dims = (10, 8)
fig, ax = plt.subplots(figsize= fig_dims)
ax = sns.heatmap(corr_df)
display(fig)

# COMMAND ----------

fig, ax2 = plt.subplots()
ax2= corr_df['scaled_nlabel'].sort_values().plot(kind='bar',sort_columns=True)
display(ax2)

# COMMAND ----------

# MAGIC %md ##Pipelining Different Stages 

# COMMAND ----------

cols_to_drop =['id','diagnosis']

df = df1.drop(*cols_to_drop)

df_train, df_test = df.randomSplit([0.7,0.3],seed=13)

stringIndexer= StringIndexer(inputCol='scaled_nlabel',outputCol='label')

Vc_assembler = VectorAssembler(inputCols = df.columns, 
                               outputCol='non_reduced_features')

scaler = StandardScaler(inputCol="non_reduced_features", 
                       outputCol="scaled_features",
                       withStd=True, withMean=True)

pca = PCA(k=2, inputCol='scaled_features',outputCol="features")

pipeline0= Pipeline(stages=[stringIndexer,Vc_assembler,scaler,pca])
pipeline_model = pipeline0.fit(df_train)
df_train_1 = pipeline_model.transform(df_train)
pipeline_model = pipeline0.fit(df_test)
df_test_1 = pipeline_model.transform(df_test)


# COMMAND ----------

display(df_train_1)

# COMMAND ----------

# MAGIC %md ## Machine Learning Algorithms

# COMMAND ----------

!pip install mlflow
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

algorithm =[]
f1 =[]
Acc =[]
Recall =[]
Precision=[]
AreaUnderRoc = []

# COMMAND ----------

evaluator1 = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction', metricName='f1')
evaluator2 = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction', metricName='accuracy')
evaluator3 = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction', metricName='weightedRecall')
evaluator4 = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction', metricName='weightedPrecision')
evaluator5 = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',metricName='areaUnderROC')

def metrics(Algorithm,prediction):
  algorithm.append(Algorithm)
  f1.append(evaluator1.evaluate(prediction))
  Acc.append(evaluator2.evaluate(prediction))
  Recall.append(evaluator3.evaluate(prediction))
  Precision.append(evaluator4.evaluate(prediction))
  AreaUnderRoc.append(evaluator5.evaluate(prediction)) 
  
  

# COMMAND ----------

# MAGIC %md #### Decision Tree Classifier

# COMMAND ----------

dTree = DecisionTreeClassifier(labelCol='label', featuresCol='features')
pipeline = Pipeline(stages=[dTree])
#evaluator1= BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator1 = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction', metricName='f1')
params = ParamGridBuilder().addGrid(dTree.maxDepth,[2,3,4,5,6,7]).build()
cv = CrossValidator(estimator = pipeline, 
                    estimatorParamMaps=params, 
                    evaluator = evaluator1).setNumFolds(5).setSeed(13)
cv_model = cv.fit(df_train_1)
model_predictions= cv_model.transform(df_test_1)
metrics('DecisionTree',model_predictions)


# COMMAND ----------

# MAGIC %md #### Logistic Regression

# COMMAND ----------

lr = LogisticRegression(labelCol='label', featuresCol='features')
params =  ParamGridBuilder() \
         .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
         .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
         .build()        
pipeline1 = Pipeline(stages=[lr])
cv = CrossValidator(estimator = pipeline1,
                   estimatorParamMaps=params,
                   evaluator = evaluator1).setNumFolds(5).setSeed(13)
cv_model = cv.fit(df_train_1)
model_predictions1= cv_model.transform(df_test_1)
metrics('logisticRegression',model_predictions1)


# COMMAND ----------

# MAGIC %md #### Gradient Boosted Tree Classifier

# COMMAND ----------

gbt = GBTClassifier(labelCol='label', featuresCol='features')
params =  ParamGridBuilder()\
          .addGrid(gbt.maxDepth, [2, 5, 7])\
          .build()
pipeline3 = Pipeline(stages=[gbt])

cv = CrossValidator(estimator = pipeline3,
                   estimatorParamMaps=params,
                   evaluator = evaluator1).setNumFolds(5).setSeed(13)

cv_model = cv.fit(df_train_1)

model_predictions2= cv_model.transform(df_test_1)

metrics('GradientBoostingTree',model_predictions2)

# COMMAND ----------

# MAGIC %md #### Linear Support Vector Classifier

# COMMAND ----------

from pyspark.ml.classification import LinearSVC
lscv = LinearSVC(labelCol='label', featuresCol='features')
params =  ParamGridBuilder()\
            .addGrid(lscv.regParam,[0.1,0.3,0.5,0.9,1])\
           .build()      #   .addGrid(lscv.fitIntercept,[True,False])\  .addGrid(lscv.aggregationDepth,[2,10,20,50])\
pipeline4 = Pipeline(stages=[lscv])

cv = CrossValidator(estimator = pipeline4,
                   estimatorParamMaps=params,
                   evaluator = evaluator1).setNumFolds(5).setSeed(13)

cv_model = cv.fit(df_train_1)

model_predictions3= cv_model.transform(df_test_1)

metrics('LinearSVC',model_predictions3)

# COMMAND ----------

# MAGIC %md #### Random Forest Classifier

# COMMAND ----------

forest = RandomForestClassifier(labelCol='label', featuresCol='features')
params = ParamGridBuilder().addGrid(forest.featureSubsetStrategy,['all','onethird','sqrt','log2']).addGrid(forest.maxDepth,[2,5,10]).build()

pipeline5 = Pipeline(stages=[forest])
cv = CrossValidator(estimator = pipeline5,
                   estimatorParamMaps=params,
                   evaluator = evaluator1).setNumFolds(5).setSeed(13)

cv_model = cv.fit(df_train_1)

model_predictions4= cv_model.transform(df_test_1)

metrics('RandomForest',model_predictions4)

# COMMAND ----------

# MAGIC %md ### Comparison of Model Performances

# COMMAND ----------

algorithm =[]
f1 =[]
Acc =[]
Recall =[]
Precision=[]
AreaUnderRoc = []

# COMMAND ----------

dataframe = pd.DataFrame(data={'Algorithm':algorithm, 'F1':f1, 'Acc':Acc, 'Recall':Recall, 'Precision':Precision, 'AreaunderRoc':AreaUnderRoc})

# COMMAND ----------

display(dataframe)

# COMMAND ----------

display(dataframe)

# COMMAND ----------


