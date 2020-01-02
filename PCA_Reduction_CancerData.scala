// Databricks notebook source
// MAGIC %md
// MAGIC ###PCA Applied to Breast Cancer Dataset

// COMMAND ----------

// MAGIC %md
// MAGIC ###Loading Dataset

// COMMAND ----------

// DBTITLE 0,Loading Dataset
//Importing Libraries
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

// COMMAND ----------

//Starting a Spark session called "PCA Example"
val spark = SparkSession.builder().appName("PCA_Example").getOrCreate()

// COMMAND ----------

//Reading data
val data = (spark.read.option("header","true")
            .option("inferSchema","true")
            .format("csv")
            .load("/FileStore/tables/BreastCancerData.csv"))

display(data)

// COMMAND ----------

data.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC ###Modelling for PCA

// COMMAND ----------

// Import PCA, VectorAssembler and StandardScaler from ml.feature
import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler}

// Import Vectors from ml.linalg
import org.apache.spark.ml.linalg.Vectors

// COMMAND ----------

val colnames = (Array("id","radius_mean", "texture_mean", "perimeter_mean",
                      "area_mean", "smoothness_mean", "compactness_mean",
                      "concavity_mean", "concave points_mean", "symmetry_mean",
                      "fractal_dimension_mean", "radius_se", "texture_se",
                      "perimeter_se", "area_se", "smoothness_se",
                      "compactness_se", "concavity_se",
                      "concave points_se", "symmetry_se",
                      "fractal_dimension_se", "radius_worst",
                      "texture_worst", "perimeter_worst", "area_worst",
                      "smoothness_worst", "compactness_worst",
                      "concavity_worst", "concave points_worst",
                      "symmetry_worst", "fractal_dimension_worst"))

//Part of performing the operation within PCA
  //i.e. Creating a features column to store the high dimensional data 
val assembler =(new VectorAssembler()
              .setInputCols(colnames)
              .setOutputCol("features"))

// COMMAND ----------

// Transform our DataFrame to a single column: features
val output = assembler.transform(data).select("features")

// COMMAND ----------

// StandardScaler on the data
// Create a new StandardScaler() object called scaler
// Set the input to the features column and the ouput to a column called
// scaledFeatures

val scaler = (new StandardScaler()
              .setInputCol("features")
              .setOutputCol("scaledFeatures")
              .setWithStd(true)
              .setWithMean(false))

// COMMAND ----------

// Summary statistics by fitting the StandardScaler.
// Basically create a new object called scalerModel by using scaler.fit()
// on the output of the VectorAssembler

val scalerModel = scaler.fit(output)

// COMMAND ----------

// Normalize each feature to have unit standard deviation.
// transform() off of this scalerModel object to create scaledData

val scaledData = scalerModel.transform(output)



// COMMAND ----------

// MAGIC %md
// MAGIC ###Applying PCA

// COMMAND ----------

// Create a new PCA() object that will take in the scaledFeatures
// and output the pcs features
val pca = (new PCA()
  .setInputCol("scaledFeatures")
  .setOutputCol("pcaFeatures")
  .setK(10)
  .fit(scaledData))

// COMMAND ----------

// Transform the scaledData
val pcaDF = pca.transform(scaledData)


// COMMAND ----------

display(result)

// COMMAND ----------

val colnames2 = (Array("pcaFeatures"))

val assembler2 =(new VectorAssembler()
              .setInputCols(colnames)
              .setOutputCol("features"))

// COMMAND ----------

// Show the new pcaFeatures
val result = pcaDF.select("pcaFeatures")
result.show()

// only has 4 principal components
result.head()

// COMMAND ----------

display(pcaDF)

// COMMAND ----------

pca.explainedVariance

// COMMAND ----------


