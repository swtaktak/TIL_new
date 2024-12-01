# ml_tutorial에서 했던 작업에 pipeline을 묻어보자.
# pyspark로 ML 분류 모델 만들기 따라하기 _ iris spark ver
from pyspark.sql import SparkSession # 세션 시작시 import
from pyspark.sql.types import * # 스키마에 자료형 부여시 import 필요
from pyspark.ml.feature import VectorAssembler # ML 관련 작업시 필요
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
# SparkSession이란, 스파크 응용 프로그램의 통합 진입점.
spark = SparkSession.builder\
        .master("local[*]")\
        .appName('PySpark_Tutorial')\
        .getOrCreate()
        
# 앞에서 보았던 데이터로 실습 진행
data_schema = [
    StructField('timestamp', StringType(), True),
    StructField('user_id', StringType(), True),
    StructField('amount', DoubleType(), True),
    StructField('location', StringType(), True),
    StructField('device_type', StringType(), True),
    StructField('is_fraud', IntegerType(), True),
    StructField('age', IntegerType(), True),
    StructField('income', DoubleType(), True),
    StructField('debt', DoubleType(), True),
    StructField('credit_score', IntegerType(), True)
]

final_struc = StructType(fields=data_schema)

data = spark.read.csv(
    'C:/Users/USER/Desktop/TIL_new/Data Engineer/spark_tutorial_dataset/fraud_detection_dataset.csv',
    sep = ',',
    header = True,
    schema = final_struc
)

train_df, test_df = data.randomSplit(weights = [0.8, 0.2], seed = 99)

# 과정을 합쳐서 한번에 적는다.
ftr_columns = ['amount', 'income', 'debt', 'credit_score']
stage_1 = VectorAssembler(inputCols=ftr_columns, outputCol='feature')
stage_2 = DecisionTreeClassifier(featuresCol='feature', labelCol='is_fraud')

# 파이프라인 만들기
pipes = Pipeline(stages = [stage_1, stage_2])
Pipeline_model = pipes.fit(train_df)

train_pred = Pipeline_model.transform(train_df)
test_pred = Pipeline_model.transform(test_df)