# pyspark로 ML 분류 모델 만들기 따라하기 _ iris spark ver
from pyspark.sql import SparkSession # 세션 시작시 import
from pyspark.sql.types import * # 스키마에 자료형 부여시 import 필요
from pyspark.ml.feature import VectorAssembler # ML 관련 작업시 필요
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
data.printSchema()
data.show(3)

# is_fraud를 예측하는 단순 분류 트리 모델을 우선 만들어보자.
# 우선 tvt split
train_df, test_df = data.randomSplit(weights = [0.8, 0.2], seed = 99)

# Feature Vector로 만들기
ftr_columns = ['amount', 'income', 'debt', 'credit_score']
vec_assembler = VectorAssembler(inputCols=ftr_columns, outputCol='feature')
train_ftr_vec = vec_assembler.transform(train_df)
test_ftr_vec = vec_assembler.transform(test_df)

train_ftr_vec.show(3) # 학습 후의 결과 보기 / 별도의 feature 행이 발생했다.
dt_clf = DecisionTreeClassifier(featuresCol='feature', labelCol='is_fraud',
                                maxDepth=7)
# model fit, pred 과정
dt_model = dt_clf.fit(train_ftr_vec)
train_pred = dt_model.transform(train_ftr_vec)
test_pred = dt_model.transform(test_ftr_vec)

# 정확도를 측정하자.
check = MulticlassClassificationEvaluator(predictionCol='prediction',
                                          labelCol='is_fraud',
                                          metricName='accuracy')
train_acc = check.evaluate(train_pred)
test_acc = check.evaluate(test_pred)
print('Train Acc:', train_acc)
print('Test Acc:', test_acc)