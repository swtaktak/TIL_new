# pyspark로 스케일링/인코딩 해보기
from pyspark.sql import SparkSession # 세션 시작시 import
from pyspark.sql.types import * # 스키마에 자료형 부여시 import 필요
from pyspark.ml.feature import StringIndexer # 문자열을 인덱스로 적용하기
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StandardScaler
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

# encoding 해보기 - device_type에 대해서 작업하기
indexer = StringIndexer(inputCol='device_type', outputCol='device_ind')
en_df = indexer.fit(data).transform(data)
en_df.show(5)

# onehot encoder
# 주의사항 : pyspark에서는 사전 작업인 stringindexer를 해야만 가능하다.
# 내부적으로 문자열을 받아주지 않아...
# dropLast = '마지막 벡터를 드랍할 것인가?' 마지막을 드랍해도 구분이 됨.
oh_en = OneHotEncoder(inputCol='device_ind', outputCol='device_ohe',
                      dropLast = True)
oh_df = oh_en.fit(en_df).transform(en_df)

oh_df.show(5)


# Scaler. 
# pandas에서는 바로 해도 되지만, pyspark에서는 Feature vectorization 되어야함
# VectorAssembler를 하고 진행해야 한다.
ftr_columns = ['amount', 'income', 'debt', 'credit_score']
vec_assembler = VectorAssembler(inputCols=ftr_columns, outputCol='feature')
data_vec = vec_assembler.transform(data)

# True, True일 경우 표준정규분포를 기준으로 계산
# False로 할 경우 데이터들의 평균치로 적용된다.
scaler = StandardScaler(inputCol='feature', outputCol='feature_scaled',
                        withMean=True, withStd=True)
data_scaled = scaler.fit(data_vec).transform(data_vec)
data_scaled.show(5)