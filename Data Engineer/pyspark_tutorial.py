'''
import findspark
findspark.init()
 
 
from pyspark.sql import SparkSession
#sparksession 드라이버 프로세스 얻기
spark = SparkSession.builder.appName("sample").master("local[*]").getOrCreate()
#클러스터모드의 경우 master에 local[*] 대신 yarn이 들어간다.
spark.conf.set("spark.sql.repl.eagerEval.enabled",True)
#jupyter환경에서만 가능한 config, .show()메소드를 사용할 필요없이 dataframe만 실행해도,정렬된 프린팅을 해준다.
'''

from pyspark.sql import SparkSession # 세션 시작시 import
from pyspark.sql.types import * # 자료형 부여시 import 필요

spark = SparkSession.builder\
        .master("local[*]")\
        .appName('PySpark_Tutorial')\
        .getOrCreate()
        
data = spark.read.csv(
    'C:/Users/USER/Desktop/Data Engineer/spark_tutorial_dataset/fraud_detection_dataset.csv',
    sep = ',',
    header = True
)

#%%

data.printSchema()
data.show(5)

# schema에 자료형을 부여하는 방법.
# 변수명, 변수 타입, null 가능 여부를 기록한다.
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
    'C:/Users/USER/Desktop/Data Engineer/spark_tutorial_dataset/fraud_detection_dataset.csv',
    sep = ',',
    header = True,
    schema = final_struc
)
data.printSchema()

# 데이터 첫 N개 보기
data.show(3)
# 데이터 행마다 통계 내기
data.describe().show()
# 컬럼 명 맽기
data.columns
# 데이터 개수 세기
data.count()

# data의 기본적인 편집
data = data.withColumn('new_timestamp', data.timestamp)
data = data.withColumnRenamed('user_id', 'UID')
data = data.drop('location')
data.show(5)

# data의 querying
data.select(['is_fraud', 'age', 'income', 'debt']).describe().show()
data.groupby('device_type').count().show()

from pyspark.sql.functions import col, lit, when, avg # filter을 위해 필요.
# 특정 컬럼의 값이 어떻게 되나를 체크하는 용도, show를 붙여야 함을 잊어서는 안된다.
data.filter((col('credit_score')) <= lit(300)).show(5)
# case when 역할을 하는 쿼리임. when도 import 붙여야함.
# alias를 붙여 컬럼명 변경 가능.
data.select('income', 'debt', 'is_fraud', when(data.credit_score>= 500, "high").otherwise("low").alias("c_score_high_low")).show(5)
# 통계량 계산하기
data.groupby('is_fraud').agg(
    avg("debt").alias("AvgDebt"),
    avg("income").alias("AvgIncome")   
).show(5)