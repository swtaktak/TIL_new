'''
# 최초 단계에 spark가 잘 설치되었는지 확인 용도
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
from pyspark.sql.types import * # 스키마에 자료형 부여시 import 필요

# SparkSession이란, 스파크 응용 프로그램의 통합 진입점.
spark = SparkSession.builder\
        .master("local[*]")\
        .appName('PySpark_Tutorial')\
        .getOrCreate()

'''
# 기본적으로 데이터를 불러오는 방법
# 그러나 해당 평태로 불러올 경우        
data = spark.read.csv(
    'C:/Users/USER/Desktop/TIL_new/Data Engineer/spark_tutorial_dataset/fraud_detection_dataset.csv',
    sep = ',',
    header = True
)
'''
#%%
# Remark. 스파크의 기본 데이터 구조는 RDD이다. (Resilient Destributed Dataset)
# Map reduce는 파일 스토리지 시스템에 저장하고 읽는 구조.
# RDD는 메모리 내 연산 지원. 


# schema에 자료형을 부여하는 방법으로 변수명, 변수 타입, null 가능 여부를 기록한다.
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

"""
# 데이터 첫 N개 보기
data.show(3)
# 데이터 행마다 통계 내기
data.describe().show()
# 컬럼 명 맽기
data.columns
# 데이터 개수 세기
data.count()
"""

# data의 기본적인 편집
# 주의사항 : pandas와 다르게, rdd는 immutable 하다.
# 즉 하나의 데이터를 원 상태로 보존하면서 계속해서 새로운걸 만들 수 있다.
data2 = data.withColumn('new_timestamp', data.timestamp) # 컬럼 복사, 뒤에서 data 부르면 안바뀌어있음 immutable해서.
data2 = data2.withColumnRenamed('user_id', 'UID') # 컬럼명 변경
data2 = data2.drop('location') # 컬럼 drop
data2.show(5)

# data의 querying : 실제 SQL 문법과 유사함.
data.select(['is_fraud', 'age', 'income', 'debt']).describe().show()
data.groupby('device_type').count().show()

# data filtering 및 간단한 전처리/파생 변수 생성하는 방법.
from pyspark.sql.functions import col, lit, when, avg # filter을 위해 필요.
# 특정 컬럼의 값이 어떻게 되나를 체크하는 용도, show를 붙여야 함을 잊어서는 안된다.
data.filter((col('credit_score')) <= lit(300)).show(5)

# sql에서 case when 역할을 하는 쿼리임. when도 import 붙여야함. alias를 붙여 컬럼명 변경 가능.
data.select('income', 'debt', 'is_fraud', when(data.credit_score>= 500, "high").otherwise("low").alias("c_score_high_low")).show(5)
# 통계량 계산하기(aggregation은 groupby로)
data.groupby('is_fraud').agg(
    avg("debt").alias("AvgDebt"),
    avg("income").alias("AvgIncome")   
).show(5)
# 특정 칼럼 값을 기준으로 정렬하기. 오름차순은 ascending으로! 빚이 가장적은 5명을 출력하기.
data.orderBy("debt", ascending = True).show(5)
