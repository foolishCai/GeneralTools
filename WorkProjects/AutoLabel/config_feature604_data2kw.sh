#!/bin/bash

input_hdfs=${1}
output_hdfs=${2}
output_local=${3}

/opt/spark23/bin/spark-submit --master yarn --deploy-mode client  --conf spark.sql.hive.caseSensitiveInferenceMode=NEVER_INFER --conf spark.dynamicAllocation.enabled=false /home/chenyw_yzx/AutoLabel/config_feature605_data2kw.py ${input_hdfs} ${output_hdfs}

hadoop fs -getmerge ${output_hdfs}  ${output_local}
