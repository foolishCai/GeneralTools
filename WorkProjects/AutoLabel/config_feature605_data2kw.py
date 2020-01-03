#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/24 11:40
# @Author  : Cai
# @File    : config_feature605_data2kw.py
# @Note    : pyspark


from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import math
import sys


class LabelOutput(object):

    def __init__(self, cus_df_file):

        cus_df = spark.read.csv(cus_df_file, sep='|', header=True)
        self.cus_df = cus_df.toPandas()

        config_df = spark.read.csv('hdfs://hzxs-yzx-hadoop-ds/user/hive/warehouse/chenyw_yzx.db/udf/import_miss.csv',
                                   sep='|',
                                   header=True)
        self.config_df = config_df.toPandas()
        self.config_df['if_continuous'] = self.config_df.if_continuous.astype(int)
        self.config_df['min_value'] = self.config_df.min_value.astype(float)
        self.config_df['max_value'] = self.config_df.max_value.astype(float)
        self.other_var = ['number', 'id', 'id_type', 'create_date', 'gid']
        self.blacklist_var = ["ft_gz_black_list", "ft_gz_grey_list"]
        self.continuous_var = [i for i in self.config_df[self.config_df.if_continuous == 1].feature.tolist() if
                               i in self.cus_df.columns]
        self.class_var = [i for i in self.config_df[self.config_df.if_continuous == 0].feature.tolist() if
                          i not in self.blacklist_var and i not in self.other_var and i in self.cus_df.columns]

        # 对大盘数据进行月份比例加减乘除

    def get_gz_result(self):
        self.cus_df["month"] = self.cus_df["create_date"].map(lambda x: str(x)[:7].replace('-', ''))
        month_dict = self.cus_df["month"].value_counts().to_dict()
        month_dict = {k: round(v / len(self.cus_df), 2) for k, v in month_dict.items()}
        monthList = ",".join([str(k) for k in month_dict])
        sql = "select * from chenyw_yzx.autolabel_map where month in ({})".format(monthList)
        self.gz_df = spark.sql(sql).toPandas()
        self.gz_df["tag"] = self.gz_df.tag.astype(str)
        self.gz_df["cnt"] = self.gz_df.cnt.astype(int)

        # 处理'ft_dev_phone_brand', 'ft_dev_phone_model'
        tmp_df = self.gz_df[self.gz_df.feature.isin(['ft_dev_phone_brand', 'ft_dev_phone_model'])]
        tmp_df['tag'] = tmp_df.apply(lambda x: 'other' if x.tag not in self.config_df[self.config_df.feature == x.feature].tag.tolist() else x, axis=1)

        self.bin_result = pd.DataFrame()

        bin_ratio = self.gz_df['cnt'].groupby(by=[self.gz_df['month']]).agg(['sum'])
        bin_ratio['month'] = bin_ratio.index
        bin_ratio['total_cnt'] = bin_ratio['sum'].map(lambda x: x / 605)
        bin_ratio.index = [i for i in range(len(bin_ratio))]
        self.gz_df = pd.merge(self.gz_df, bin_ratio[['month', 'total_cnt']], how='left', on='month')
        self.gz_df['ratio'] = self.gz_df.apply(lambda x: round(month_dict[str(x.month)] * x.cnt / x.total_cnt, 4),
                                               axis=1)
        gz_bin_result = pd.pivot_table(self.gz_df, index=["feature", "tag"], values=["cnt", "ratio"], aggfunc=[sum])
        gz_bin_result.columns = ['gz_cnt', 'gz_ratio']
        gz_bin_result['feature'] = [i[0] for i in gz_bin_result.index]
        gz_bin_result['tag'] = [i[1] for i in gz_bin_result.index]
        gz_bin_result.index = [i for i in range(gz_bin_result.shape[0])]
        self.bin_result = gz_bin_result[['feature', 'tag', 'gz_cnt', 'gz_ratio']]

    # 对甲方爸爸数据进行处理
    def get_cus_result(self):
        for c in self.blacklist_var:
            self.cus_df[c] = self.cus_df[c].map(lambda x: 'blank' if x is None or x != x or x == '' or x == 'None' else 1)
        for c in self.class_var:
            self.cus_df[c] = self.cus_df[c].map(lambda x: 'blank' if x is None or x != x or x == '' or x == 'None' else x)
            # 甲方爸爸的离散变量tag值不一定存在于2KW中, 对应的woe为正无穷，iv肯定为负无穷
            cus_tag_list = self.cus_df[c].unique().tolist()
            gz_tag_list = self.gz_df[self.gz_df.feature == c].tag.tolist()
            delta_tag = list(set(cus_tag_list) - set(gz_tag_list))
            if len(delta_tag) == 0:
                continue
            delta_dict = {
                "feature": [c] * len(delta_tag),
                "tag": delta_tag,
                "gz_cnt": [0] * len(delta_tag),
                "gz_ratio": [0] * len(delta_tag)
            }
            self.bin_result = pd.concat(
                [self.bin_result, pd.DataFrame(data=delta_dict, columns=['feature', 'tag', 'gz_cnt', 'gz_ratio'])])
        for c in self.continuous_var:
            tmp = self.config_df[self.config_df.feature == c]
            cut_points = sorted(list(set(tmp.min_value.tolist() + tmp.max_value.tolist())))
            self.cus_df[c] = self.cus_df[c].map(
                lambda x: float(np.nan) if x is None or x == 'None' or x == 'blank' else float(x))
            self.cus_df.loc[(~self.cus_df[c].isnull()), c] = pd.cut(self.cus_df[~self.cus_df[c].isnull()][c],
                                                                    cut_points, labels=['0', '1', '2', '3', '4'])
            self.cus_df[c].fillna("blank", inplace=True)
            self.cus_df[c] = self.cus_df[c].astype(str)

        self.bin_result['cus_cnt'] = self.bin_result.apply(
            lambda x: len(self.cus_df[self.cus_df[x.feature] == str(x.tag)]), axis=1)
        self.bin_result['gz_cnt'] = self.bin_result['gz_cnt'].astype(int)
        self.bin_result['gz_ratio'] = self.bin_result['gz_ratio'].astype(float)
        self.bin_result = self.bin_result[(self.bin_result.gz_cnt > 0) | (self.bin_result.cus_cnt > 0)]
        self.bin_result['cus_ratio'] = self.bin_result["cus_cnt"].map(lambda x: round(x / len(self.cus_df), 2))
        self.bin_result['delta_ratio'] = self.bin_result.apply(
            lambda x: float("inf") if x["gz_cnt"] == 0 or x["gz_ratio"] == 0 else round(
                (x["cus_ratio"] - x["gz_ratio"]) / x["gz_ratio"], 4), axis=1)

    # 计算WOE
    def get_woe_iv(self):
        self.bin_result['woe'] = self.bin_result.apply(
            lambda x: float("-inf") if x["cus_ratio"] == 0 else float("inf") if x["gz_ratio"] == 0 else round(
                math.log(x["cus_ratio"] / x["gz_ratio"]), 4), axis=1)
        self.bin_result["iv"] = self.bin_result.apply(lambda x: float("inf") if x.woe == float("-inf") else
        float("-inf") if x.woe == float("inf") else (x.cus_ratio - x.gz_ratio) * x.woe, axis=1)
        self.bin_result = pd.concat(
            [self.bin_result[~self.bin_result.iv.isin(["inf", "-inf"])].sort_values(by='iv', ascending=False),
             self.bin_result[self.bin_result.iv.isin(["inf", "-inf"])]])
        self.bin_result = self.bin_result.sort_values(by='iv', ascending=False)
        miss_df = self.config_df[['feature', 'chinese_name']].drop_duplicates()
        self.bin_result = pd.merge(self.bin_result, miss_df[['feature', 'chinese_name']], how='left', on='feature')
        self.bin_result = self.bin_result[
            ["feature", "chinese_name", "tag", "gz_cnt", "gz_ratio", "cus_cnt", "cus_ratio", "delta_ratio", "woe",
             "iv"]]
        self.bin_result.columns = ["特征", "中文名称", "属性", "随机样本数量", "随机样本占比", "客户样本数量", "客户样本占比", "差异比", "WOE", "IV"]
        self.bin_result = pd.concat(
            [self.bin_result[~self.bin_result.IV.isin(["inf", "-inf"])].sort_values(by='IV', ascending=False),
             self.bin_result[self.bin_result.IV.isin(["inf", "-inf"])]])

    def main(self):
        self.get_gz_result()
        self.get_cus_result()
        self.get_woe_iv()



if __name__ == "__main__":
    spark = SparkSession.builder.appName("config_feature605_data2kw").getOrCreate()
    cus_df_file = str(sys.argv[1])
    output_file = str(sys.argv[2])
    lo = LabelOutput(cus_df_file)
    lo.main()
    spark_df = spark.createDataFrame(lo.bin_result)
    flag = 3
    while flag > 0:
        try:
            spark_df.write.format('csv').mode('overwrite').option('sep', '|').option('header', True).option("encoding", "utf-8").save(output_file)
            sys.exit(0)
        except:
            flag = flag-1
            pass