#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 09:55
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : AddressCheckUtil.py
# @Note    : 离线地址核验工具


import numpy as np
import pandas as pd

from AddressBase import *


class AddressCheck(object):

    def __init__(self, df, ak):
        """
        传入的df：默认四列
        第一列，number，唯一主键
        第二列：id，gid or pn-md5
        第三列：客户提供地址，大多为中文
        第四列：我司库内地址，可能是geohash，可能是中文
        """
        self.df = self.check_df(df)
        self.ak = ak

    def check_df(self, df):

        blank_cols = [i for i in list(df) if df[i].astype(str).str.findall(r'^\s*$').apply(lambda x:0 if len(x) == 0 else 1).sum() > 0]
        if len(blank_cols) == 0:
            self.df = df
        else:
            self.df = df.drop(blank_cols, axis=1)
        if len(df.columns) != 4:
            print("不符合文件格式，OUT！")
            sys.exit(0)

        df.columns = ["number", "id", "addr1", "addr2"]
        if df[df.addr1.isnull()].any().any():
            print("第三列应为客户提供对比地址，有空，OUT！")
            sys.exit(0)

        return df


    def main(self):
        total_test = self.df.shape[0]
        self.df = self.df[~self.df.addr2.isnull()]
        print("需要比对的地址共{}条，其中库内可查得地址{}条，有{}条样本id存在重复".format(total_test, self.df.shape[0],
                                                                     (self.df.shape[0]-len(self.df.id.unique()))))
        print(">>>>>>>>>>>> 地址规整中")
        # step0: 判断是否为中文
        # step1.1: 若是中文地址，利用百度接口返回经纬度(百度坐标系)，返回结果：经度，维度
        # step1.2: 若不是中文地址，则默认为geohash，直接geohash.decode(WGS84坐标系)
        # step2: 将step1.1中的经纬度转为(WGS84坐标系)
        self.df["addr1_is_chi"] = self.df.addr1.map(lambda x: is_chi_word(x))
        self.df["addr1_format"] = self.df.apply(lambda x: get_chi2ll(x["addr1"], self.ak) if x["addr1_is_chi"] else get_geo2ll(x["addr1"]), axis=1)
        self.df["addr1_format"] = self.df.apply(lambda x: bd09_to_wgs84(x["addr1_format"][0], x["addr1_format"][1]) if x["addr1_is_chi"] and x["addr1_format"][0]!=-1 and  x["addr1_format"][1]!=-1 else x["addr1_format"], axis=1)

        self.df["addr2_is_chi"] = self.df.addr2.map(lambda x: is_chi_word(x))
        self.df["addr2_format"] = self.df.apply(
            lambda x: get_chi2ll(x["addr2"], self.ak) if x["addr2_is_chi"] else get_geo2ll(x["addr2"]), axis=1)
        self.df["addr2_format"] = self.df.apply(
            lambda x: bd09_to_wgs84(x["addr2_format"][0], x["addr2_format"][1]) if x["addr2_is_chi"] and x["addr2_format"][0]!=-1 and  x["addr2_format"][1]!=-1 else x["addr2_format"], axis=1)

        self.df = self.df[(self.df.addr1_format.astype(str) != "[-1,-1]") & (self.df.addr2_format.astype(str) != "[-1,-1]")]
        print("可得经纬度地址，即可通过map接口计算地址有{}条".format(self.df.shape[0]))

        print(">>>>>>>>>>>> 距离计算中")
        self.df["distance"] = self.df.apply(lambda x: get_distance(x.addr1_format[0], x.addr1_format[1], x.addr2_format[0], x.addr2_format[1]), axis=1)
        self.df["same_city"] = self.df.apply(
            lambda x: -1 if x.distance < 50000 else is_same_city(x.addr1_format[0], x.addr1_format[1], x.addr2_format[0], x.addr2_format[1], self.ak), axis=1)

        self.df["level"] = self.df.apply(lambda x: np.nan if x.distance <= -1 else
                                                1.1 if x.distance < 1000 else
                                                1.2 if x.distance < 2000 else
                                                1.3 if x.distance < 3000 else
                                                2 if x.distance < 5000 else
                                                3 if x.distance < 10000 else
                                                4 if x.distance < 20000 else
                                                5 if x.distance < 30000 else
                                                6 if x.distance < 50000 else
                                                7 if x.same_city == 1 else 8, axis=1)

        print(">>>>>>>>>>>> 统计指标中")
        writer = pd.ExcelWriter("AddressCheck.xlsx")
        self.df["distance"] = self.df.distance.map(lambda x: round(x, 4))
        tmp = self.df[["number", "id", "addr1", "addr2", "distance", "level"]]
        tmp.to_excel(writer, 'AllDetail')

        tmp = tmp[["number", "id", "addr1", "level"]]
        tmp = tmp.drop_duplicates()
        tmp = tmp["level"].groupby([tmp.id, tmp.addr1]).agg(["min"]).reset_index()
        tmp.rename(columns={"min": "level"}, inplace=True)
        tmp.to_excel(writer, 'OutputDetail')

        tmp = tmp["id"].groupby([tmp["level"]]).agg(["count"]).reset_index()
        tmp.rename(columns={"count": "num"}, inplace=True)
        tmp["ratio"] = tmp.num.map(lambda x: round(x/self.df.shape[0], 4))
        tmp["total_ratio"] = tmp.num.map(lambda x: round(x / total_test, 4))
        tmp.to_excel(writer, "OutputReport")
        writer.save()
        writer.close()

        print("""结果已输出至AddressCheck.xlsx中\n
                 所有结果可查sheet_name=AllDetail\n
                 输出明细可查sheet_name=OutputDetail\n
                 输出统计量可查sheet_name=OutputReport""")




# if __name__ == "__main__":
#     df = pd.read_excel("${your_file_path}", ak="95f458da4f90faa19c88ddbb6d249130")
# #     ac = AddressCheck(df, "95f458da4f90faa19c88ddbb6d249130")
# #     ac.main()