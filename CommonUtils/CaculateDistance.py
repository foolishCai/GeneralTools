#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 09:56
# @Author  : Cai
# @File    : CaculateDistance.py
# @Note    : 经纬度计算距离，包括调用百度接口经纬度化


import requests
import json
import numpy as np
import geohash
import pandas as pd
import os
import sys

'''
函数功能：根据传入的比较模糊的地址获取精确的结构化地址
函数实现流程：利用百度地图api的正地理编码可以获取该位置的经纬度
           接着使用经纬度采用逆地址编码获取结构化地址
'''



class get_distance(object):

    def __init__(self, file_path):
        # self.file_path = "/Users/cai/Desktop/address_test.xlsx"
        self.file_path = file_path
        df = pd.read_excel(self.file_path)
        df.columns = ["id", "addr1", "addr2"]
        blank_cols = [i for i in list(df) if df[i].astype(str).str.findall(r'^\s*$').apply(lambda x:0 if len(x)==0 else 1).sum()>0]
        if len(blank_cols) == 0:
            self.df = df
        else:
            self.df = df.drop(blank_cols, axis=1)

    def get_formatted_address(self):
        self.df["addr1_format"] = self.df.addr1.map(lambda x: self.get_ll_map(x) if self.is_Chinese(x) else self.get_ll_geo(x))
        self.df["addr2_format"] = self.df.addr2.map(lambda x: self.get_ll_map(x) if self.is_Chinese(x) else self.get_ll_geo(x))

    def get_result(self):
        self.df["distance"] = self.df.apply(lambda x: self.get_map_Distance(x.addr1_format, x.addr2_format) if x.addr1_format!="-1,-1" and x.addr2_format!="-1,-1"
                                            else -1, axis=1)
        self.df["same_city"] = self.df.apply(lambda x: -1 if x.distance < 50000 else self.get_same_city(x.addr1_format, x.addr2_format), axis=1)

    def get_level(self):
        self.df["level"] = self.df.apply(lambda x: 1.1 if x.distance < 1000  else
                                                   1.2 if x.distance < 2000  else
                                                   1.3 if x.distance < 3000  else
                                                   2   if x.distance < 5000  else
                                                   3   if x.distance < 10000 else
                                                   4   if x.distance < 20000 else
                                                   5   if x.distance < 30000 else
                                                   6   if x.distance < 50000 else
                                                   7   if x.same_city == 1 else 8, axis=1)

    def output(self):
        self.df["distance"] = self.df.distance.map(lambda x: round(x, 4))
        self.df = self.df[["id", "addr1", "addr2", "distance", "level"]]
        path = "/Users/cai/Desktop/address_test.xlsx".split(os.sep)
        output_file = os.sep.join(path[:-1]) + os.sep + "level_ouput.xlsx"
        self.df.to_excel(output_file, index=False)


    @ staticmethod
    def is_Chinese(word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    @ staticmethod
    def get_ll_map(address):
        ak = "你的key"

        # 根据百度地图api接口获取正地址编码也就是经纬度
        flag = 3
        url = 'http://api.map.baidu.com/geocoding/v3/?address=' + address + '&output=json&ak={}&callback=showLocation'.format(
            ak)
        lng, lat = -1, -1
        while flag:
            try:
                resp = requests.get(url)
                resp_str = resp.text
                resp_str = resp_str.replace('showLocation&&showLocation', '')
                resp_str = resp_str[1:-1]
                resp_json = json.loads(resp_str)
                location = resp_json.get('result').get('location')
                lng, lat = location.get('lng'), location.get('lat')
                flag = 0
                break
            except:
                flag = flag - 1
                lng, lat = -1, -1
        return ",".join([str(lng), str(lat)])

    @ staticmethod
    def get_ll_geo(address):
        try:
            lat, lng = geohash.decode(address)
        except:
            lat, lng = "-1", "-1"
        return ",".join([str(lng), str(lat)])

    @ staticmethod
    def get_map_Distance(addr1, addr2):
        lng1, lat1 = float(addr1.split(",")[0]), float(addr1.split(",")[1])
        lng2, lat2 = float(addr2.split(",")[0]), float(addr2.split(",")[1])
        R = 6378137
        lat1 = lat1 * np.pi / 180.0
        lat2 = lat2 * np.pi / 180.0
        a = lat1 - lat2
        b = (lng1 - lng2) * np.pi / 180.0
        sa2 = np.sin(a / 2.0)
        sb2 = np.sin(b / 2.0)
        d = 2 * R * np.sin(np.sqrt(sa2 * sa2 + np.cos(lat1) * np.cos(lat2) * sb2 * sb2))
        return d

    @ staticmethod
    def get_same_city(addr1, addr2):
        ak = "你的key"

        lng1, lat1 = float(addr1.split(",")[0]), float(addr1.split(",")[1])
        lng2, lat2 = float(addr2.split(",")[0]), float(addr2.split(",")[1])

        flag = 3
        return_result = 0
        while flag:
            try:
                url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak={}&output=json&coordtype=wgs84ll&location='.format(
                    ak) + str(lat1) + ',' + str(lng1) + ''
                resp = requests.get(url)
                resp_json = json.loads(resp.text)
                # 提取结构化地址
                # formattted_address = resp_json.get('result').get('formatted_address')
                # province = resp_json["result"]["addressComponent"]["province"]
                city1 = resp_json["result"]["addressComponent"]["city"]
                # district = resp_json["result"]["addressComponent"]["district"]

                url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak={}&output=json&coordtype=wgs84ll&location='.format(
                    ak) + str(lat2) + ',' + str(lng2) + ''
                resp = requests.get(url)
                resp_json = json.loads(resp.text)
                city2 = resp_json["result"]["addressComponent"]["city"]
                return_result = (city1 == city2)
                flag = 0
                break
            except:
                flag = flag-1
                return_result = 0
        return int(return_result)


if __name__ == "__main__":
    file_path = sys.argv[1]

    gd = get_distance(file_path)
    gd.get_formatted_address()
    gd.get_result()
    gd.get_level()
    gd.output()