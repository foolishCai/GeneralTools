Cac#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/5 09:56
# @Author  : Cai
# @File    : CaculateDistance.py
# @Note    : 经纬度计算距离，包括调用百度接口经纬度化


import requests
import json
import numpy as np
'''
函数功能：根据传入的比较模糊的地址获取精确的结构化地址
函数实现流程：利用百度地图api的正地理编码可以获取该位置的经纬度
           接着使用经纬度采用逆地址编码获取结构化地址
'''


ak = "95f458da4f90faa19c88ddbb6d249130"


def get_lng_lat(address):
    # 根据百度地图api接口获取正地址编码也就是经纬度
    url = 'http://api.map.baidu.com/geocoding/v3/?address=' + address + '&output=json&ak={}&callback=showLocation'.format(ak)

    # 获取经纬度
    resp = requests.get(url)
    resp_str = resp.text
    resp_str = resp_str.replace('showLocation&&showLocation', '')
    resp_str = resp_str[1:-1]
    resp_json = json.loads(resp_str)
    location = resp_json.get('result').get('location')
    lng, lat = location.get('lng'), location.get('lat')
    return lng, lat


def get_formatted_address(address, return_type):
    lng, lat = get_lng_lat(address)
    url = 'http://api.map.baidu.com/reverse_geocoding/v3/?ak={}&output=json&coordtype=wgs84ll&location='.format(ak) + str(lat) + ',' + str(lng) + ''
    resp = requests.get(url)

    resp_json = json.loads(resp.text)
    # 提取结构化地址
    formattted_address = resp_json.get('result').get('formatted_address')
    province = resp_json["result"]["addressComponent"]["province"]
    city = resp_json["result"]["addressComponent"]["city"]
    district = resp_json["result"]["addressComponent"]["district"]
    result = eval(return_type)
    return result


def get_map_Distance(lng1, lat1, lng2, lat2):
    R = 6378137
    lat1 = lat1 * np.pi / 180.0
    lat2 = lat2 * np.pi / 180.0
    a = lat1 - lat2
    b = (lng1 - lng2)*np.pi/180.0
    sa2 = np.sin(a / 2.0)
    sb2 = np.sin(b / 2.0)
    d = 2 * R * np.sin(np.sqrt(sa2 * sa2 + np.cos(lat1) * np.cos(lat2) * sb2 * sb2))
    return d