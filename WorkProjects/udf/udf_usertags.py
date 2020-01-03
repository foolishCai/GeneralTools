#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
import numpy as np

tag_dict = {
    "age": {
        "011100": "0-17",
        "011200": "18-24",
        "011300": "25-34",
        "011400": "35+",
        "011500": "35-44",
        "011600": "45+"
    },
    "sex": {
        "012100": "female",
        "012200": "male"
    },
    "consumption_level": {
        "014100": "low",
        "014200": "middle",
        "014300": "high"
    },
    "marital_status": {
        "015100": "未婚",
        "015200": "找对象",
        "015300": "已婚"
    },
    "occupation": {
        "016100": "中小学生",
        "016200": "大学生",
        "016300": "求职中",
        "016400": "白领",
        "016600": "教师",
        "016700": "程序员",
        "016800": "医生"
    },
    "finance":{
        "021100": "股票交易",
        "021200": "投资理财",
        "021300": "记账",
        "021400": "银行",
        "021500": "信用卡",
        "021600": "彩票",
        "021700": "网贷p2p"
    },
    "car_ownership":{
        "01c100": "车主"
    }
}

def is_null_value(s):
    if s is None:
        return True
    elif isinstance(s, list):
        if len(s) == 0:
            return True
        else:
            return False
    elif isinstance(s, str):
        if s == '' or len(s) == 0:
            return True
        else:
            return False
    elif isinstance(s, np.nan):
        return True

for line in sys.stdin:
    age, sex, consumtion_level, marital_status, occupation, finance, residence_city, hometown, car_ownership = \
        "", "", "", "", "", [], "", "", ""

    try:
        gid, usertags = line.strip().split('\t')
        usertags = usertags.split(',')

        pattern = re.compile(r'h\d{8}')
        residence_city = [pattern.findall(i) for i in usertags if len(pattern.findall(i)) > 0]
        if is_null_value(residence_city):
            residence_city = '未知'
        else:
            residence_city = residence_city[0][0]

        pattern = re.compile(r'n\d{8}')
        hometown = [pattern.findall(i) for i in usertags if len(pattern.findall(i)) > 0]
        if is_null_value(hometown):
            hometown = '未知'
        else:
            hometown = hometown[0][0]

        for tag in usertags:

            if tag[:3] == '011':
                age = tag_dict['age'][tag]
            elif tag[:3] == '012':
                sex = tag_dict['sex'][tag]
            elif tag[:3] == '014':
                consumtion_level = tag_dict['consumption_level'][tag]
            elif tag[:3] == '015':
                marital_status = tag_dict['marital_status'][tag]
            elif tag[:3] == '016':
                occupation = tag_dict['occupation'][tag]
            elif tag[:3] == '021':
                finance.append(tag_dict["finance"][tag])
            elif tag[:3] == '01c':
                car_ownership = tag_dict["car_ownership"][tag]
        return_line = '\t'.join([gid, age, sex, consumtion_level, marital_status, occupation, finance, residence_city, hometown, car_ownership])
        print(return_line)
    except Exception as e:
        print(e)
