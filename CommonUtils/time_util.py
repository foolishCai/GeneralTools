# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''


import time
import datetime
import re


# 把时间戳转成字符串形式
def tmp2str(tmStamp, format="%Y-%m-%d %H:%M:%S"):
    tmStamp = str(tmStamp)
    if tmStamp.find('.') <= -1:
        if re.match('^\d{0,10}$', tmStamp):
            tmStamp = float(tmStamp)
        else:
            tmStamp = float(tmStamp)/pow(10, len(tmStamp)-10)
    else:
        if re.match('^(\d{0,10})(\.)(\d{0,6})$', tmStamp):
            tmStamp = float(tmStamp)
        else:
            return '无法正确转换，该timestamp位数不对'

    tmStr = time.strftime(format, time.localtime(tmStamp))
    return tmStr


# 把字符串转成时间戳形式
def str2tmp(tmStr, format="%Y-%m-%d %H:%M:%S"):
    return time.mktime(time.strptime(tmStr, format))





# 转换日期格式
def change_date_format(date, former_format='%Y%m%d',to_format ='%Y-%m-%d'):
    if former_format == to_format:
        return date
    else:
        date = datetime.datetime.strptime(date, former_format)
        return date.strftime(to_format)


# 计算日期加减
def date_calc(date, delta_day, date_format=None):
    ##能够自动判断string作为date的类型，然后根据delta_day(+ 代表往后，-代表往前）推几天
    ##支持 '%Y-%m-%d' '%Y%m%d' '%Y/%m/%d' 的自动判断，如果是其他的形式，可以写在date_format中
    ##默认返回相同的格式
    if date_format == None:
        if re.match('^\d{8}$',date):
            date_format = '%Y%m%d'
        elif re.match('^\d{4}-\d{2}-\d{2}$',date):
            date_format = '%Y-%m-%d'
        elif re.match('^\d{4}/\d{2}/\d{2}$',date):
            date_format = '%Y/%m/%d'
        else:
            print('未能匹配出格式，请手动加入 date_format参数')
    time = datetime.datetime.strptime(date,date_format)
    time_result = time + datetime.timedelta(days=delta_day)
    return datetime.datetime.strftime(time_result,date_format)


# 计算天数差, 小时差
def delta_days(from_dt,to_dt,from_dt_format=None,to_dt_format=None):
    ## 默认两个dt的格式是一致的
    if from_dt_format == None:
        if re.match('^\d{8}$',from_dt):
            from_dt_format = '%Y%m%d'
        elif re.match('^\d{4}-\d{2}-\d{2}$',from_dt):
            from_dt_format = '%Y-%m-%d'
        elif re.match('^\d{4}/\d{2}/\d{2}$',from_dt):
            from_dt_format = '%Y/%m/%d'
        else:
            print('未能匹配出格式，请手动加入 from_dt 参数')
    if to_dt_format == None:
        if re.match('^\d{8}$',to_dt):
            to_dt_format = '%Y%m%d'
        elif re.match('^\d{4}-\d{2}-\d{2}$',to_dt):
            to_dt_format = '%Y-%m-%d'
        elif re.match('^\d{4}/\d{2}/\d{2}$',to_dt):
            to_dt_format = '%Y/%m/%d'
        else:
            print('未能匹配出格式，请手动加入 to_dt 参数')
    from_dt = datetime.datetime.strptime(from_dt, from_dt_format)
    to_dt = datetime.datetime.strptime(to_dt, to_dt_format)
    days = (to_dt - from_dt).days
    return abs(days)


# 计算小时差
def delta_hours(from_dt,to_dt,from_dt_format = None,to_dt_format = None):
    ## 默认两个dt的格式是一致的
    if from_dt_format == None:
        if re.match('^\d{8} \d{2}:\d{2}:\d{2}$',from_dt):
            from_dt_format = '%Y%m%d %H:%M:%S'
        elif re.match('^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',from_dt):
            from_dt_format = '%Y-%m-%d %H:%M:%S'
        elif re.match('^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$',from_dt):
            from_dt_format = '%Y/%m/%d %H:%M:%S'
        else:
            print('未能匹配出格式，请手动加入 from_dt 参数')
    if to_dt_format == None:
        if re.match('^\d{8} \d{2}:\d{2}:\d{2}$',to_dt):
            to_dt_format = '%Y%m%d %H:%M:%S'
        elif re.match('^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',to_dt):
            to_dt_format = '%Y-%m-%d %H:%M:%S'
        elif re.match('^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$',to_dt):
            to_dt_format = '%Y/%m/%d %H:%M:%S'
        else:
            print('未能匹配出格式，请手动加入 to_dt 参数')
    from_dt = datetime.datetime.strptime(from_dt, from_dt_format)
    to_dt = datetime.datetime.strptime(to_dt, to_dt_format)
    hours = round((to_dt - from_dt).seconds/3600, 2)
    return abs(hours)


def get_std_time(tmStr, level="m"):
    if re.match('^\d{8}$', tmStr):
        formatDate = datetime.datetime.strptime(tmStr, "%Y%m%d")
    elif re.match('^\d{4}-\d{2}-\d{2}$', tmStr):
        formatDate = datetime.datetime.strptime(tmStr, "%Y-%m-%d")
    elif re.match('^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', tmStr):
        formatDate = datetime.datetime.strptime(tmStr[:10], "%Y-%m-%d")
    else:
        formatDate = None
    if formatDate:
        if level == "y":
            return formatDate.strftime("%Y")
        elif level == "m":
            return formatDate.strftime("%Y%m")
        else:
            return formatDate.strftime("%Y%m%d")
    else:
        raise Exception("Format Error!")