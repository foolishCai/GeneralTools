#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/11/4 10:34
@desc: 补历史数据
'''


import logging
import datetime
import os


class LogUtil(object):
    def __init__(self, log_name=None):
        if log_name is None:
            log_name = "log_test"
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        # 定义输出格式
        formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')

        # 创建一个文件的handler
        fh = logging.FileHandler("/home/chenyw_yzx/data_bak/logs" + os.sep + log_name + '.log')
        fh.setFormatter(fmt=formatter)
        self.logger.addHandler(fh)

        # 创建一个控制台输出的handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt=formatter)
        self.logger.addHandler(ch)

    def info(self, msg):
        msg = str(msg)
        self.logger.info(msg)

    def debug(self, msg):
        msg = str(msg)
        self.logger.debug(msg)

    def error(self, msg):
        msg = str(msg)
        self.logger.error("!!!ERROR!!!", msg)



class GetData(object):
    def __init__(self, start, end, date_type='day'):
        self.log = LogUtil("LoopSql")
        self.start = datetime.datetime.strptime(start, "%Y%m%d")
        self.end = datetime.datetime.strptime(end, "%Y%m%d")
        self.user_name = 'chenyw_yzx'
        self.password_path = '/home/chenyw_yzx/beeline.password'
        self.date_type = date_type
        self.sql_file = "/home/chenyw_yzx/AutoLabel/get_bin_result.sql"

    def get_loop(self):
        if self.date_type == 'day':
            for i in range((self.end- self.start).days):
                day = (self.start + datetime.timedelta(days=i)).strftime("%Y%m%d")
                self.log.info("Now is dealing the SqlFile for date={}".format(day))
                cmd = """beeline -u jdbc:hive2://beeline-ds.bi.getui.com:10000 -n {} -w {} -hivevar day={} -f {}""".format(
                    self.user_name, self.password_path, day, self.sql_file)
                try:
                    run_start = datetime.datetime.now()
                    os.system(cmd)
                    run_end = datetime.datetime.now()
                    self.log.info("SqlFile for date={} has been done,cost_time {} sec".format(day, (run_start-run_end).seconds))
                except Exception as e:
                    self.log.info("SqlFile for date={} has been ERROR, {}!!!!".format(day, e))

        elif self.date_type == 'month':
            current_month = self.start
            while current_month <= self.end:
                hive_var_month = current_month.strftime("%Y%m")
                self.log.info("Now is dealing the SqlFile for date={}".format(hive_var_month))
                cmd = """beeline -u jdbc:hive2://beeline-ds.bi.getui.com:10000 -n {} -w {} --hivevar month={} --hivevar day={} -f {}""".format(
                    self.user_name, self.password_path, hive_var_month, current_month.strftime("%Y%m%d"), self.sql_file)
                try:
                    run_start = datetime.datetime.now()
                    os.system(cmd)
                    run_end = datetime.datetime.now()
                    self.log.info("SqlFile for date={} has been done,cost_time {} sec".format(hive_var_month, (run_start-run_end).seconds))
                except Exception as e:
                    self.log.info("SqlFile for date={} has been ERROR, {}!!!!".format(hive_var_month, e))
                current_month = datetime.datetime(current_month.year + (current_month.month == 12), current_month.month == 12 or current_month.month + 1, current_month.day)


if __name__ == '__main__':
    gd = GetData(start='20190601', end='20191001', date_type='month')
