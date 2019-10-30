#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 09:52
@desc:
'''


import os
import logging
from configs import log_config

class LogUtil(object):
    def __init__(self, log_name=None):
        if log_name is None:
            log_name = log_config['log_name'] + '.log'
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        # 定义输出格式
        formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')

        # 创建一个文件的handler
        fh = logging.FileHandler(log_config['log_path'] + os.sep + log_name + '.log')
        fh.setFormatter(fmt=formatter)
        self.logger.addHandler(fh)

        # 创建一个控制台输出的handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt=formatter)
        self.logger.addHandler(ch)

        # self.log_path = log_config['log_path']
        # self.log_name = log_config['log_name']
        # logging.basicConfig(level=logging.INFO,
        #                     format=formatter,
        #                     datefmt='%Y-%m-%d %H:%M:%S',
        #                     filename=os.path.join(self.log_path, self.log_name),
        #                     filemode='w')
        #
        # # 输出日志到shell
        # console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        # formatter = logging.Formatter(formatter)
        # console.setFormatter(formatter)
        # logging.getLogger(log_name).addHandler(console)

    def info(self, msg):
        msg = str(msg)
        self.logger.info(msg)

    def debug(self, msg):
        msg = str(msg)
        self.logger.debug(msg)

    def error(self, msg):
        msg = str(msg)
        self.logger.error("!!!ERROR!!!", msg)


if __name__ == '__main__':

    t = LogUtil(log_name='test')
    t.info('hello 1')
    t.info('hello 2')