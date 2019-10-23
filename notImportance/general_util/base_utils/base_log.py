# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note:
'''

import os
import logging
from general_util.config import log_config

class BaseLog(object):
    def __init__(self):
        self.log_path = log_config['log_path']
        self.log_name = log_config['log_name']

        fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=fmt,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=os.path.join(self.log_path, self.log_name),
                            filemode='w')

        # 输出日志到shell
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def info(self, msg):
        msg = str(msg)
        logging.info(msg)

    def debug(self, msg):
        pass

    def error(self, msf):
        pass


if __name__ == '__main__':

    t = BaseLog()
    t.info('hello 1')
    t.info('hello 2')
