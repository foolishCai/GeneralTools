# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''



import re
import os
import logging
from logging import handlers,Formatter,StreamHandler
from general_util.config import log_config


class CustomLog(object):
    def __init__(self):
        self.log_path=log_config['log_path']
        self.log_name =log_config['log_name']


    def get_log_info(self):
        log_info = {
            'loging_level': logging.INFO,
            'log_path':self.log_path,
            'log_name':self.log_name,
            'log_suffix':'%Y-%m-%d.log',
            'log_extMatch':r"^\d{4}-\d{2}-\d{2}(\.\w+)?$",
            'log_fmt':'%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_dtfmt':'%Y-%m-%d %H:%M:%S',
            'add_stream': False
        }
        return log_info


def get_logger(name, log_path, log_name):
    log_info = CustomLog(log_path=log_path, log_name=log_name).get_log_info()
    fmt = Formatter(log_info['log_fmt'], datefmt=log_info['log_dtfmt'])
    logger = logging.getLogger(name)
    logger.setLevel(log_info['loging_level'])
    log_path = os.path.join(log_info['log_path'],log_info['log_name'])
    handler = handlers.TimedRotatingFileHandler(filename=log_path, when='midnight',interval=1,backupCount=10)
    handler.suffix = log_info['log_suffix']
    handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}(\.\w+)?$")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    if log_info['add_stream']:
        handler_stream = StreamHandler()
        handler_stream.setFormatter(fmt)
        logger.addHandler(handler_stream)
    return logger


if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.info('日志记录器设置成功')