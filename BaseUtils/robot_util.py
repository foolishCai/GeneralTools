# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 
'''


from ToolBars.configs.robot_config import RobotConfig
import requests
import json


def str2list(s):
    '''当s是str时，转化成 ['s']的格式 '''
    if isinstance(s, str):
        tmp = []
        tmp.append(s)
        return tmp
    elif isinstance(s, list):
        return s


class RobotUtil(object):
    '''钉钉机器人'''

    def __init__(self):
        self.url = RobotConfig['傻瓜菜地']['url']


    def send_text(self, text='No Info', atMobiles=None, isAtall=False):
        '''
            发送文字
            :var text 要发送的文字
            :var atMobiles 要at的电话号码，注意要用字符串表示
            :var isAtall Bool 是否at所有人
        '''
        self.headers = {'Content-Type': 'application/json; charset=utf-8'}
        self.data = {
            "msgtype": "text",
            "text": {
                "content": text
            },
            "at": {
                "isAtAll": isAtall
            }
        }
        if atMobiles is not None:
            atMobiles = str2list(atMobiles)
            self.data['at']['atMobiles'] = atMobiles
        self.s = json.dumps(self.data)
        self.req_result = requests.post(url=self.url, data=self.s, headers=self.headers)
        return self.req_result.text

    def send_markdown(self, title='No Info', text='No Info', atMobiles=None, isAtall=False):
        self.headers = {'Content-Type': 'application/json; charset=utf-8'}
        self.data = {
            "msgtype": "markdown",
            "markdown": {
                "title": title,
                "text": text
            },
            "at": {
                "isAtAll": isAtall
            }
        }
        self.s = json.dumps(self.data)
        self.req_result = requests.post(url=self.url, data=self.s, headers=self.headers)
        return self.req_result.text
