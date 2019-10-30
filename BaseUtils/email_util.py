#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/15 13:50
@desc:
'''

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.header import Header

mail_host = "smtp.qq.com"  # 设置服务器
mail_user = "XXX"  # 用户名
mail_pass = "YYY"  # 三方邮箱口令
sender = 'bbbbb.qq.com'  # 发送者邮箱
receivers = ['bbbbbbb.qq.com']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱


class Email():
    def __init__(self, msg=''):
        versionDsc = '您好！有新的IPA包可以下载，下载地址...\n%s\n' % msg
        message = MIMEText(versionDsc, 'plain', 'utf-8')
        message['From'] = Header("某某某有限公司", 'utf-8')
        message['To'] = Header("某某某", 'utf-8')
        subject = '此邮件来自自动化打包'  #邮件来源
        message['Subject'] = Header(subject, 'utf-8')  #编码方式
        try:
            smtpObj = smtplib.SMTP()
            smtpObj.connect(mail_host, 25)  # 25 为 SMTP 端口号
            smtpObj.login(mail_user, mail_pass)
            smtpObj.sendmail(sender, receivers, message.as_string())
            print("邮件发送成功")
        except smtplib.SMTPException as e:
            print("Error: 无法发送邮件" + str(e))
