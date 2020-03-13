#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/15 13:50
@desc: 邮件发送编写
'''

import smtplib
from email.mime.text import MIMEText
from configs import email_config
from email.mime.multipart import MIMEMultipart

from configs import log

class EmailUtil(object):
    def __init__(self, sender_emails=None):
        self.email_host = email_config['mail_host']
        self.email_port = email_config['mail_port']
        self.email_pwd = email_config['mail_pwd']
        self.user_name = email_config['mail_sender']
        self.log = log
        if isinstance(sender_emails, str):
            self.sender = sender_emails
        elif isinstance(sender_emails, list):
            self.sender = ','.join(sender_emails)
        else:
            print("没有收件人列表，请检查！！！")

    def get_mail(self, subject, text,  is_attach=False, file_path=None):
        msg = MIMEMultipart()
        if is_attach:
            if file_path is None:
                print("请添加附件的文件地址！！！")
            att = MIMEText(open(file_path).read())
            att["Content-Type"] = 'application/octet-stream'
            att["Content-Disposition"] = 'attachment; filename="%s"' % file_path
            msg.attach(att)
        msg['Subject'] = subject
        msg.attach(MIMEText(text))
        msg['From'] = self.user_name
        msg['To'] = self.sender
        smtp = smtplib.SMTP(self.email_host, port=self.email_port)
        smtp.login(self.user_name, self.email_pwd)
        try:
            smtp.sendmail(self.user_name, self.sender, msg.as_string())
            smtp.quit()
            print('email send success.')
        except Exception as e:
            self.log.info("ERROR!!!-{}".format(e))
