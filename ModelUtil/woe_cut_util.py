#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: chenyuwei_03@hotmail.com
@application:
@time: 2019/10/22 15:59
@desc: 利用python的woe包进行分箱，傻瓜式分箱
'''


import woe.feature_process as fp
import woe.eval as eval
import pandas as pd
from configs import log
import numpy as np

class GetWoeBin(object):
    def __init__(self, X, y_true, is_changed=True, file_name=None, is_tag=False, special_value=None):
        if isinstance(y_true, pd.Series):
            self.y = y_true.tolist()
        if isinstance(y_true, pd.DataFrame):
            self.y = y_true.iloc[:, 0].tolist()
        self.X = X
        self.is_changed = is_changed
        self.file_name = file_name
        self.is_tag = is_tag
        self.log = log
        if isinstance(special_value, dict):
            self.special_value = {k: special_value[k] for k in X.columns}
        elif isinstance(special_value, list):
            self.special_value = {k: special_value for k in X.columns}
        elif special_value is None:
            self.special_value = {k: [np.nan] for k in X.columns}
        else:
            self.special_value = {k: [special_value] for k in X.columns}

        self.global_bt = sum(self.y)
        self.global_gt = len(X) - self.global_bt

    def get_cols_dtypes(self, classVars=None, continousVars=None):
        if classVars is None or continousVars is None:
            self.classVars = list(self.X.columns[self.X.dtypes == 'object'])
            self.continousVars = list(self.X.columns[self.X.dtypes != 'object'])
        else:
            self.classVars = classVars
            self.continousVars = continousVars

    def get_cut_result(self):
        df = self.X.copy()
        df['target'] = self.y

        self.civ_dict = {}
        woe_fp = {}

        for column in list(df.columns):
            if column == 'target':
                continue
            self.log.info("------ Now dealing the var={}".format(column))
            tmp_df = df[~df[column].isin(self.special_value[column])]
            if column in self.classVars:
                civ = fp.proc_woe_discrete(tmp_df, column, self.global_bt, self.global_gt, 0.05 * len(tmp_df), alpha=0.05)
            else:
                civ = fp.proc_woe_continuous(tmp_df, column, self.global_bt, self.global_gt, 0.05 * len(tmp_df), alpha=0.05)
            self.civ_dict[column] = civ
            woe_fp[column] = fp

        if self.file_name is not None:
            feature_detail = eval.eval_feature_detail([v for k, v in self.civ_dict.items()], self.file_name)
        else:
            feature_detail = None

        if self.is_changed:
            changed_df = df.copy()
            changed_df = changed_df.drop(['target'], axis=1)
            for column in list(df.columns):
                if column == 'target':
                    continue
                changed_df[column] = woe_fp[column].woe_trans(df[column], self.civ_dict[column])
            if self.is_tag:
                for column in list(df.columns):
                    woe_list = [str(i) for i in self.civ_dict[column].woe_list]
                    split_list = self.civ_dict[column].split_list
                    if len(split_list) == 0:
                        continue
                    changed_df[column] = changed_df[column].map(
                        lambda x: '(' + str(split_list[-1]) + ',' + 'inf]' if x == woe_list[-1]
                        else '(' + str(split_list[woe_list.index(x) - 1]) + ',' + str(
                            split_list[woe_list.index(x)]) + ']')

        else:
            changed_df = None

        return woe_fp, changed_df, feature_detail

    def get_bins(self):
        feature = [k for k, v in self.civ_dict.items()]
        split_list = [v.split_list for k, v in self.civ_dict.items()]
        woe_list = [v.woe_list for k, v in self.civ_dict.items()]

        woe_value = []
        split_value = []
        for i in range(len(feature)):
            woe_value.append("#".join([str(round(k, 4)) for k in woe_list[i]]))
            split_value.append("#".join([str(round(k, 4)) for k in split_list[i]]))


        woe_df = pd.DataFrame(data={"feature": feature, "split_point": split_value, "woe_value": woe_value},
                              columns=['feature', 'split_point', 'woe_value'])
        woe_detail_df = pd.DataFrame(columns=['feature', 'is_continous', 'tag', 'woe'])
        for index, row in woe_df.iterrows():
            if row['feature'] in self.classVars:
                woe_list = row['woe_value'].split("#")
                feature_list = [row['feature']] * len(woe_list)
                tag_list = row['split_point'].split("#")
                is_continous = ['0'] * len(woe_list)
                woe_detail_df = pd.concat([woe_detail_df, pd.DataFrame(data={"feature": feature_list,
                                                                             "is_continous": is_continous,
                                                                             "tag": tag_list,
                                                                             "woe": woe_list})])
            elif row['feature'] in self.continousVars:
                woe_list = row['woe_value'].split("#")
                tag_list = row['split_point'].split("#")
                if len(woe_list) == len(tag_list):
                    tag_list[-1] = str('inf')
                    tag_list.append('-inf')
                    tag_list = [float(i) for i in tag_list]
                    tag_list.sort()
                    tag_list = [str(i) for i in tag_list]

                feature_list = [row['feature']] * len(woe_list)
                is_continous = ['1'] * len(woe_list)
                tag_list = ['(' + tag_list[i] + ',' + tag_list[i+1] + ']' for i in range(len(tag_list)-1)]
                woe_detail_df = pd.concat([woe_detail_df, pd.DataFrame(data={"feature": feature_list,
                                                                             "is_continous": is_continous,
                                                                             "tag": tag_list,
                                                                             "woe": woe_list})])
            return woe_detail_df