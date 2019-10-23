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


def get_woe_bin(df, target_name, is_changed=True, file_name=None):
    df.rename(columns={target_name: 'target'}, inplace=True)
    n_positive = sum(df['label'])
    n_negtive = len(df) - n_positive
    civ_dict = {}
    woe_fp = {}
    for column in list(df.columns):
        if column == 'target':
            continue
        if df[column].dtypes == 'object':
            civ = fp.proc_woe_discrete(df, column, n_positive, n_negtive, 0.05*len(df), alpha=0.05)
        else:
            civ = fp.proc_woe_continuous(df, column, n_positive, n_negtive, 0.05*len(df), alpha=0.05)
        civ_dict[column] = civ
        woe_fp[column] = fp

    if file_name is not None:
        feature_detail = eval.eval_feature_detail([v for k, v in civ_dict.items()], file_name)
    else:
        feature_detail = None

    if is_changed:
        changed_df = df.copy()
        for column in list(df.columns):
            changed_df[column] = woe_fp[column].woe_trans(df[column], civ_dict[column])
    else:
        changed_df = None

    return civ_dict, woe_fp, changed_df, feature_detail