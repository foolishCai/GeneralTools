#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 14:51
# @Author  : cai
# @contact : chenyuwei_0303@yeah.net
# @File    : monotonic_woe_util.py
# @Note    :

import os
import sys

sys.path.append("..")

from multiprocessing import Pool, cpu_count, Manager
import pandas as pd
import numpy as np
import datetime

# from BaseUtils import log
from itertools import combinations


class MonotonicWoe(object):
    def __init__(self, df_train, df_test, target_name, del_col, dist_col, serial_col, df_ott=None, filne_name=None,
                 na_value=None, max_bins=5, min_rate=0.01, min_bins_cnt=50):
        #         self.log = log

        ## 检查y列
        if not target_name:
            self.log.info("爸，你清醒一点，没有lable列啦！")
            sys.exit(0)
        else:
            self.target_name = target_name
            del_col.append(target_name)

        ## 检查特征列
        if set(del_col).intersection(set(df_train.columns.tolist())) == set(del_col):
            self.del_col = list(set(del_col))
        else:
            self.log.info(
                "del_col中{}特征列没有在df_train.columns当中".format(set(del_col).difference(set(df_train.columns.tolist()))))
            sys.exit(0)

        if set(dist_col).intersection(set(df_train.columns.tolist())) == set(dist_col):
            self.dist_col = dist_col
        else:
            self.log.info(
                "dist_col中{}特征列没有在df_train.columns当中".format(set(dist_col).difference(set(df_train.columns.tolist()))))
            sys.exit(0)

        if set(serial_col).intersection(set(df_train.columns.tolist())) == set(serial_col):
            self.serial_col = serial_col
        else:
            self.log.info(
                "serial_col中{}特征列没有在df_train.columns当中".format(
                    set(serial_col).difference(set(df_train.columns.tolist()))))
            sys.exit(0)

        if not na_value:
            self.na_value = -99998
            self.na_value_list = [-99998, "-99998"]
            self.log.info("没有显示缺失值提示，默认用-99998代入;将执行df=df.fillna(-99998)")
        elif na_value.isdigit():
            self.na_value = na_value
            self.na_value_list = [na_value, str(na_value), float(na_value)]
        ## 留一个空荡，多类缺失值
        # else:
        #     self.na_value = [na_value]

        self._check_y(df_train)
        df_train = df_train[self.del_col + self.dist_col + self.serial_col]
        self.df_train = self._format_df(df_train)
        self.df_train_woe = self.df_train[[self.target_name]]

        if df_test is not None and isinstance(df_test, pd.DataFrame):
            self._check_y(df_test)
            df_test = df_test[self.del_col + self.dist_col + self.serial_col]
            self.df_test = self._format_df(df_test)
            self.df_test_woe = self.df_test[[self.target_name]]
        else:
            self.df_test = pd.DataFrame()
            self.log.info("啊哦，没有测试训练集～")

        if df_ott is not None and isinstance(df_ott, pd.DataFrame):
            self._check_y(df_ott)
            df_ott = df_ott[self.del_col + self.dist_col + self.serial_col]
            self.df_ott = self._format_df(df_ott)
            self.df_ott_woe = self.df_ott[[self.target_name]]
        else:
            self.df_ott = pd.DataFrame()
            self.log.info("好棒！没有跨时间验证窗口的数据～")

        if filne_name:
            self.filename = filne_name
        else:
            self.filename = "model" + datetime.datetime.today().strftime("%Y%m%d")

        self.max_bins = max_bins
        self.min_rate = min_rate
        self.min_bins_cnt = min_bins_cnt

    def _check_y(self, tmp_df):
        if len(tmp_df[self.target_name].unique()) != 2:
            self.log.info("旁友，你有点多心哦，本对象只适用于二分类！")

    def _format_df(self, df):
        df = pd.DataFrame(df, dtype=str)
        df[self.target_name] = df[self.target_name].apply(int)
        df.replace('\\N', self.na_value, inplace=True)
        df.replace('none', self.na_value, inplace=True)
        df.replace('nan', self.na_value, inplace=True)
        for col in self.dist_col:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        df = df.reset_index()
        return df

    ## 离散变量woe转化，针对self.dist_col
    def dist_woe_caculation(self):
        data = self.df_train[self.dist_col + [self.target_name] + ["index"]]
        bad_count = self.df_train[self.target_name].sum()
        good_count = len(self.df_train) - bad_count
        self.dist_woe_df = pd.DataFrame(columns=["feature", "values", "total", "bad", "bad_rate", "good",
                                                 "bad_pct", "good_pct", "WOE", 'IV', 'rank'])

        for index, i in enumerate(self.dist_col):
            tmp1 = data[[i, self.target_name, "index"]]
            tmp2 = tmp1[self.target_name].groupby([tmp1[i]]).agg(["sum", "count"]).reset_index()
            tmp2.rename(columns={"sum": "bad", "count": "total", i: "values"}, inplace=True)

            tmp2.loc[:, 'good'] = tmp2["total"] - tmp2['bad']
            tmp2.loc[:, 'bad_rate'] = tmp2.apply(lambda x: float("inf") if x["total"] == 0 else x['bad'] / x["total"],
                                                 axis=1)
            if good_count == 0:
                tmp2.loc[:, 'good_pct'] = float("inf")
            else:
                tmp2.loc[:, 'good_pct'] = (tmp2["total"] - tmp2['bad']) / good_count
            if bad_count == 0:
                tmp2.loc[:, 'bad_pct'] = float("inf")
            else:
                tmp2.loc[:, 'bad_pct'] = tmp2['bad'] / bad_count
            tmp2.loc[:, 'WOE'] = tmp2.apply(
                lambda x: float("inf") if x['bad_pct'] == 0 else np.log(x['good_pct'] / x['bad_pct']), axis=1)
            tmp2.loc[:, 'rank'] = tmp2.bad_rate.rank(axis=0, method='first')
            tmp2.loc[:, 'KS'] = 0
            tmp2.loc[:, 'feature'] = i
            tmp2.loc[:, 'IV'] = (tmp2['good_pct'] - tmp2['bad_pct']) * tmp2['WOE']
            tmp2 = tmp2[
                ["feature", "values", "total", "bad", "bad_rate", "good", "bad_pct", "good_pct", "WOE", 'IV', 'rank']]
            tmp2 = tmp2.sort_values('rank')
            self.df_train_woe[i] = self.df_train.merge(tmp2, how='left', left_on=i, right_on="values")['WOE']

            if self.df_test.any().any():
                self.df_test_woe[i] = self.df_test.merge(tmp2, how='left', left_on=i, right_on="values")['WOE']

            if self.df_ott.any().any():
                self.df_ott_woe[i] = self.df_ott.merge(tmp2, how='left', left_on=i, right_on="values")['WOE']
            self.dist_woe_df = pd.concat([self.dist_woe_df, tmp2])
        self.dist_woe_df.to_excel("{}_DistBin.xlsx".format(self.filename))
        self.log.info("The dist_col woe result can be checked in {}_DistBin.xlsx".format(self.filename))
        del data

    ## 连续变量woe转化，针对self.serial_col
    def serial_woe_caculation(self):
        pull_num = min(3, cpu_count() - 2)
        self.log.info("CPU内核数:{}，本次调用{}进程".format(cpu_count(), pull_num))
        self.log.info("仰天长笑，哈哈哈，CPU要飞起来啦！！！请做好降温防暑工作")

        with Manager() as manager:
            m = manager.dict()
            cons = manager.dict()

            p = Pool(pull_num)
            for col in self.serial_col:
                p.apply_async(self._multi_woe, args=(col, m, cons,))
            p.close()
            p.join()

            best_knots_df = dict(m)
            self.conditions = dict(cons)

        self.serial_df = pd.DataFrame(
            columns=["feature", "bin_name", "bad", "bad_rate", "good", "good_pct", "bad_pct", "woe", "iv", "ks"])
        for key, value in best_knots_df.items():
            self.serial_df = pd.concat([self.serial_df, value[
                ["feature", "bin_name", "bad", "bad_rate", "good", "good_pct", "bad_pct", "woe", "iv", "ks"]]])
            data = self.df_train[[key]]
            data[key] = data[key].astype(float)
            self.df_train_woe[key] = eval(self.conditions[key])
            if self.df_test.any().any():
                data = self.df_test[[key]]
                data[key] = data[key].astype(float)
                self.df_test_woe[key] = eval(self.conditions[key])
            if self.df_ott.any().any():
                data = self.df_ott[[key]]
                data[key] = data[key].astype(float)
                self.df_ott_woe[key] = eval(self.conditions[key])
        self.serial_df.to_excel("{}_SerialBin.xlsx".format(self.filename))
        self.log.info("The serial_col woe result can be checked in {}_SerialBin.xlsx".format(self.filename))

    def _multi_woe(self, col, m, cons):
        data = self.df_train[[col, self.target_name]]
        self.log.info('子进程: {} - 特征{}'.format(os.getpid(), col))

        cut_point = []
        work_data = data[self.target_name].groupby([data[col], data[self.target_name]]).count()
        work_data = work_data.unstack().reset_index().fillna(0)
        work_data.columns = [col, 'good', 'bad']
        na_df = work_data[work_data[col].isin(self.na_value_list)]
        non_na_df = work_data[~work_data[col].isin(self.na_value_list)]
        non_na_df[col] = non_na_df[col].astype(float)

        ## 对non_na_df进行处理
        total_len = sum(work_data['good']) + sum(work_data['bad'])
        current_rate = min(self.min_rate, self.min_bins_cnt / total_len)
        tmp_result = self._best_ks_knots(non_na_df, total_len=total_len, current_rate=current_rate, start_knot=0,
                                         end_knot=non_na_df.shape[0], current_time=0)
        tmp_result = [x for x in tmp_result if x is not None]
        tmp_result.sort()
        return_piece_num = min(self.max_bins, len(tmp_result) + 1)
        sorted(list(range(2, return_piece_num + 1)), reverse=True)
        ## cost alot time
        best_knots = []
        for current_piece_num in sorted(list(range(2, return_piece_num + 1)), reverse=True):
            knots_list = list(combinations(tmp_result, current_piece_num - 1))
            knots_list = [sorted(x + (0, len(non_na_df) - 1)) for x in knots_list]
            IV_for_bins = [self._IV_calculator(non_na_df, x) for x in knots_list]
            filtered_IV = [float("-inf") if str(x) == "None" else float(x) for x in IV_for_bins]
            if knots_list[int(np.argmax(filtered_IV))]:
                best_knots = knots_list[int(np.argmax(filtered_IV))]
                break
        if best_knots:
            bin_name, bad, good = [], [], []
            if not (na_df.values[0][1] == 0 or na_df.values[0][2] == 0):
                cut_point.append(self.na_value)
                bin_name.append(str(self.na_value))
                good.append(na_df.values[0][1])
                bad.append(na_df.values[0][2])

            cut_point.extend([non_na_df[col][best_knots[i]] for i in range(1, len(best_knots))])
            for i in range(1, len(best_knots)):
                if i == 1:
                    left_margin, right_margin = float("-inf"), non_na_df[col][best_knots[1]]
                elif i == len(best_knots) - 1:
                    left_margin, right_margin = non_na_df[col][best_knots[i - 1]], float("inf")
                else:
                    left_margin, right_margin = non_na_df[col][best_knots[i - 1]], non_na_df[col][best_knots[i]]
                tmp = non_na_df[
                    (non_na_df[col].astype(float) > left_margin) & (non_na_df[col].astype(float) <= right_margin)]
                bin_name.append("(" + str(left_margin) + "," + str(right_margin) + "]")
                good.append(sum(tmp["good"]))
                bad.append(sum(tmp["bad"]))
        else:
            self.log.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>该特征{}无法进行有效分箱".format(col))
        if cut_point:
            tmp = pd.DataFrame(columns=["bin_name", "bad", "good"],
                               data={"bin_name": bin_name, "bad": bad, "good": good})
            total_good = tmp.good.sum()
            total_bad = tmp.bad.sum()
            tmp["total"] = tmp.good + tmp.bad
            tmp["bad_rate"] = tmp.bad / tmp.total
            tmp["good_cumsum"] = np.cumsum(tmp["good"])
            tmp["bad_cumsum"] = np.cumsum(tmp["bad"])
            tmp["good_pct"] = tmp.good / total_good
            tmp["bad_pct"] = tmp.bad / total_bad
            tmp["woe"] = tmp.apply(
                lambda x: float("inf") if x['bad_pct'] == 0 else np.log(x['good_pct'] / x['bad_pct']), axis=1)
            tmp["ks"] = np.abs(tmp.good_pct - tmp.bad_pct)
            tmp['iv'] = (tmp['good_pct'] - tmp['bad_pct']) * tmp['woe']
            tmp["feature"] = col
            m[col] = tmp

            ## 组中conditions
            condition = ""
            for index, row in tmp.iterrows():
                if row["bin_name"] in mw.na_value_list:
                    condition = condition + "np.where(data['{}'].isin(self.na_value_list), float('{}'),".format(
                        row["feature"], row["woe"])
                else:
                    bin_min = row["bin_name"].split(',')[0].replace("(", "")
                    bin_max = row["bin_name"].split(',')[1].replace("]", "")
                    condition = condition + "np.where((data['{}']>float('{}')) & (data['{}']<=float('{}')), float('{}'),".format(
                        row["feature"], bin_min, row["feature"], bin_max, row["woe"])
            condition = condition + 'np.nan' + ')' * len(tmp)
            cons[col] = condition

    ## 迭代找到最优分割点
    def _best_ks_knots(self, data, total_len, current_rate, start_knot, end_knot, current_time):
        tmp_df = data.loc[start_knot:end_knot]
        tmp_df_len = sum(tmp_df["good"]) + sum(tmp_df["bad"])
        # 限制箱内最小样本数
        if tmp_df_len < current_rate * total_len * 2 or current_time >= self.max_bins:
            return []
        else:
            data_len = sum(data["good"]) + sum(data["bad"])
            start_add_num = sum(np.cumsum(tmp_df["good"] + tmp_df["bad"]) < 0.007 * data_len)
            end_add_num = sum(np.cumsum(tmp_df["good"] + tmp_df["bad"]) <= tmp_df_len - 0.007 * data_len)
            processed_start_knot = start_knot + start_add_num
            processed_end_knot = start_knot + end_add_num - 1
            if processed_end_knot >= processed_start_knot:
                if sum(tmp_df["bad"]) != 0 and sum(tmp_df["good"]) != 0:
                    tmp_df.loc[:, "cumsum_bad_rate"] = np.cumsum(tmp_df["bad"]) / tmp_df["bad"].sum()
                    tmp_df.loc[:, "cumsum_good_rate"] = np.cumsum(tmp_df["good"]) / tmp_df["good"].sum()
                    tmp_df.loc[:, "ks_value"] = abs(tmp_df["cumsum_bad_rate"] - tmp_df["cumsum_good_rate"])
                    new_knot = tmp_df.index[np.argmax(tmp_df.ks_value)]
                else:
                    new_knot = None
            else:
                new_knot = None

        if new_knot:
            upper_result = self._best_ks_knots(data, total_len, current_rate, start_knot, new_knot, current_time + 1)
            lower_result = self._best_ks_knots(data, total_len, current_rate, new_knot + 1, end_knot, current_time + 1)
        else:
            upper_result = []
            lower_result = []
        return upper_result + [new_knot] + lower_result

    ## 计算IV
    def _IV_calculator(self, data_df, knots_list):
        temp_df_list = []
        for i in range(1, len(knots_list)):
            if i == 1:
                temp_df_list.append(data_df.loc[knots_list[i - 1]:knots_list[i]])
            else:
                temp_df_list.append(data_df.loc[knots_list[i - 1] + 1:knots_list[i]])
        total_good = sum(data_df["good"])
        total_bad = sum(data_df["bad"])
        good_percent_series = pd.Series(list([float(sum(x["good"])) / total_good for x in temp_df_list]))
        bad_percent_series = pd.Series(list([float(sum(x["bad"])) / total_bad for x in temp_df_list]))
        woe_list = list(np.log(good_percent_series / bad_percent_series))
        if sorted(woe_list) != woe_list and sorted(woe_list, reverse=True) != woe_list:  # 判断是否单调
            return None
        IV_series = (good_percent_series - bad_percent_series) * np.log(good_percent_series / bad_percent_series)
        if np.inf in list(IV_series) or -np.inf in list(IV_series):
            return None
        else:
            return sum(IV_series)