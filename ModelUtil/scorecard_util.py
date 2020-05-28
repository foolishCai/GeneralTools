#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 14:51
# @Author  : cai
# @contact : chenyuwei_0303@yeah.net
# @File    : scorecard_util.py
# @Note    :


import sys
sys.path.append("..")

from multiprocessing import Pool, cpu_count, Manager
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import math


from BaseUtils import log



class MonotonicWoe(object):
    def __init__(self, df_train, target_name, del_col, dist_col, serial_col, df_test=None, df_ott=None, filne_name=None,
                 na_value=None, max_bins=5, min_rate=0.01, min_bins_cnt=50, max_process=2):
        self.log = log

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
        self.pull_num = min(max_process, cpu_count() - 2)
        self.log.info("CPU内核数:{}，本次调用{}进程".format(cpu_count(), self.pull_num))

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
        self.dist_woe_df = pd.DataFrame(columns=["feature", "bin_name", "total", "bad", "bad_rate", "good",
                                                 "bad_pct", "good_pct", "woe", 'iv', 'rank'])

        for index, i in enumerate(self.dist_col):
            tmp1 = data[[i, self.target_name, "index"]]
            tmp2 = tmp1[self.target_name].groupby([tmp1[i]]).agg(["sum", "count"]).reset_index()
            tmp2.rename(columns={"sum": "bad", "count": "total", i: "bin_name"}, inplace=True)

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
            tmp2.loc[:, 'woe'] = tmp2.apply(
                lambda x: float("inf") if x['bad_pct'] == 0 else np.log(x['good_pct'] / x['bad_pct']), axis=1)
            tmp2.loc[:, 'rank'] = tmp2.bad_rate.rank(axis=0, method='first')
            tmp2.loc[:, 'ks'] = 0
            tmp2.loc[:, 'feature'] = i
            tmp2.loc[:, 'iv'] = (tmp2['good_pct'] - tmp2['bad_pct']) * tmp2['woe']
            tmp2 = tmp2[
                ["feature", "bin_name", "total", "bad", "bad_rate", "good", "bad_pct", "good_pct", "woe", 'iv', 'rank']]
            tmp2 = tmp2.sort_values('rank')
            self.df_train_woe[i] = self.df_train.merge(tmp2, how='left', left_on=i, right_on="bin_name")['woe']

            if self.df_test.any().any():
                self.df_test_woe[i] = self.df_test.merge(tmp2, how='left', left_on=i, right_on="bin_name")['woe']

            if self.df_ott.any().any():
                self.df_ott_woe[i] = self.df_ott.merge(tmp2, how='left', left_on=i, right_on="bin_name")['woe']
            self.dist_woe_df = pd.concat([self.dist_woe_df, tmp2])
        self.dist_woe_df.to_excel("{}_DistBin.xlsx".format(self.filename))

        del data

    ## 连续变量woe转化，针对self.serial_col
    def serial_woe_caculation(self):
        self.log.info("仰天长笑，哈哈哈，CPU要飞起来啦！！！请做好降温防暑工作")

        with Manager() as manager:
            m = manager.dict()
            cons = manager.dict()
            fill_woe_value = manager.dict()

            p = Pool(self.pull_num)
            for col in self.serial_col:
                p.apply_async(self._multi_woe, args=(col, m, cons, fill_woe_value))
            p.close()
            p.join()

            best_knots_df = dict(m)
            conditions = dict(cons)
            fill_values = dict(fill_woe_value)

        self.serial_df = pd.DataFrame(
            columns=["feature", "bin_name", "bad", "bad_rate", "good", "good_pct", "bad_pct", "woe", "iv", "ks"])
        for key, value in best_knots_df.items():
            self.serial_df = pd.concat([self.serial_df, value[
                ["feature", "bin_name", "bad", "bad_rate", "good", "good_pct", "bad_pct", "woe", "iv", "ks"]]])
            data = self.df_train[[key]]
            data[key] = data[key].astype(float)
            self.df_train_woe[key] = eval(conditions[key])
            if self.df_train_woe[key].isnull().max():
                self.df_train_woe[key] = self.df_train_woe[key].fillna(value[value.feature==key].woe.max() if fill_values[key]==1 else value[value.feature==key].woe.min())
            if self.df_test.any().any():
                data = self.df_test[[key]]
                data[key] = data[key].astype(float)
                self.df_test_woe[key] = eval(conditions[key])
                if self.df_test_woe[key].isnull().max():
                    self.df_test_woe[key]=self.df_test_woe[key].fillna(value[value.feature==key].woe.max() if fill_values[key]==1 else value[value.feature==key].woe.min())
            if self.df_ott.any().any():
                data = self.df_ott[[key]]
                data[key] = data[key].astype(float)
                self.df_ott_woe[key] = eval(conditions[key])
                if self.df_ott_woe[key].isnull().max():
                    self.df_ott_woe[key] = self.df_ott_woe[key].fillna(value[value.feature==key].woe.max() if fill_values[key]==1 else value[value.feature==key].woe.min())
        self.serial_df.to_excel("{}_SerialBin.xlsx".format(self.filename))

    def _multi_woe(self, col, m, cons, fill_woe_value):
        data = self.df_train[[col, self.target_name]]
        # self.log.info('子进程: {} - 特征{}'.format(os.getpid(), col))

        cut_point = []
        work_data = data[self.target_name].groupby([data[col], data[self.target_name]]).count()
        work_data = work_data.unstack().reset_index().fillna(0)
        work_data.columns = [col, 'good', 'bad']
        na_df = work_data[work_data[col].isin(self.na_value_list)]

        non_na_df = work_data[~work_data[col].isin(self.na_value_list)]
        non_na_df[col] = non_na_df[col].astype(float)
        non_na_df = non_na_df.sort_values(by=[col], ascending=True)
        non_na_df = non_na_df.reset_index(drop=True)

        ## 对non_na_df进行处理
        total_len = sum(work_data['good']) + sum(work_data['bad'])
        current_rate = min(self.min_rate, self.min_bins_cnt / total_len)
        tmp_result = self._best_ks_knots(non_na_df, total_len=total_len, current_rate=current_rate, start_knot=0,
                                         end_knot=non_na_df.shape[0], current_time=0)
        tmp_result = [x for x in tmp_result if x is not None]
        tmp_result.sort()
        return_piece_num = min(self.max_bins, len(tmp_result) + 1)
        ## cost alot time
        best_knots = []
        for current_piece_num in sorted(list(range(2, return_piece_num + 1)), reverse=True):
            knots_list = list(combinations(tmp_result, current_piece_num - 1))
            knots_list = [sorted(x + (0, len(non_na_df) - 1)) for x in knots_list]
            IV_for_bins = [self._IV_calculator(non_na_df, x) for x in knots_list]
            filtered_IV = [float("-inf") if str(x) == "None" else float(x) for x in IV_for_bins]
            if int(np.argmax(filtered_IV)) == 0 and filtered_IV[0] == float("-inf"):
                continue
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
            elif na_df.values[0][1] > 0:
                fill_woe_value[col] = 0
            elif na_df.values[0][2] > 0:
                fill_woe_value[col] = 1

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
            self.log.info("！！！！！！！！！！该特征{}无法进行有效分箱".format(col))
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
                if row["bin_name"] in self.na_value_list:
                    condition = condition + "np.where(data['{}'].isin(self.na_value_list), float('{}'),".format(
                        row["feature"], row["woe"])
                else:
                    bin_min = row["bin_name"].split(',')[0].replace("(", "")
                    bin_max = row["bin_name"].split(',')[1].replace("]", "")
                    condition = condition + "np.where((~data['{}'].isin(self.na_value_list)) & (data['{}']>float('{}')) & (data['{}']<=float('{}')), float('{}'),".format(
                        row["feature"], row["feature"], bin_min, row["feature"], bin_max, row["woe"])
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
            start_add_num = sum(np.cumsum(tmp_df["good"] + tmp_df["bad"]) < current_rate * data_len)
            end_add_num = sum(np.cumsum(tmp_df["good"] + tmp_df["bad"]) <= tmp_df_len - current_rate * data_len)
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


    def main(self):
        stime = datetime.datetime.now()

        self.log.info(">>>>>>>>>>>>>>>>>离散变量分箱开始<<<<<<<<<<<<<<")
        self.dist_woe_caculation()
        self.log.info("The dist_col woe result can be checked {}.dist_woe_df".format(self.__class__.__name__))
        self.log.info("数据文件保存在{}_DistBin.xlsx".format(self.filename))

        self.log.info(">>>>>>>>>>>>>>>>>连续变量分箱开始<<<<<<<<<<<<<<")
        self.serial_woe_caculation()
        etime = datetime.datetime.now()
        self.log.info("最耗时part已结束！哇o～，cost time {} seconds.".format((etime-stime).seconds))
        self.log.info("The serial_col woe result can be checked {}.serial_df".format(self.__class__.__name__))
        self.log.info("数据文件保存在{}_SerialBin.xlsx".format(self.filename))

        self.log.info(">>>>>>>>>>>>>>>>>WOE映射开始<<<<<<<<<<<<<<")
        self.df_train_woe.to_csv("{}_train_woe.csv".format(self.filename), sep="|", index=False)
        self.log.info("训练集的数据保存在{}_train_woe.csv".format(self.filename))

        if self.df_test.any().any():
            self.df_test_woe.to_csv("{}_test_woe.csv".format(self.filename), sep="|", index=False)
            self.log.info("测试集的数据保存在{}_test_woe.csv".format(self.filename))

        if self.df_ott.any().any():
            self.df_ott_woe.to_csv("{}_ott_woe.csv".format(self.filename), sep="|", index=False)
            self.log.info("ott的数据保存在{}_ott_woe.csv".format(self.filename))


class ScorecardUtil(object):
    def __init__(self, df_train, target_name, woe_df, df_test=None, df_ott=None, filne_name=None, max_corr=0.75, max_vif=10,
                 na_value=None, base_score=550, double_score=20, cut_num=10):

        self.log = log
        self.label = target_name
        if filne_name:
            self.filename = filne_name
        else:
            self.filename = "model" + datetime.datetime.today().strftime("%Y%m%d")

        self.df_train_woe = df_train
        if df_test is not None and isinstance(df_test, pd.DataFrame):
            self.df_test_woe = df_test
        else:
            self.df_test_woe = pd.DataFrame()
            self.log.info("啊哦，没有测试训练集～")

        if df_ott is not None and isinstance(df_ott, pd.DataFrame):
            self.df_ott_woe = df_ott
        else:
            self.df_ott_woe = pd.DataFrame()
            self.log.info("好棒！没有跨时间验证窗口的数据～")

        self.max_corr = max_corr
        self.max_vif = max_vif
        self.base_score = base_score
        self.double_score = double_score
        self.cut_num = cut_num

        self.woe_df = woe_df

        if isinstance(na_value, int):
            self.na_list = [na_value, float(na_value), str(na_value)]
        elif isinstance(na_value, float):
            self.na_list = [str(na_value)]
        elif isinstance(na_value, str):
            self.na_list = [na_value]

    def check_woe_value(self):
        tmp = self.woe_df[~self.woe_df.woe.isin(self.na_list)]
        tmp = tmp["feature"].groupby(tmp["feature"]).agg(["count"]).reset_index()
        if len(tmp[tmp["count"] <= 1]):
            self.del_col = tmp.feature.tolist() + [self.label]
        else:
            self.del_col = [self.label]

        features = [i for i in self.df_train_woe if i not in self.del_col]
        tmp = self.woe_df[
            (~self.woe_df.woe.isin([float("-inf"), float("inf")])) & (self.woe_df.feature.isin(features))]
        tmp = tmp["woe"].groupby([tmp["feature"]]).agg(["min", "max"]).reset_index()

        key_ = [k for k in tmp.feature.tolist()]
        value_min = [v for v in tmp["min"].tolist()]
        value_max = [v for v in tmp["max"].tolist()]
        woe_min = {key_[i]: value_min[i] for i in range(len(tmp))}
        woe_max = {key_[i]: value_max[i] for i in range(len(tmp))}

        for col in features:
            self.df_train_woe[col] = self.df_train_woe[col].map(
                lambda x: woe_min[col] if x == float("-inf") else woe_max[col] if x == float("inf") else x)
            if self.df_test_woe.any().any():
                self.df_test_woe[col] = self.df_test_woe[col].map(
                    lambda x: woe_min[col] if x == float("-inf") else woe_max[col] if x == float("inf") else x)
            if self.df_ott_woe.any().any():
                self.df_ott_woe[col] = self.df_ott_woe[col].map(
                    lambda x: woe_min[col] if x == float("-inf") else woe_max[col] if x == float("inf") else x)

    def check_corr(self):
        tmp = self.woe_df[["feature", "iv"]]
        tmp["iv"] = tmp.iv.map(lambda x: 0 if float(x) == float("-inf") or float(x) == float("inf") else x)
        tmp = tmp["iv"].groupby(tmp["feature"]).agg(["sum"]).reset_index()
        self.iv_df = tmp.sort_values(by="sum")
        self.final_features = list(self.iv_df.feature)

        drop_set, var_set = set(), set(list(self.iv_df.feature))
        for i in list(tmp.feature):
            if i in drop_set:
                continue
            else:
                drop_set |= {v for v in var_set if
                             np.corrcoef(self.df_train_woe[i].values, self.df_train_woe[v].values)[
                                 0, 1] >= self.max_corr and v != i}
                var_set -= drop_set
        return list(drop_set)

    def check_vif(self):
        drop_col = []
        while 1:
            vars = [i for i in self.final_features if i not in drop_col]
            tmp = self.df_train_woe[vars]
            col = list(range(tmp.shape[1]))
            vif_df = pd.DataFrame()
            vif_df['vif_value'] = [variance_inflation_factor(tmp.iloc[:, col].values, ix) for ix in
                                   range(tmp.iloc[:, col].shape[1])]
            vif_df['feature'] = tmp.columns

            vif_df = pd.merge(vif_df, self.iv_df, how="inner", on="feature")
            vif_df = vif_df[vif_df.vif_value > self.max_vif]
            if len(vif_df) <= 0:
                break
            col = vif_df.sort_values(by="sum").feature.values[0]
            drop_col.append(col)
            self.log.info("The vif value is not accepted, delete the col={}".format(col))
        return drop_col

    ## 逐步回归挑选特征
    def check_stepwise(self):
        x_train = self.df_train_woe[self.final_features]
        x_train = sm.add_constant(x_train)
        logit_1 = sm.Logit(self.df_train_woe[self.label], x_train)
        result_1 = logit_1.fit()
        wald_chi2 = np.square((result_1.params) / np.square(result_1.bse))
        tmp = pd.DataFrame(wald_chi2, columns=['value'])
        tmp = tmp.sort_values('value', ascending=False)
        sorted_cols = tmp.index.tolist()
        if "const" in sorted_cols:
            sorted_cols.remove("const")
        train_cols = []
        for i in sorted_cols:
            train_cols.append(i)
            x_train = self.df_train_woe[train_cols]
            x_train = sm.add_constant(x_train)
            logit = sm.Logit(self.df_train_woe[self.label], x_train)
            result = logit.fit()
            train_cols = result.pvalues[result.pvalues < 0.05].index.tolist()
            if 'const' in train_cols:
                train_cols.remove('const')
            var_coef = result.params.reset_index()
            var_coef.columns = ['var', 'coef']
            to_drop_cols = var_coef.loc[var_coef['coef'] > 0]["var"].tolist()
            train_cols = [i for i in train_cols if i not in to_drop_cols]
        return train_cols

    ## 进模型
    def model_build(self):
        x_train = self.df_train_woe[self.final_features]
        x_train = sm.add_constant(x_train)
        y_train = self.df_train_woe[self.label]
        logit = sm.Logit(y_train, x_train)
        result = logit.fit()

        f = open("{}_model.pkl".format(self.filename), 'wb')
        pickle.dump(result, f)
        f.close()
        self.model = result

    ## 预测分数，进行模型评估
    def pred_evaluate(self):
        from scToolkits.att_sc_draw import get_curve, get_feature_importance, get_cm

        x_data = self.df_train_woe[self.final_features]
        x_data = sm.add_constant(x_data)
        self.train_y_pred = self.model.predict(x_data).tolist()
        get_curve(y_true=self.df_train_woe[self.label], y_pred=self.train_y_pred,
                  file_name="{}_train_report.png".format(self.filename))

        from sklearn.metrics import roc_curve
        fpr, tpr, threshold = roc_curve(y_true=self.df_train_woe[self.label], y_score=self.train_y_pred)
        thresh = sorted(self.train_y_pred)[int(np.argmax(abs(fpr - tpr)))]

        get_cm(y_true=self.df_train_woe[self.label], y_pred=self.train_y_pred, thresh=thresh)

        report_df = pd.DataFrame(columns=["precision", "recall", "accuracy", "f1_score"])
        report_df = pd.concat([pd.DataFrame(data={"precision": [precision_score(y_true=self.df_train_woe[self.label],
                                                                                y_pred=[int(i > thresh) for i in
                                                                                        list(self.train_y_pred)])],
                                                  "recall": [recall_score(y_true=self.df_train_woe[self.label],
                                                                          y_pred=[int(i > thresh) for i in
                                                                                  list(self.train_y_pred)])],
                                                  "accuracy": [accuracy_score(y_true=self.df_train_woe[self.label],
                                                                              y_pred=[int(i > thresh) for i in
                                                                                      list(self.train_y_pred)])],
                                                  "f1_score": [f1_score(y_true=self.df_train_woe[self.label],
                                                                        y_pred=[int(i > thresh) for i in
                                                                                list(self.train_y_pred)])]
                                                  }, index=["train"]), report_df])

        if self.df_test_woe.any().any():
            x_data = self.df_test_woe[self.final_features]
            x_data = self._check_row_na(x_data)
            self.df_test_woe = self.df_test_woe[self.df_test_woe.index.isin(x_data.index.tolist())]
            x_data = sm.add_constant(x_data)
            self.test_y_pred = self.model.predict(x_data).tolist()
            get_curve(y_true=self.df_test_woe[self.label], y_pred=self.test_y_pred,
                      file_name="{}_test_report.png".format(self.filename))
            get_cm(y_true=self.df_test_woe[self.label], y_pred=self.test_y_pred, thresh=thresh)
            report_df = pd.concat([pd.DataFrame(data={"precision": [
                precision_score(y_true=self.df_test_woe[self.label],
                                y_pred=[int(i > thresh) for i in list(self.test_y_pred)])],
                "recall": [recall_score(y_true=self.df_test_woe[self.label],
                                        y_pred=[int(i > thresh) for i in
                                                list(self.test_y_pred)])],
                "accuracy": [accuracy_score(y_true=self.df_test_woe[self.label],
                                            y_pred=[int(i > thresh) for i in
                                                    list(self.test_y_pred)])],
                "f1_score": [f1_score(y_true=self.df_test_woe[self.label],
                                      y_pred=[int(i > thresh) for i in
                                              list(self.test_y_pred)])]
            }, index=["test"]), report_df])

        if self.df_ott_woe.any().any():
            x_data = self.df_ott_woe[self.final_features]
            x_data = self._check_row_na(x_data)
            self.df_ott_woe = self.df_ott_woe[self.df_ott_woe.index.isin(x_data.index.tolist())]
            x_data = sm.add_constant(x_data)
            self.ott_y_pred = self.model.predict(x_data).tolist()
            get_curve(y_true=self.df_ott_woe[self.label], y_pred=self.ott_y_pred,
                      file_name="{}_ott_report.png".format(self.filename))
            get_cm(y_true=self.df_ott_woe[self.label], y_pred=self.ott_y_pred, thresh=thresh)
            report_df = pd.concat([pd.DataFrame(data={"precision": [
                precision_score(y_true=self.df_ott_woe[self.label],
                                y_pred=[int(i > thresh) for i in list(self.ott_y_pred)])],
                "recall": [recall_score(y_true=self.df_ott_woe[self.label],
                                        y_pred=[int(i > thresh) for i in
                                                list(self.ott_y_pred)])],
                "accuracy": [accuracy_score(y_true=self.df_ott_woe[self.label],
                                            y_pred=[int(i > thresh) for i in
                                                    list(self.ott_y_pred)])],
                "f1_score": [f1_score(y_true=self.df_ott_woe[self.label],
                                      y_pred=[int(i > thresh) for i in
                                              list(self.ott_y_pred)])]
            }, index=["ott"]), report_df])

        params = self.model.params.reset_index()
        params.columns = ["var", "coef"]
        get_feature_importance(feature=list(params["var"])[1:], importance=list(params["coef"])[1:],
                               filename="{}_feature_importance.png".format(self.filename))
        print(report_df)

    def _check_row_na(self, data):
        rows_null_percent = data.isnull().sum(axis=1) / data.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent > 0]
        to_drop_rows = to_drop_rows.to_dict()
        if to_drop_rows:
            self.log.info(
                "该数据集集有{}个样本仍含有缺失值，暂删除，占比{}".format(len(to_drop_rows), round(len(to_drop_rows) / data.shape[0], 4)))
        data = data[~data.index.isin(to_drop_rows.keys())]
        return data

    def get_score_table(self):
        y_1 = self.df_train_woe[self.label].value_counts()[1]
        y_0 = self.df_train_woe[self.label].value_counts()[0]

        odds = float(y_1) / float(y_0)
        self.B = self.double_score / math.log(2)
        self.A = self.base_score + self.B * math.log(odds)

        writer = pd.ExcelWriter("{}_model_result.xlsx".format(self.filename))

        train_res = self._get_table(y_true=list(self.df_train_woe[self.label]), y_pred=list(self.train_y_pred),
                                    flag="qcut")
        train_res.to_excel(writer, "amount_train")
        if self.df_test_woe.any().any():
            test_res = self._get_table(y_true=list(self.df_test_woe[self.label]), y_pred=list(self.test_y_pred),
                                       flag="qcut")
            test_res.to_excel(writer, "amount_test")
        if self.df_ott_woe.any().any():
            ott_res = self._get_table(y_true=list(self.df_ott_woe[self.label]), y_pred=list(self.ott_y_pred),
                                      flag="qcut")
            ott_res.to_excel(writer, "amount_ott")

        train_res = self._get_table(y_true=list(self.df_train_woe[self.label]), y_pred=list(self.train_y_pred),
                                    flag="cut")
        train_res.to_excel(writer, "score_train")
        if self.df_test_woe.any().any():
            test_res = self._get_table(y_true=list(self.df_test_woe[self.label]), y_pred=list(self.test_y_pred),
                                       flag="cut")
            test_res.to_excel(writer, "score_test")
        if self.df_ott_woe.any().any():
            ott_res = self._get_table(y_true=list(self.df_ott_woe[self.label]), y_pred=list(self.ott_y_pred),
                                      flag="cut")
            ott_res.to_excel(writer, "score_ott")
        writer.save()
        writer.close()

    def _get_table(self, y_true, y_pred, flag="qcut"):
        rs = pd.DataFrame({'p': y_pred, 'flag': y_true})
        rs.loc[:, "p_"] = 1 - rs["p"]
        rs.loc[:, 'log_odds'] = np.log(rs.loc[:, 'p'] / rs.loc[:, 'p_'])
        rs.loc[:, 'score'] = rs.apply(lambda x: self.A - self.B * np.log(x['p'] / x['p_']), axis=1)
        rs['score'] = rs['score'].astype(int)

        rs = rs[["flag", "score"]]

        if flag == "qcut":
            if not hasattr(self, 'qcut_point'):
                rs["score_bin"] = pd.qcut(rs["score"].tolist(), self.cut_num, duplicates='drop')
                score_value = [str(x).replace('(', '').replace('[', '').replace(')', '').replace(']', '').split(',') for
                               x in
                               rs['score_bin'].tolist()]
                score_value_list = [(float(x[0]), float(x[1])) for x in score_value]
                self.qcut_point = score_value_list
            else:
                rs["score_bin"] = rs.score.map(
                    lambda x: self.qcut_point[int(np.argmax([x > i[0] and x <= i[1] for i in self.qcut_point]))])
                rs["score_bin"] = rs.score_bin.map(lambda x: str(x).replace(")", "]"))
        elif flag == "cut":
            if not hasattr(self, 'cut_point'):
                rs["score_bin"] = pd.cut(rs["score"].tolist(), self.cut_num, include_lowest=True)
                score_value = [str(x).replace('(', '').replace('[', '').replace(')', '').replace(']', '').split(',') for
                               x in
                               rs['score_bin'].tolist()]
                score_value_list = [(float(x[0]), float(x[1])) for x in score_value]
                self.cut_point = score_value_list
            else:
                rs["score_bin"] = rs.score.map(
                    lambda x: self.cut_point[int(np.argmax([x > i[0] and x <= i[1] for i in self.cut_point]))])
                rs["score_bin"] = rs.score_bin.map(lambda x: str(x).replace(")", "]"))

        rs = pd.crosstab(rs.score_bin, rs.flag, rownames=['score_bin'], colnames=['flag']).reset_index()
        rs.columns = ["score_bin", "good", "bad"]
        rs["min_score"] = rs.score_bin.map(
            lambda x: float(str(x).replace('(', '').replace('[', '').replace(')', '').replace(']', '').split(',')[0]))
        rs["max_score"] = rs.score_bin.map(
            lambda x: float(str(x).replace('(', '').replace('[', '').replace(')', '').replace(']', '').split(',')[1]))
        rs["pred_min"] = rs.max_score.map(lambda x: round(math.exp((self.A - x) / self.B) / (math.exp((self.A - x) / self.B) + 1), 4))
        rs["pred_max"] = rs.min_score.map(lambda x: round(math.exp((self.A - x) / self.B) / (math.exp((self.A - x) / self.B) + 1), 4))
        rs["pred_bin"] = rs.apply(lambda x: "[" + str(x["pred_min"]) + "," + str(x["pred_max"]) + ")", axis=1)
        rs = rs[["pred_bin", "score_bin", "good", "bad"]]

        total_good = rs['good'].sum()
        total_bad = rs['bad'].sum()
        rs['good_per'] = rs['good'] / total_good
        rs['bad_per'] = rs['bad'] / total_bad
        rs['cum_good_per'] = np.cumsum(rs['good_per'], axis=0)
        rs['cum_bad_per'] = np.cumsum(rs['bad_per'], axis=0)
        rs['total'] = rs['good'] + rs['bad']
        rs['bad_rate'] = rs['bad'] / rs['total']
        rs['KS'] = abs(rs['cum_bad_per'] - rs['cum_good_per'])

        AUC = []
        AUC.append(0.5 * rs['cum_bad_per'].loc[0] * rs['cum_good_per'].loc[0])
        for i in range(1, len(rs)):
            value = 0.5 * float(rs['cum_bad_per'].loc[i] + rs['cum_bad_per'].loc[i - 1]) * float(
                rs['cum_good_per'].loc[i] - rs['cum_good_per'].loc[i - 1])
            AUC.append(value)
        rs['AUC'] = pd.Series(AUC, index=rs.index)
        return rs

    @staticmethod
    def _draw_report(y_true, y_pred, file_name):
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.subplot(121)
        plt.xlabel('Percentage', fontsize=15)
        plt.ylabel('tpr / fpr', fontsize=15)
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.title("ks-curve", fontsize=20)

        percentage = np.round(np.array(range(1, len(fpr) + 1)) / len(fpr), 4)
        ks_delta = tpr - fpr
        ks_index = ks_delta.argmax()
        plt.plot([percentage[ks_index], percentage[ks_index]],
                 [tpr[ks_index], fpr[ks_index]],
                 color='limegreen', lw=2, linestyle='--')
        plt.text(percentage[ks_index] + 0.02, (tpr[ks_index] + fpr[ks_index]) / 2,
                 'ks: {0:.4f}'.format(ks_delta[ks_index]),
                 fontsize=13)
        plt.plot(percentage, tpr, color='dodgerblue', lw=2, label='tpr')
        plt.plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
        plt.plot(percentage, fpr, color='tomato', lw=2, label='fpr')
        plt.legend(fontsize='x-large')

        plt.subplot(122)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")

        plt.savefig(file_name)
        plt.show()


    def feature_report(self):
        writer = pd.ExcelWriter("{}_feature.xlsx".format(self.filename))
        i = 0
        res_param = self.model.params
        nn = self.model.params.shape[0]
        woe_df = self.woe_df[self.woe_df['feature'].isin(self.final_features)]
        woe_df['woe'].replace(float('inf'), 0, inplace=True)
        woe_df['woe'].replace(float('-inf'), 0, inplace=True)
        for var in self.final_features:
            tmp = woe_df[woe_df['feature'] == var]
            tmp['score'] = (self.A - self.B * res_param['const']) / (nn - 1) - res_param[var] * tmp['woe'] * self.B
            tmp.to_excel(writer, 'model_var_score', startrow=i)
            len_df = tmp.shape[0] + 1
            i += len_df + 2

        woe_df['iv'].replace(float('inf'), 0, inplace=True)
        woe_df['iv'].replace(float('-inf'), 0, inplace=True)
        tmp1 = woe_df["iv"].groupby(woe_df["feature"]).agg(["sum"]).reset_index()
        tmp2 = woe_df["ks"].groupby(woe_df["feature"]).agg(["max"]).reset_index()
        tmp = pd.merge(tmp1, tmp2, how="inner", on="feature")
        tmp.columns = ["feature", "iv", "ks"]
        tmp.to_excel(writer, 'model_bin_summary')
        writer.save()
        writer.close()


    def main(self):
        self.check_woe_value()
        self.log.info("相关性检验, max_corr={}".format(self.max_corr))
        drop_set = self.check_corr()
        if drop_set:
            self.log.info("已过滤{}个特征".format(len(drop_set)))
            self.del_col.extend(list(drop_set))
            self.final_features = [i for i in list(self.iv_df.feature) if i not in self.del_col]
        self.log.info("*"*30)

        self.log.info("VIF检验, max_vif={}".format(self.max_vif))
        drop_col = self.check_vif()
        if drop_col:
            self.log.info("已过滤{}个特征".format(len(drop_col)))
            self.final_features = [i for i in self.final_features if i not in drop_col]
            self.del_col.append(drop_col)
        self.log.info("*" * 30)

        self.log.info("逐步回归挑选特征, 同时检验符号异向性")
        self.final_features = self.check_stepwise()
        self.log.info(("剩余{}个特征可以进入模型，分别是{}".format(len(self.final_features), self.final_features)))
        self.log.info("*" * 30)

        self.log.info("激动人心，要建模啦！！")
        self.model_build()
        self.log.info("训练集预测结果，请查看：{}.train_y_pred".format(self.__class__.__name__))
        if self.df_test_woe.any().any():
            self.log.info("测试集预测结果，请查看：{}.test_y_pred".format(self.__class__.__name__))
        if self.df_ott_woe.any().any():
            self.log.info("ott预测结果，请查看：{}.ott_y_pred".format(self.__class__.__name__))

        self.log.info("***************基本评估指标图示如下")
        self.pred_evaluate()
        self.log.info("*" * 30)

        self.log.info("开始应付爸爸，分数观测结果，分为{}段进行观测".format(self.cut_num))
        self.get_score_table()
        self.log.info("分数均分与人数均分两种观测方式，均以train-data-set为标准")
        self.log.info("最终特征说明保存在{}_model_result.xlsx".format(self.filename))
        self.log.info("*" * 30)

        self.feature_report()
        self.log.info("最终特征说明保存在{}_feature.xlsx".format(self.filename))