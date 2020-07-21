#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 14:51
# @Author  : cai
# @contact : chenyuwei_0303@yeah.net
# @File    : MonotonicWoe.py
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
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from BaseUtils import log


class MonotonicWoe(object):
    def __init__(self, df_train, target_name, del_col, dist_col, serial_col, df_test=None, df_ott=None, file_name=None,
                 na_value=None, max_bins=5, min_rate=0.01, min_bins_cnt=50, max_process=2):
        self.log = log

        """
        df_train:       训练集
        target_name:    y
        del_col:        不需要分箱的列
        dist_col:       离散变量
        serial_col:     连续变量
        df_test:        测试集
        df_ott:         跨时间验证的数据集
        file_name:      文件名字
        na_value:       缺失值
        max_bins:       最大分箱数
        min_rate:       最小分箱占比
        min_bins_cnt:   每箱最少绝对值数量
        max_process:    最大进程数
        """

        # 检查y列
        if not target_name:
            self.log.info("爸，你清醒一点，没有lable列啦！")
            sys.exit(0)
        else:
            self.target_name = target_name
            del_col.append(target_name)

        # 检查特征列
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
        # 留一个口子，多类缺失值
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

        if file_name:
            self.filename = file_name
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

    # 离散变量woe转化，针对self.dist_col
    def dist_woe_caculation(self):
        data = self.df_train[self.dist_col + [self.target_name] + ["index"]]
        bad_count = self.df_train[self.target_name].sum()
        good_count = len(self.df_train) - bad_count

        # 离散变量的分箱结果
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
                lambda x: float("inf") if x['bad_pct'] == 0 else float("-inf") if x['good_pct'] == 0 else np.log(x['good_pct'] / x['bad_pct']), axis=1)
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

        del data

    # 连续变量woe转化，针对self.serial_col
    def serial_woe_caculation(self):
        self.log.info("仰天长笑，哈哈哈，CPU要飞起来啦！！！请做好降温防暑工作")

        with Manager() as manager:
            m = manager.dict()
            cons = manager.dict()
            # fill_woe_value = manager.dict()

            p = Pool(self.pull_num)
            for col in self.serial_col:
                p.apply_async(self._multi_woe, args=(col, m, cons))
            p.close()
            p.join()

            """
            best_knots_df：    每个特征的分箱点；key值为特征，value值分箱点的index值
            conditions：       映射条件句，key值为特征，value值为np.where执行语句
            # fill_values：      处理na_value时出现的inf问题，key值为特征，value值为该箱中的y情况
            #                    特征若为空值的样本单独成一箱，这一箱在y上有三种表现：
            #                    1. 全部为good - fill_values的value值为0 - woe取float("inf")
            #                    2. 全部为bad - fill_values的value值为1 - woe取float("-inf")
            #                    3. 有正有负，有正常的woe值
            """
            best_knots_df = dict(m)
            conditions = dict(cons)
            # fill_values = dict(fill_woe_value)

        # 连续变量的分箱结果
        self.serial_df = pd.DataFrame(
            columns=["feature", "bin_name", "bad", "bad_rate", "good", "good_pct", "bad_pct", "woe", "iv", "ks"])
        for key, value in best_knots_df.items():
            self.serial_df = pd.concat([self.serial_df, value[
                ["feature", "bin_name", "bad", "bad_rate", "good", "good_pct", "bad_pct", "woe", "iv", "ks"]]])
            data = self.df_train[[key]]
            data[key] = data[key].astype(float)
            self.df_train_woe[key] = eval(conditions[key])
            # if self.df_train_woe[key].isnull().max():
            #     self.df_train_woe[key] = self.df_train_woe[key].fillna(float("-inf") if fill_values[key] == 1 else float("inf"))
            if self.df_test.any().any():
                data = self.df_test[[key]]
                data[key] = data[key].astype(float)
                self.df_test_woe[key] = eval(conditions[key])
                # if self.df_test_woe[key].isnull().max():
                #     self.df_test_woe[key] = self.df_test_woe[key].fillna(float("-inf") if fill_values[key] == 1 else float("inf"))
            if self.df_ott.any().any():
                data = self.df_ott[[key]]
                data[key] = data[key].astype(float)
                self.df_ott_woe[key] = eval(conditions[key])
                # if self.df_ott_woe[key].isnull().max():
                #     self.df_ott_woe[key] = self.df_ott_woe[key].fillna(float("-inf") if fill_values[key] == 1 else float("inf"))


    def _multi_woe(self, col, m, cons):
        data = self.df_train[[col, self.target_name]]
        # self.log.info('子进程: {} - 特征{}'.format(os.getpid(), col))

        cut_point = []
        work_data = data[self.target_name].groupby([data[col], data[self.target_name]]).count()
        work_data = work_data.unstack().reset_index().fillna(0)
        work_data.columns = [col, 'good', 'bad']

        """
        na_df：     特征为空的情况
        non_na_df： 特征不为空的情况，针对这部分进行分箱
        分箱基本思路：
        1. 以non_na_df的index键作为主键进行识别，将连续变量视为离散变量
        2. 计算每个离散点符合各个分箱条件的ks值
        3. 对最后所有的ks值进行全排序，找出最大iv的组合
        """
        na_df = work_data[work_data[col].isin(self.na_value_list)]

        non_na_df = work_data[~work_data[col].isin(self.na_value_list)]
        non_na_df[col] = non_na_df[col].astype(float)
        non_na_df = non_na_df.sort_values(by=[col], ascending=True)
        non_na_df = non_na_df.reset_index(drop=True)

        # 对non_na_df进行处理
        total_len = sum(work_data['good']) + sum(work_data['bad'])
        current_rate = min(self.min_rate, self.min_bins_cnt / total_len)
        tmp_result = self._best_ks_knots(non_na_df, total_len=total_len, current_rate=current_rate, start_knot=0,
                                         end_knot=non_na_df.shape[0], current_time=0)
        tmp_result = [x for x in tmp_result if x is not None]
        tmp_result.sort()
        return_piece_num = min(self.max_bins, len(tmp_result) + 1)

        # cost a lot time
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
            # if not (na_df.values[0][1] == 0 or na_df.values[0][2] == 0):
            cut_point.append(self.na_value)
            bin_name.append(str(self.na_value))
            good.append(na_df.values[0][1])
            bad.append(na_df.values[0][2])
            # elif na_df.values[0][1] > 0:
            #     # 所有都是good样本
            #     # fill_woe_value[col] = 0
            # elif na_df.values[0][2] > 0:
            #     # 所有都是bad样本
            #     fill_woe_value[col] = 1

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
                    lambda x: float("inf") if x['bad_pct'] == 0 else float("-inf") if x['good_pct'] == 0 else np.log(x['good_pct'] / x['bad_pct']), axis=1)
                tmp["ks"] = np.abs(tmp.good_pct - tmp.bad_pct)
                tmp['iv'] = (tmp['good_pct'] - tmp['bad_pct']) * tmp['woe']
                tmp["feature"] = col
                m[col] = tmp

                # 组中conditions
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
        else:
            self.log.info("！！！！！！！！！！该特征{}无法进行有效分箱".format(col))

    # 迭代找到最优分割点
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

    # 计算IV
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

    def output_bin_file(self):
        dist_df = self.dist_woe_df.copy()
        dist_df["ks"] = 0
        serial_df = self.serial_df.copy()
        woe_df = pd.concat([dist_df[["feature", "bin_name", "woe", "iv", "ks", "bad", "good", "bad_rate"]],
                            serial_df[["feature", "bin_name", "woe", "iv", "ks", "bad", "good", "bad_rate"]]])
        self.woe_df = pd.DataFrame(columns=["feature", "bin_name", "woe", "iv", "ks", "bad", "good", "bad_rate"])
        writer = pd.ExcelWriter("{}_bin_result.xlsx".format(self.filename))
        i = 0
        for var in list(woe_df.feature.unique()):
            tmp = woe_df[woe_df['feature'] == var]
            woe_list = tmp["woe"].tolist()
            if float("inf") in woe_list or float("-inf") in woe_list:
                woe_list = [i for i in woe_list if i not in [float("inf"), float("-inf")]]
            min_woe, max_woe = min(woe_list), max(woe_list)
            tmp["iv"] = tmp.iv.map(lambda x: 0 if x in([float("inf"), float("-inf")]) else x)
            tmp["woe"] = tmp.woe.map(lambda x: min_woe if x == float("-inf") else max_woe if x == float("inf") else x)
            tmp.to_excel(writer, 'allBin', startrow=i)
            len_df = tmp.shape[0] + 1
            i += len_df + 2
            self.woe_df = pd.concat(self.woe_df, tmp)

        self.dist_woe_df.to_excel(writer, 'DistBin')
        self.serial_df.to_excel(writer, 'SerialBin')
        writer.save()
        writer.close()


    def main(self):

        stime = datetime.datetime.now()
        self.log.info(">>>>>>>>>>>>>>>>>离散变量分箱开始<<<<<<<<<<<<<<")
        self.dist_woe_caculation()
        self.log.info("The dist_col woe result can be checked {}.dist_woe_df".format(self.__class__.__name__))

        self.log.info(">>>>>>>>>>>>>>>>>连续变量分箱开始<<<<<<<<<<<<<<")
        self.serial_woe_caculation()
        etime = datetime.datetime.now()
        self.log.info("最耗时part已结束！哇o～，cost time {} seconds.".format((etime-stime).seconds))
        self.log.info("The serial_col woe result can be checked {}.serial_df".format(self.__class__.__name__))

        self.log.info(">>>>>>>>>>>>>>>>>WOE映射开始<<<<<<<<<<<<<<")
        self.df_train_woe.to_csv("{}_train_woe.csv".format(self.filename), sep="|", index=False)
        self.log.info("训练集的数据保存在{}_train_woe.csv".format(self.filename))

        if self.df_test.any().any():
            self.df_test_woe.to_csv("{}_test_woe.csv".format(self.filename), sep="|", index=False)
            self.log.info("测试集的数据保存在{}_test_woe.csv".format(self.filename))

        if self.df_ott.any().any():
            self.df_ott_woe.to_csv("{}_ott_woe.csv".format(self.filename), sep="|", index=False)
            self.log.info("ott的数据保存在{}_ott_woe.csv".format(self.filename))

        self.log.info(">>>>>>>>>>>>>>>>合并分箱文件开始<<<<<<<<<<<<<<")
        self.log.info(">>>>>>>>>>>>>>>>该part会改变woe中inf与-inf的问题<<<<<<<<<<<<<<")
        self.output_bin_file()
        self.log.info("所有分箱结果可以在{}_bin_result.xlsx中查看".format(self.filename))
        self.log.info("sheet_name = DistBin为离散特征原始分箱结果")
        self.log.info("sheet_name = SerialBin为连续特征原始分箱结果")
        self.log.info("sheet_name = allBin为处理过的特征分箱结果，主要是处理了inf问题")



