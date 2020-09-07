#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 09:37
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : FastScToolkit.py.py
# @Note    :


import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import datetime

from BaseUtils import log
import scorecardpy as sc
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression

class FastScWoe(object):
    def __init__(self, df_train, target_name, dist_col, serial_col, df_test=None, df_ott=None, file_name=None,
                 max_bins=5, method="tree", max_corr=0.75, max_vif=10, cut_nums=50, base_score=500, double_score=20):
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
        max_bins:       最大分箱数
        method:         分箱方法，tree或者chimerge
        """

        # 检查y列
        if not target_name:
            self.log.info("爸，你清醒一点，没有lable列啦！")
            sys.exit(0)
        else:
            self.target_name = target_name

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

        self._check_y(df_train)
        df_train = df_train[self.dist_col + self.serial_col + [self.target_name]]
        self.df_train = self._format_df(df_train)
        self.df_train_woe = self.df_train[[self.target_name]]

        if df_test is not None and isinstance(df_test, pd.DataFrame):
            self._check_y(df_test)
            df_test = df_test[self.dist_col + self.serial_col + [self.target_name]]
            self.df_test = self._format_df(df_test)
            self.df_test_woe = self.df_test[[self.target_name]]
        else:
            self.df_test = pd.DataFrame()
            self.log.info("啊哦，没有测试训练集～")

        if df_ott is not None and isinstance(df_ott, pd.DataFrame):
            self._check_y(df_ott)
            df_ott = df_ott[self.dist_col + self.serial_col + [self.target_name]]
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
        self.method = method

        self.max_corr = max_corr
        self.max_vif = max_vif

        self.base_score = base_score
        self.double_score = double_score

        self.cut_nums = cut_nums

    def _check_y(self, tmp_df):
        if len(tmp_df[self.target_name].unique()) != 2:
            self.log.info("旁友，你有点多心哦，本对象只适用于二分类！")

    def _format_df(self, df):
        df = pd.DataFrame(df, dtype=str)
        df[self.target_name] = df[self.target_name].apply(int)
        for col in df.columns:
            if col in self.dist_col:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
            elif col in self.serial_col:
                df[col] = df[col].astype(float)
        return df

    def cut_main(self):
        breaks_adj = {}
        for col in self.dist_col:
            if len([i for i in list(self.df_train[col].unique()) if i == i]) <= 1:
                self.log.info("{} has been deleted, because only has one single value".format(col))
                continue
            breaks_adj[col] = [i for i in list(self.df_train[col].unique()) if i == i]

        self.dist_col = list(breaks_adj.keys())
        cut_cols = self.dist_col + self.serial_col
        max_bins = self.max_bins
        self.woe_df = pd.DataFrame(columns=["variable", "bin", "woe", "bin_iv", "bad", "badprob"])
        self.bins_adj = {}
        while max_bins > 2 and len(cut_cols) > 0:
            self.log.info("现在是分{}箱, 有{}个变量需要分箱".format(max_bins, len(cut_cols)))
            if self.target_name not in cut_cols:
                cut_cols.append(self.target_name)
            bins_adj = sc.woebin(self.df_train[cut_cols], y=self.target_name, breaks_list=breaks_adj,
                                 bin_num_limit=max_bins, method=self.method)
            cut_cols = []
            for key, value in bins_adj.items():
                tmp = bins_adj[key].copy()
                if key in self.serial_col:
                    tmp1 = tmp[tmp.bin != "missing"]
                    if len(tmp1) == 1:
                        continue
                    else:
                        tmp1["min"] = tmp1.bin.map(
                            lambda x: float(x.split(",")[0].replace("[", "")) if x.find(",") > -1 else x)
                        tmp1 = tmp1.sort_values(by="min")
                        if not all(x < y for x, y in zip(tmp1.woe.tolist(), tmp1.woe.tolist()[1:])):
                            cut_cols.append(key)
                            continue
                self.woe_df = pd.concat([self.woe_df, tmp[["variable", "bin", "woe", "bin_iv", "bad", "badprob"]]])
            max_bins = max_bins - 1
            self.bins_adj = dict(self.bins_adj, **bins_adj)
        self.log.info("仍有{}个特征无法满足单调的需求，不能分箱".format(len(cut_cols)))
        self.woe_df.to_excel("{}_bin_result.xlsx".format(self.filename))
        self.log.info("*" * 50)
        self.log.info("WOE Detail can be checked in {}_bin_result.xlsx".format(self.filename))

        self.df_train_woe = sc.woebin_ply(self.df_train, self.bins_adj)
        self.df_train_woe.columns = [i.replace("_woe", "") for i in self.df_train_woe.columns]
        if self.df_test.any().any():
            self.df_test_woe = sc.woebin_ply(self.df_test, self.bins_adj)
            self.df_test_woe.columns = [i.replace("_woe", "") for i in self.df_test_woe.columns]
        if self.df_ott.any().any():
            self.df_ott_woe = sc.woebin_ply(self.df_ott, self.bins_adj)
            self.df_ott_woe.columns = [i.replace("_woe", "") for i in self.df_ott_woe.columns]
        self.log.info("CutBin has finished!")

    def _check_row_na(self, data):
        rows_null_percent = data.isnull().sum(axis=1) / data.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent > 0]
        to_drop_rows = to_drop_rows.to_dict()
        return to_drop_rows

    def _check_corr(self):
        self.final_features = self.iv_df[self.iv_df.iv >= 0.02].variable.tolist()
        data = self.df_train_woe[self.final_features].copy()


        drop_set, var_set = set(), set(self.final_features)

        for i in list(self.final_features):
            if i in drop_set:
                continue
            else:
                drop_set |= {v for v in var_set if
                             np.corrcoef(data[i].values, data[v].values)[0, 1] >= self.max_corr and v != i}
                var_set -= drop_set
        self.log.info("根据iv删除{}个相关性较高的特征".format(len(drop_set)))
        self.final_features = [i for i in self.final_features if i not in drop_set]

    def _check_vif(self):
        drop_col = []
        data = self.df_train_woe[self.final_features].copy()
        while 1:
            cols = [i for i in self.final_features if i not in drop_col]
            tmp = data[cols]
            col = list(range(tmp.shape[1]))
            vif_df = pd.DataFrame()
            vif_df['vif_value'] = [variance_inflation_factor(tmp.iloc[:, col].values, ix) for ix in
                                   range(tmp.iloc[:, col].shape[1])]
            vif_df['variable'] = tmp.columns

            vif_df = pd.merge(vif_df, self.iv_df, how="inner", on="variable")
            vif_df = vif_df[vif_df.vif_value > self.max_vif]
            if len(vif_df) <= 0:
                break
            col = vif_df.sort_values(by="iv").feature.values[0]
            drop_col.append(col)
            self.log.info("The vif value is not accepted, delete the col={}".format(col))
        self.final_features = [i for i in self.final_features if i not in drop_col]

    def _check_stepwise(self, class_weights):
        y_train = self.df_train_woe[self.target_name]

        tmp = self.iv_df[self.iv_df.variable.isin(self.final_features)].sort_values(by="iv", ascending=False)
        sorted_cols = list(tmp.variable)
        train_cols = []
        for i in sorted_cols:
            train_cols.append(i)
            x_train = self.df_train_woe[train_cols]
            lr = LogisticRegression(penalty='l2', C=1, solver='saga', n_jobs=-1, class_weight={0: class_weights[0], 1:class_weights[1]})
            lr.fit(x_train, y_train)
            importance = {x: y for x, y in zip(train_cols, lr.coef_[0])}
            to_drop_cols = [k for k, v in importance.items() if v < 0]
            train_cols = [i for i in train_cols if i not in to_drop_cols]
        self.final_features = train_cols

    def _get_score_table(self, rs, a, b):
        if not hasattr(self, 'cut_point'):
            rs["pred_bin"] = pd.cut(rs["pred"].tolist(), self.cut_nums, include_lowest=True)
            pred_point = [str(x).replace('(', '').replace('[', '').replace(')', '').replace(']', '').split(',') for
                           x in rs['pred_bin'].tolist()]
            pred_point = [(float(x[0]), float(x[1])) for x in pred_point]
            self.pred_point = pred_point
        else:
            rs["pred_bin"] = rs.pred.map(
                lambda x: self.pred_point[int(np.argmax([x > i[0] and x <= i[1] for i in self.pred_point]))])
            rs["pred_bin"] = rs.pred_bin.map(lambda x: str(x).replace(")", "]"))

        rs = pd.crosstab(rs.pred_bin, rs.flag, rownames=['pred_bin'], colnames=['flag']).reset_index()
        rs.columns = ["pred_bin", "good", "bad"]
        rs["pred_min"] = rs.pred_bin.map(lambda x: float(str(x).replace("(", "").replace("]", "").split(",")[0]))
        rs["pred_max"] = rs.pred_bin.map(lambda x: float(str(x).replace("(", "").replace("]", "").split(",")[1]))

        rs["score_max"] = rs.pred_min.map(lambda x: a - b * np.log(x / (1 - x)))
        rs["score_min"] = rs.pred_max.map(lambda x: a - b * np.log(x / (1 - x)))

        rs["score_bin"] = rs.apply(lambda x: "(" + str(round(x.score_max, 2)) + "," + str(round(x.score_min, 2)) + "]",
                                   axis=1)

        rs = rs.sort_values(by="pred_bin", ascending=False)
        total_good = rs['good'].sum()
        total_bad = rs['bad'].sum()
        rs['good_per'] = rs['good'] / total_good
        rs['bad_per'] = rs['bad'] / total_bad
        rs['cum_good_per'] = np.cumsum(rs['good_per'], axis=0)
        rs['cum_bad_per'] = np.cumsum(rs['bad_per'], axis=0)
        rs['total'] = rs['good'] + rs['bad']
        rs['bad_rate'] = rs['bad'] / rs['total']
        rs['KS'] = round(abs(rs['cum_bad_per'] - rs['cum_good_per']), 4)

        rs = rs[["pred_bin", "score_bin", "total", "bad", "bad_rate", "cum_bad_per", "KS"]]
        return rs

    def model_main(self):
        self.iv_df = self.woe_df["bin_iv"].groupby([self.woe_df.variable]).agg(["sum"]).reset_index()
        self.iv_df.columns = ["variable", "iv"]
        self.iv_df = self.iv_df.sort_values(by="iv", ascending=False)
        self.log.info("有{}个特征iv值小于0.02".format(len(self.iv_df[self.iv_df.iv < 0.02])))
        if self.df_test.any().any():
            to_drop_rows = self._check_row_na(self.df_test_woe)
            if len(to_drop_rows) > 0:
                self.log.info(
                    "TestData有{}个样本仍含有缺失值，暂删除，占比{}".format(len(to_drop_rows), round(len(to_drop_rows) / self.df_test_woe.shape[0], 4)))
                self.df_test = self.df_test[~self.df_test.index.isin(to_drop_rows.keys())]
                self.df_test_woe = self.df_test_woe[~self.df_test_woe.index.isin(to_drop_rows.keys())]

        if self.df_ott.any().any():
            to_drop_rows = self._check_row_na(self.df_ott_woe)
            if len(to_drop_rows) > 0:
                self.log.info(
                    "OttData有{}个样本仍含有缺失值，暂删除，占比{}".format(len(to_drop_rows),
                                                        round(len(to_drop_rows) / self.df_ott_woe.shape[0], 4)))
                self.df_ott = self.df_ott[~self.df_ott.index.isin(to_drop_rows.keys())]
                self.df_ott_woe = self.df_ott_woe[~self.df_ott_woe.index.isin(to_drop_rows.keys())]

        self._check_corr()
        self._check_vif()

        class_weights = compute_class_weight('balanced', [0, 1], self.df_train_woe.y)
        self._check_stepwise(class_weights)

        X_train = self.df_train_woe[self.final_features]
        y_train = self.df_train_woe[self.target_name]

        lr = LogisticRegression(penalty='l2', C=1, solver='saga', n_jobs=-1, class_weight={0: class_weights[0], 1: class_weights[1]})
        lr.fit(X_train, y_train)
        self.model = lr

    def evaluate_main(self):
        writer = pd.ExcelWriter("{}_report.xlsx".format(self.filename))

        odds0 = float(self.df_train_woe[self.target_name].value_counts()[1]) / float(self.df_train_woe[self.target_name].value_counts()[0])
        b = self.double_score / np.log(2)
        a = self.base_score + b * np.log(odds0)
        card = sc.scorecard(self.bins_adj, self.model, self.final_features, points0=self.base_score,
                            odds0=odds0,
                            pdo=self.double_score)
        card_df = pd.DataFrame(columns=["variable", "bin", "points"])
        for key, value in card.items():
            card_df = pd.concat([card_df, value])
        card_df.to_excel(writer, 'card_result')

        self.train_pred = self.model.predict_proba(self.df_train_woe[self.final_features])[:, 1]
        perf = sc.perf_eva(self.df_train_woe[self.target_name], self.train_pred, title="train")
        print("On train-data, the evaluation follows:\nks={}, auc={}, gini={}".format(perf["KS"], perf["AUC"],
                                                                                      perf["Gini"]))
        perf["pic"].savefig("{}_train.png".format(self.filename))

        _score = sc.scorecard_ply(self.df_train, card, print_step=0)
        _score["flag"] = self.df_train_woe[self.target_name]
        _score["pred"] = self.train_pred

        _rs = self._get_score_table(_score, a, b)
        _rs.to_excel(writer, 'train_result')

        if self.df_test.any().any():
            y_test = self.df_test_woe[self.target_name]
            self.test_pred = self.model.predict_proba(self.df_test_woe[self.final_features])[:, 1]
            perf = sc.perf_eva(y_test, self.test_pred, title="test")
            print("On test-data, the evaluation follows:\nks={}, auc={}, gini={}".format(perf["KS"],
                                                                                          perf["AUC"],
                                                                                          perf["Gini"]))
            perf["pic"].savefig("{}_test.png".format(self.filename))

            _score = sc.scorecard_ply(self.df_test, card, print_step=0)
            _score["flag"] = self.df_test_woe[self.target_name]
            _score["pred"] = self.test_pred

            _rs = self._get_score_table(_score, a, b)
            _rs.to_excel(writer, 'test_result')


        if self.df_ott.any().any():
            y_ott = self.df_ott_woe[self.target_name]
            self.ott_pred = self.model.predict_proba(self.df_ott_woe[self.final_features])[:, 1]
            try:
                perf = sc.perf_eva(y_ott, self.ott_pred , title="ott")
                print("On ott-data, the evaluation follows:\nks={}, auc={}, gini={}".format(perf["KS"],
                                                                                        perf["AUC"],
                                                                                        perf["Gini"]))
                perf["pic"].savefig("{}_test.png".format(self.filename))

                _score = sc.scorecard_ply(self.df_ott, card, print_step=0)
                _score["flag"] = self.df_ott_woe[self.target_name]
                _score["pred"] = self.ott_pred

                _rs = self._get_score_table(_score,a,b)
                _rs.to_excel(writer, 'ott_result')

            except:
                self.log.info("Cannot caculation the ott data!")


        importance = {x: y for x, y in zip(self.final_features, self.model.coef_[0])}

        iv_df = self.iv_df[self.iv_df['variable'].isin(self.final_features)]
        iv_df["coef"] = iv_df.variable.map(lambda x: importance[x])
        iv_df.to_excel(writer, 'feature_importance')

        writer.close()

        self.log.info("全部环节结束，请查看相关文件！")