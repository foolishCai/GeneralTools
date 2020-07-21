#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 14:36
# @Author  : cai
# @contact : yuwei.chen@yunzhenxin.com
# @File    : ScoreCard.py
# @Note    :

import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from BaseUtils import log

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
        from ModelUtil.draw_util import get_curve, get_feature_importance, get_cm

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
        report_df = pd.concat([pd.DataFrame(data={"precision": [precision_score(y_true=self.df_train_woe[self.label], y_pred=[int(i > thresh) for i in list(self.train_y_pred)])],
                                      "recall": [recall_score(y_true=self.df_train_woe[self.label], y_pred=[int(i > thresh) for i in list(self.train_y_pred)])],
                                      "accuracy": [accuracy_score(y_true=self.df_train_woe[self.label], y_pred=[int(i > thresh) for i in list(self.train_y_pred)])],
                                      "f1_score": [f1_score(y_true=self.df_train_woe[self.label], y_pred=[int(i > thresh) for i in list(self.train_y_pred)])]
                                      }, index=["train"]), report_df])

        if self.df_test_woe.any().any():
            x_data = self.df_test_woe[self.final_features]
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
        get_feature_importance(feature=list(params["var"])[1:], importance=list(params["coef"])[1:], filename="{}_feature_importance.png".format(self.filename))
        print(report_df)


    def _check_row_na(self, data):
        rows_null_percent = data.isnull().sum(axis=1) / data.shape[1]
        to_drop_rows = rows_null_percent[rows_null_percent > 0]
        to_drop_rows = to_drop_rows.to_dict()
        return to_drop_rows

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

        self.log.info(">>>>>>>>>>>>> 检查空值")
        to_drop_rows = self._check_row_na(self.df_train_woe)
        if len(to_drop_rows) > 0:
            self.log.info("训练集中有映射空值，请检查！")

        if self.df_test_woe.any().any():
            to_drop_rows = self._check_row_na(self.df_test_woe)
            if len(to_drop_rows) > 0:
                self.log.info(
                    "该数据集集有{}个样本仍含有缺失值，暂删除，占比{}".format(len(to_drop_rows), round(len(to_drop_rows) / self.df_test_woe.shape[0], 4)))
                self.df_test_woe = self.df_test_woe[self.df_test_woe.index.isin(to_drop_rows.keys())]

        if self.df_ott_woe.any().any():
            to_drop_rows = self._check_row_na(self.df_ott_woe)
            if len(to_drop_rows) > 0:
                self.log.info(
                    "该数据集集有{}个样本仍含有缺失值，暂删除，占比{}".format(len(to_drop_rows),
                                                        round(len(to_drop_rows) / self.df_ott_woe.shape[0], 4)))
                self.df_ott_woe = self.df_ott_woe[self.df_ott_woe.index.isin(to_drop_rows.keys())]

        self.log.info(">>>>>>>>>>>>> 检查inf")
        self.check_woe_value()

        self.log.info(">>>>>>>>>>>>> 相关性检验, max_corr={}".format(self.max_corr))
        drop_set = self.check_corr()
        if drop_set:
            self.log.info("已过滤{}个特征".format(len(drop_set)))
            self.del_col.extend(list(drop_set))
            self.final_features = [i for i in list(self.iv_df.feature) if i not in self.del_col]
        self.log.info("*"*30)

        self.log.info(">>>>>>>>>>>>> VIF检验, max_vif={}".format(self.max_vif))
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
        self.log.info("变量重要性，请查看{}_feature_importance.png".format(self.filename))

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
