# -*- coding: utf-8 -*-

"""
Split bins and compute woe.

Created on Fri Aug 16 16:45:21 2019

@author: wangxj <wangxj@getui.com>
"""

import numpy as np
import pandas as pd
import woe.feature_process as fp


def calc_chi2(arr):
    """
    计算卡方值
    :param arr: 2-dim array, 频数统计表
    :return: float, chi2 value
    """
    assert(arr.ndim == 2)

    # 行、列总频数
    R_N = arr.sum(axis=1)
    C_N = arr.sum(axis=0)
    N = arr.sum()

    # 期望频次
    E = np.ones(arr.shape) * C_N / N
    E = (E.T * R_N).T

    chi_sq = (arr - E)**2 / E
    # 期望频次为0时，作为分母无意义，不计入卡方值
    chi_sq[E == 0] = 0

    val = chi_sq.sum()
    return val


class TreeNode:
    def __init__(self, var_name=None, iv=0, split_point=None, left=None, right=None):
        """

        :param var_name:
        :param iv:
        :param split_point:
        :param left:
        :param right:
        """
        self.var_name = var_name
        self.iv = iv
        self.split_point = split_point
        self.left = left
        self.right = right


class DisInfoValue(object):
    '''
    A Class for the storage of discrete variables transformation information
    '''
    def __init__(self):
        self.var_name = None
        self.origin_value = []
        self.woe_before = []


class WOE:
    def __init__(self, max_bins=8, min_sample_rate=0.05, alpha=0.01, iv_additional=1e-4):
        """

        :param max_bins:
        :param min_smaple:
        :param alpha:
        :param iv_additional:
        """
        self.max_bins = max_bins
        self.alpha = alpha
        self.iv_additional = iv_additional
        self.min_sample_rate = min_sample_rate
        self.min_sample = None

    def fit(self, df, var, target, ft_type, method='bitree', global_pos_cnt=None, global_neg_cnt=None,
            particular_list=None, chi2_threshold=3.841):
        """

        :param df:
        :param var:
        :param target:
        :param ft_type:
        :param method:
        :param global_pos_cnt:
        :param global_neg_cnt:
        :param particular_list:
        :return:
        """
        if global_pos_cnt is None or global_neg_cnt is None:
            global_pos_cnt = df[target].sum()
            global_neg_cnt = df.shape[0] - global_pos_cnt

        if ft_type:
            s = 'process continuous variable: ' + var
        else:
            s = 'process discrete variable: ' + var
        print(s)

        self.min_sample = self.min_sample_rate * (global_neg_cnt + global_pos_cnt)
        vals = df[var].values
        targets = df[target].values

        split_list = self._split_bins(vals, targets, ft_type, global_pos_cnt, global_neg_cnt,
                                      particular_list, chi2_threshold, method)
        civ = self._proc_iv_bins(var, vals, targets, ft_type, split_list, global_pos_cnt, global_neg_cnt)
        return civ

    def _proc_iv_bins(self, var, vals, targets, ft_type, split_list, global_pos_cnt, global_neg_cnt):
        """
        Given the dataframe and split points list then return an InfoValue instance.
        Just for continuous variable
        :param df:
        :param split_list:
        :param global_pos_cnt:
        :param global_neg_cnt:
        :return:
        """
        if ft_type:  # continuous
            split_list.insert(0, float('-inf'))
            split_list.append(float('inf'))
            split_list = list(zip(split_list[:-1], split_list[1:]))
            format_split_list = ['(' + str(split_point[0]) + ',' + str(split_point[1]) + ']'
                                 for split_point in split_list[:-1]]
            format_split_list.append('(' + str(split_list[-1][0]) + ',' + str(split_list[-1][1]) + ')')
        else:
            format_split_list = split_list
        var_name = [var] * format_split_list.__len__()

        sub_total_sample_num = []
        positive_sample_num = []
        negative_sample_num = []
        sub_total_num_percentage = []
        positive_rate_in_sub_total = []
        negative_rate_in_sub_total = []
        woe_list = []
        iv_list = []
        for split_point in split_list:
            if ft_type:  # continuous
                tmp_targets = targets[(vals > split_point[0]) & (vals <= split_point[1])]
            else:  # discrete
                tmp_targets = targets[np.isin(vals, split_point)]

            if tmp_targets.__len__() == 0:
                continue

            iv_info = self._calculate_iv(tmp_targets, global_pos_cnt, global_neg_cnt)
            woei, ivi = iv_info['woei'], iv_info['ivi']
            woe_list.append(woei)
            iv_list.append(ivi)
            sub_total_sample_num.append(iv_info['sub_total_num'])
            positive_sample_num.append(iv_info['sub_pos_num'])
            negative_sample_num.append(iv_info['sub_neg_num'])
            sub_total_num_percentage.append(iv_info['sub_total_num_percentage'])
            positive_rate_in_sub_total.append(iv_info['sub_pos_rate'])
            negative_rate_in_sub_total.append(iv_info['sub_neg_rate'])

        iv = [sum(iv_list)]*format_split_list.__len__()

        columns = ['var_name', 'split_list', 'sub_total_sample_num', 'positive_sample_num',
                   'negative_sample_num', 'sub_total_num_percentage', 'positive_rate_in_sub_total',
                   'woe_list', 'iv_list', 'iv']
        civ_df = pd.DataFrame({'var_name': var_name,
                               'split_list': format_split_list,
                               'sub_total_sample_num': sub_total_sample_num,
                               'positive_sample_num': positive_sample_num,
                               'negative_sample_num': negative_sample_num,
                               'sub_total_num_percentage': sub_total_num_percentage,
                               'positive_rate_in_sub_total': positive_rate_in_sub_total,
                               'negative_rate_in_sub_total': negative_rate_in_sub_total,
                               'woe_list': woe_list,
                               'iv_list': iv_list,
                               'iv': iv}, columns=columns)

        return civ_df

    def _sort_vals(self, vals, targets, ft_type, particular_list, global_pos_cnt, global_neg_cnt):
        """

        :param vals:
        :param targets:
        :param ft_type:
        :return:
        """

        if particular_list is not None:
            tmp_index_flag = np.isin(vals, particular_list)
            if np.sum(tmp_index_flag) < self.min_sample:
                particular_list = None
            else:
                vals = vals[~tmp_index_flag]
                targets = targets[~tmp_index_flag]

        div = None
        if ft_type:  # continuous

            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            sorted_targets = targets[sorted_idx]
        else:
            div = DisInfoValue()
            uniq_vals = list(set(vals))
            rdict = dict()
            for uval in uniq_vals:
                tmp_targets = targets[(vals == uval)]
                iv_info = self._calculate_iv(tmp_targets, global_pos_cnt, global_neg_cnt)
                div.origin_value.append(uval)
                div.woe_before.append(iv_info['woei'])
                rdict[uval] = iv_info['woei']

            map_vals = np.vectorize(rdict.get)(vals)

            sorted_idx = np.argsort(map_vals)
            sorted_vals = map_vals[sorted_idx]
            sorted_targets = targets[sorted_idx]

        return sorted_vals, sorted_targets, div

    def _split_bins(self, vals, targets, ft_type, global_pos_cnt, global_neg_cnt, particular_list, chi2_threshold, method='bitree'):
        """

        :param df:
        :param var:
        :param target:
        :param ft_type:
        :param method:
        :param global_pos_cnt:
        :param global_neg_cnt:
        :param particular_list:
        :return:
        """
        if vals.__len__() == 0:
            return []
        if method == 'chi2':
            split_list = self._case_to_split_method(method)(vals, targets, ft_type, global_pos_cnt,
                                                            global_neg_cnt, particular_list, chi2_threshold)
        else:
            split_list = self._case_to_split_method(method)(vals, targets, ft_type,
                                                            global_pos_cnt, global_neg_cnt, particular_list)
        return split_list

    def _case_to_split_method(self, method):
        """

        :param method:
        :return:
        """
        fun_name = "_split_bins_" + method
        method = getattr(self, fun_name, self._split_bins_bitree)
        return method

    def _split_bins_bitree(self, vals, targets, ft_type, global_pos_cnt, global_neg_cnt, particular_list):
        """

        :param df:
        :param var:
        :param target:
        :param ft_type:
        :param global_pos_cnt:
        :param global_neg_cnt:
        :param particular_list:
        :return:
        """
        sorted_vals, sorted_targets, div = self._sort_vals(vals, targets, ft_type,
                                                           particular_list, global_pos_cnt, global_neg_cnt)
        iv_tree = self._bitree_data_split(sorted_vals, sorted_targets, global_pos_cnt, global_neg_cnt)
        iv_tree_split_list = self._ivtree_to_split(iv_tree)

        if ft_type:
            split_list = iv_tree_split_list
            if particular_list is not None:
                split_list = particular_list + iv_tree_split_list
            split_list.sort()
        if ft_type == 0:
            split_list = iv_tree_split_list
            split_list.sort()
            split_list_temp = list()
            split_list_temp.append(float("-inf"))
            split_list_temp.extend([i for i in split_list])
            split_list_temp.append(float("inf"))
            split_list_temp.sort()

            map_split_list = []
            for i in range(split_list_temp.__len__() - 1):
                temp = []
                for j in range(div.origin_value.__len__()):
                    if (div.woe_before[j] > split_list_temp[i]) & (div.woe_before[j] <= split_list_temp[i + 1]):
                        temp.append(div.origin_value[j])

                if temp.__len__() > 0:
                    map_split_list.append(temp)

            split_list = map_split_list
            if particular_list is not None:
                split_list.append(particular_list)

        return split_list

    def _split_bins_chi2(self, vals, targets, ft_type, global_pos_cnt, global_neg_cnt, particular_list, chi2_threshold):
        """

        :param df:
        :param ft:
        :param target:
        :param ft_type:
        :return:
        """
        sorted_vals, sorted_targets, div = self._sort_vals(vals, targets, ft_type,
                                                           particular_list, global_pos_cnt, global_neg_cnt)
        uniq_vals = np.unique(sorted_vals)
        freq_arr = np.zeros((uniq_vals.__len__(), 4))
        freq_arr[:, 0] = uniq_vals
        freq_arr[:, 2] = [np.sum(sorted_targets[sorted_vals == val]) for val in uniq_vals]
        freq_arr[:, 3] = [np.sum(sorted_vals == val) for val in uniq_vals]
        freq_arr[:, 1] = freq_arr[:, 3] - freq_arr[:, 2]
        split_list = [uniq_vals]

        neg_0_idx = np.where(freq_arr[:, 1] == 0)
        pos_0_idx = np.where(freq_arr[:, 2] == 0)
        idx_0 = list(set(list(pos_0_idx[0])).union(set(list(neg_0_idx[0]))))
        i = 0
        while i < freq_arr.shape[0]:
            if freq_arr[i, 0] == 0 or freq_arr[i, 1] == 0:
                pass  # 等会再好好想想这个地方要怎么搞 全正或全负和周围合并

        less_idx = np.where(freq_arr[:, 3] < self.min_sample)[0]
        merge_idx = []
        if less_idx.__len__() > 3:
            num_flag = less_idx[0]
            idx_flag = 0
            for i in range(1, less_idx.__len__()):
                if i - idx_flag != less_idx[i] - num_flag:
                    if i - idx_flag > 1:
                        merge_idx.append([num_flag, less_idx[i-1]])
                    num_flag = less_idx[i]
                    idx_flag = i
            if num_flag != less_idx[-1]:
                merge_idx.append([num_flag, less_idx[-1]])

        del_idx = []
        for seq in merge_idx:
            freq_arr[seq[0]+1, :] = np.sum(freq_arr[seq[0]+1:seq[1], :], axis=0)
            del_idx.extend(list(np.range(seq[0]+2, seq[1])))
        freq_arr = np.delete(freq_arr, del_idx, 0)

        chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
        for i in np.arange(freq_arr.shape[0] - 1):
            chi = calc_chi2(freq_arr[i:i + 2, 1:3])
            chi_table = np.append(chi_table, chi)
        less_idx = np.where(freq_arr[:, 3] < self.min_sample)[0]

        while True:
            if len(chi_table) == 0:
                break
            if len(chi_table) <= (self.max_bins - 1) and np.min(chi_table) >= chi2_threshold:
                break
            chi_min_index = np.argwhere(chi_table == np.min(chi_table))[0][0]  # 找出卡方值最小的位置索引
            freq_arr[chi_min_index] = freq_arr[chi_min_index] + freq_arr[chi_min_index + 1]
            freq_arr = np.delete(freq_arr, chi_min_index + 1, 0)
            cutoffs_idx = np.delete(cutoffs_idx, chi_min_index + 1, 0)

            if chi_min_index == freq_arr.shape[0] - 1:  # 最小值为最后两个区间的时候
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = calc_chi2(freq_arr[chi_min_index - 1:chi_min_index + 1])
                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0)
            else:
                # 计算合并后当前区间与前一个区间的卡方值并替换
                chi_table[chi_min_index - 1] = calc_chi2(freq_arr[chi_min_index - 1:chi_min_index + 1])
                # 计算合并后当前区间与后一个区间的卡方值并替换
                chi_table[chi_min_index] = calc_chi2(freq_arr[chi_min_index:chi_min_index + 2])

                # 删除替换前的卡方值
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)





        return split_list

    def _split_bins_ks(self, vals, targets, ft_type, global_pos_cnt, global_neg_cnt, particular_list):
        """

        :param df:
        :param ft:
        :param target:
        :param ft_type:
        :return:
        """

        sorted_vals, sorted_targets, div = self._sort_vals(vals, targets, ft_type,
                                                           particular_list, global_pos_cnt, global_neg_cnt)
        ks_split_cnt = 0
        # ks_tree = self._bestks_data_split(sorted_vals, sorted_targets, 0, sorted_targets.__len__(), ks_split_cnt)
        # ks_split_list = self._ivtree_to_split(ks_tree)
        ks_split_list = self._bestks_data_split(sorted_vals, sorted_targets, 0, sorted_targets.__len__())
        if ft_type:
            split_list = ks_split_list
            if particular_list is not None:
                split_list = particular_list + ks_split_list
            split_list.sort()
        if ft_type == 0:
            split_list = ks_split_list
            split_list.sort()
            split_list_temp = list()
            split_list_temp.append(float("-inf"))
            split_list_temp.extend([i for i in split_list])
            split_list_temp.append(float("inf"))
            split_list_temp.sort()

            map_split_list = []
            for i in range(split_list_temp.__len__() - 1):
                temp = []
                for j in range(div.origin_value.__len__()):
                    if (div.woe_before[j] > split_list_temp[i]) & (div.woe_before[j] <= split_list_temp[i + 1]):
                        temp.append(div.origin_value[j])

                if temp.__len__() > 0:
                    map_split_list.append(temp)

            split_list = map_split_list
            if particular_list is not None:
                split_list.append(particular_list)

        return split_list

    def _split_bins_frequent(self, df, ft, target, ft_type, global_pos_cnt, global_neg_cnt):
        """

        :param df:
        :param ft:
        :param target:
        :param ft_type:
        :return:
        """
        print("in frequent")
        split_list = []
        return split_list

    def _check_split_point(self, vals, targets, split_list):
        """
        Check whether the segmentation points cause some bin samples to be too small.
        If there is a bin sample size of less than 5% of the total sample size,
        then merge with the adjacent bin until more than 5%.
        Applies only to continuous values
        :param df: DataFrame, dataset
        :param var: string, variable name
        :param target: string, target column name
        :return: new_split , list, modified split list
        """
        new_split = []
        if split_list is not None and split_list.__len__() > 0:

            for point in split_list:
                if new_split.__len__() == 0:
                    tmp_index_flag = (vals <= point)
                else:
                    tmp_index_flag = (vals > new_split[-1]) & (vals <= point)
                tmp_targets = targets[tmp_index_flag]
                if (tmp_targets.__len__() < self.min_sample) or (np.unique(tmp_targets).__len__() == 1):
                    continue
                else:
                    new_split.append(point)

            # last split point
            if new_split.__len__() >= 1:
                tmp_index_flag = (vals > new_split[-1])
                tmp_targets = targets[tmp_index_flag]
                if (tmp_targets.__len__() < self.min_sample) or (np.unique(tmp_targets).__len__() == 1):
                    new_split.pop()
        else:
            pass

        return new_split

    def _calculate_iv(self, targets, global_pos_cnt, global_neg_cnt):
        """
        Calculate iv and woe
        :param df: source dataframe
        :param target: string, target name
        :param global_pos_cnt: int
        :param global_neg_cnt: int
        :return: iv_info , dict
        """
        sub_total_cnt = targets.__len__()
        sub_pos_cnt = np.sum(targets)
        sub_neg_cnt = len(targets) - sub_pos_cnt
        sub_pos_rate = (sub_pos_cnt + self.iv_additional) * 1.0 / global_pos_cnt
        sub_neg_rate = (sub_neg_cnt + self.iv_additional) * 1.0 / global_neg_cnt

        iv_info = dict()
        iv_info['woei'] = np.log(sub_pos_rate / sub_neg_rate)
        iv_info['ivi'] = (sub_pos_rate - sub_neg_rate) * np.log(sub_pos_rate / sub_neg_rate)
        iv_info['sub_total_num'] = sub_total_cnt
        iv_info['sub_total_num_percentage'] = sub_total_cnt * 1.0 / (global_pos_cnt + global_neg_cnt)
        iv_info['sub_pos_num'] = sub_pos_cnt
        iv_info['sub_neg_num'] = sub_neg_cnt
        iv_info['sub_pos_rate'] = sub_pos_cnt / sub_total_cnt
        iv_info['sub_neg_rate'] = sub_neg_cnt / sub_total_cnt

        return iv_info

    def _calculate_ks(self, vals, targets, start_idx, end_idx):
        """

        :param targets: 1-d array
        :param start_idx: int, include
        :param end_idx: int, not include
        :return: float, val - max ks
        """
        #s_idx = int(start_idx + self.min_sample)
        #e_idx = int(end_idx - self.min_sample)
        s_idx = start_idx
        e_idx = end_idx
        if s_idx >= e_idx:
            return None
        calc_vals = vals[s_idx:e_idx]
        calc_targets = targets[s_idx:e_idx]
        pos_cnt = np.sum(calc_targets)
        neg_cnt = calc_targets.__len__() - pos_cnt
        uniq_vals = np.sort(np.unique(calc_vals))
        ks_list = []
        for uval in uniq_vals:
            tmp_targets = calc_targets[calc_vals >= uval]
            tmp_pos = np.sum(tmp_targets)
            tmp_neg = tmp_targets.__len__() - tmp_pos
            tmp_tpr = tmp_pos / pos_cnt
            tmp_fpr = tmp_neg / neg_cnt
            ks_list.append(np.abs(tmp_tpr - tmp_fpr))

        ks_max_idx = np.argmax(ks_list)
        # ks = ks_list[ks_max_idx]
        ks_val = uniq_vals[ks_max_idx]
        return ks_val

    def _bestks_data_split(self, vals, targets, start_idx, end_idx):
        """

        :param vals:
        :param targets:
        :param start_idx:
        :param end_idx:
        :param split_cnt:
        :return:
        """
        flag = 'left'
        split_list = []
        s_idx_list = []
        e_idx_list = []
        parent_idx = -1
        while (split_list.__len__() < self.max_bins) and (parent_idx < split_list.__len__()):
            if parent_idx == -1:
                s_idx = start_idx
                e_idx = end_idx
                parent_idx += 1
            else:
                split_val = split_list[parent_idx]
                if flag == 'left':
                    split_s_idx = s_idx_list[parent_idx]
                    ks_val_idx_l = np.where(vals == split_val)[0][0]
                    s_idx = split_s_idx
                    e_idx = ks_val_idx_l
                    flag = 'right'
                else:
                    split_e_idx = e_idx_list[parent_idx]
                    ks_val_idx_r = np.where(vals == split_val)[0][-1]
                    s_idx = ks_val_idx_r + 1
                    e_idx = split_e_idx
                    flag = 'left'
                    parent_idx += 1

            if s_idx + self.min_sample < e_idx - self.min_sample:
                split_val = self._calculate_ks(vals, targets, s_idx, e_idx)
                if split_val is not None:
                    split_list.append(split_val)
                    s_idx_list.append(s_idx)
                    e_idx_list.append(e_idx)
                    if split_list.__len__() >= self.max_bins:
                        break

        return split_list


    def _bestks_data_split_recur(self, vals, targets, start_idx, end_idx, split_cnt):
        """

        :param vals: 1-d array, sorted_vals
        :param targets: 1-d array, sorted_targes
        :param global_pos_cnt: int
        :param global_neg_cnt: int
        :return:
        """
        if split_cnt >= self.max_bins:
            return None

        split_val = self._calculate_ks(vals, targets, start_idx, end_idx)

        if split_val is None:
            return None
        split_cnt += 1

        ks_val_idx_l = np.where(vals == split_val)[0][0]
        ks_val_idx_r = np.where(vals == split_val)[0][-1]

        left = None
        right = None
        if ks_val_idx_l - self.min_sample > start_idx + self.min_sample:
            left = self._bestks_data_split(vals, targets, start_idx, ks_val_idx_l, split_cnt)
        if end_idx - self.min_sample > ks_val_idx_r + 1 + self.min_sample:
            right = self._bestks_data_split(vals, targets, ks_val_idx_r+1, end_idx, split_cnt)

        return TreeNode(split_point=[split_val], left=left, right=right)

    def _bitree_data_split(self, vals, targets, global_pos_cnt, global_neg_cnt):
        """

        :param df:
        :param var:
        :param target:
        :param global_pos_cnt:
        :param global_neg_cnt:
        :param particular_list:
        :return:
        """
        cur_iv_info = self._calculate_iv(targets, global_pos_cnt, global_neg_cnt)
        woe_cur, iv_cur = cur_iv_info['woei'], cur_iv_info['ivi']

        nval = np.unique(vals)
        nval.sort()
        if nval.__len__() <= 8:
            split = list(nval)
            split = self._check_split_point(vals, targets, split)
            split.sort()
            if split.__len__() == 0:
                split = None
            return TreeNode(split_point=split, iv=iv_cur)

        percent_value_idx = list(np.unique(np.round(np.linspace(0., 1, 101, 1) * vals.__len__())).astype(int))
        percent_value = np.unique(vals[percent_value_idx[:-1]])

        if percent_value.__len__() <= 2:
            return TreeNode(iv=iv_cur)

        best_split_iv = 0
        best_split_point = None
        best_split_index_l = None
        best_split_index_r = None
        for point in percent_value:
            tmp_index_flag_l = (vals <= point)
            tmp_index_flag_r = (vals > point)
            tmp_targets_l = targets[tmp_index_flag_l]
            tmp_targets_r = targets[tmp_index_flag_r]

            if (np.unique(tmp_targets_l).__len__() == 1) or (np.unique(tmp_targets_r).__len__() == 1) \
                or (tmp_targets_l.__len__() < self.min_sample) or (tmp_targets_r.__len__() < self.min_sample):
                continue

            iv_info_l = self._calculate_iv(tmp_targets_l, global_pos_cnt, global_neg_cnt)
            iv_info_r = self._calculate_iv(tmp_targets_r, global_pos_cnt, global_neg_cnt)

            iv = iv_info_l['ivi'] + iv_info_r['ivi']

            if iv > best_split_iv:
                best_split_iv = iv
                best_split_point = [point]
                best_split_index_l = tmp_index_flag_l
                best_split_index_r = tmp_index_flag_r

        if best_split_iv > iv_cur * (1+self.alpha):
            left = self._bitree_data_split(vals[best_split_index_l], targets[best_split_index_l],
                                           global_pos_cnt, global_neg_cnt)
            right = self._bitree_data_split(vals[best_split_index_r], targets[best_split_index_r],
                                            global_pos_cnt, global_neg_cnt)

            return TreeNode(split_point=best_split_point, iv=iv_cur, left=left, right=right)
        else:
            return TreeNode(iv=iv_cur)

    def _ivtree_to_split(self, iv_tree):
        """

        :param iv_tree:
        :return:
        """
        split_list = []
        if iv_tree.split_point is not None:
            split_list.extend(iv_tree.split_point)

        if iv_tree.left is not None:
            split_list.extend(self._ivtree_to_split(iv_tree.left))

        if iv_tree.right is not None:
            split_list.extend(self._ivtree_to_split(iv_tree.right))

        return split_list


def calc_iv_bins(df_data, target, cols_flt=[], cols_dis=[], method='bitree', global_pos_cnt=None, globla_neg_cnt=None,
                 particular_flt_list=None, particular_dis_list=None, max_bins=8, min_sample_rate=0.05):
    """
    利用iv值分箱，并使用woe标记每个分箱的指示值
    :param df_src: dataframe，原始数据
    :param label: string, 原始数据的标记列
    :param cols: list of string, 需要计算iv的列
    :return: dataframe
    """

    if global_pos_cnt is None or globla_neg_cnt is None:
        n = df_data.shape[0]
        global_pos_cnt = sum(df_data[target])
        global_neg_cnt = n - global_pos_cnt

    civ_list = []
    woe = WOE(max_bins=max_bins, min_sample_rate=min_sample_rate)
    for ft in cols_flt:
        civ = woe.fit(df_data, ft, target, 1, method=method,
                      global_pos_cnt=global_pos_cnt, global_neg_cnt=globla_neg_cnt,
                      particular_list=particular_flt_list)
        civ_list.append(civ)

    for ft in cols_dis:
        civ = woe.fit(df_data, ft, target, 0, method=method,
                      global_pos_cnt=global_pos_cnt, global_neg_cnt=global_neg_cnt,
                      particular_list=particular_dis_list)
        civ_list.append(civ)

    civ_df = pd.concat(civ_list)
    return civ_df

if __name__ == '__main__':
    """
    import numpy as np
    import pandas as pd
    from woe_bins import *
    """
    df_src = pd.read_csv('/Users/wangxj/Dropbox/getui/yingxiao/model_kaifa/kh/data/sy_flag.txt', sep='|')
    df_src = df_src[df_src.gid.str.find('ANDROID') > -1]
    df_src_1 = df_src[df_src.label == 1]
    df_src_0 = df_src[df_src.label == 0]
    df_src_sample_1 = df_src_1.sample(n=100000, replace=True, random_state=7, axis=0)
    df_src = pd.concat([df_src_0, df_src_sample_1])
    df = df_src[['ft_app_act_30d_credit_card_times',
                 'ft_dev_battery_charge_30d_weekday_allday_hours',
                 'ft_tag_p2p','ft_tag_gender','ft_tag_age','label','ft_app_ins_current_invest_cnt']].copy()


    #del df_src
    df.fillna(-1, inplace=True)
    # var = 'ft_dev_battery_charge_30d_weekday_allday_hours'
    var = 'ft_app_act_30d_credit_card_times'
    my_woe = WOE()
    target = 'label'
    df['target'] = df['label']
    woe_org = fp.proc_woe_continuous(df, var, 100000, 103540,
                                     0.05 * df.shape[0], alpha=0.01)
    civ = my_woe.fit(df, var, target, 1, method='ks')
    print('ok')
    """
    tfidf = pd.read_csv('/Users/wangxj/Dropbox/getui/yingxiao/model_kaifa/explorer/tf-idf/sy_tfidf_sy.txt', sep='|')
    

    # civ = my_woe.fit(df, 'ft_tag_age', target, 0, particular_list=[-1])
    merge = pd.read_csv('/Users/wangxj/Dropbox/getui/yingxiao/model_kaifa/explorer/tf-idf/sy_358_tfidf.txt', sep='|', )
    merge.fillna(-1, inplace=True)
    id_cols = ['number', 'id', 'id_type', 'create_date', 'flag', 'target',
               'gid', 'label_jd', 'label', 'label_xyf', 'label_ac', 'label_lhp',
               'pkg', 'sum_times', 'tagid1', 'ft_dev_phone_model','ft_dev_phone_brand'
               ]
    str_cols = [
        'ft_tag_education'
        , 'ft_gz_grey_list'
        , 'ft_dev_root_blacklist'
        , 'ft_gezhen_multi_loan_level'
        , 'ft_dev_trust'
        , 'ft_lbs_geo7_unusual'
        , 'ft_social_blackfriends_group'
        , 'ft_social_blackfriends'
        , 'ft_gz_black_list'
        , 'ft_tag_province'
        , 'ft_tag_city'
        , 'ft_tag_age'
        , 'ft_tag_gender']
    merge[str_cols] = merge[str_cols].astype(str)

    cols_flt = set(merge.columns.tolist()).difference(set(str_cols)).difference(set(id_cols))
    """
    #civ = calc_iv_bins(merge, 'label', cols_flt, str_cols, particular_flt_list=[-1], particular_dis_list=['-1', '-1.0'])
    #civ.to_excel('/Users/wangxj/Dropbox/getui/yingxiao/model_kaifa/explorer/tf-idf/sy_-1_ft_info.xlsx')
    """
    civ = calc_iv_bins(merge, 'label', cols_flt, str_cols)
    civ.to_excel('/Users/wangxj/Dropbox/getui/yingxiao/model_kaifa/explorer/tf-idf/sy_ft_info.xlsx')
    """

