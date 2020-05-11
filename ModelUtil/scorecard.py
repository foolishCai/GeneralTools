# -*- coding: utf-8 -*-

import re
from itertools import combinations
import pandas as pd
import numpy as np
# import modin.pandas as pd
from math import isnan
import multiprocessing as ps
from pandas import pivot_table


def group_by_df(data, flag_name, factor_name, bad_name, good_name, na_trigger):
    if len(data) == 0:
        return pd.DataFrame()
    data = data[flag_name].groupby([data[factor_name], data[flag_name]]).count()
    data = data.unstack()
    data = data.reset_index()
    data = data.fillna(0)
    if len(data.columns) == 3:
        data.columns = [factor_name, good_name, bad_name]
        if not na_trigger:
            data[factor_name] = data[factor_name].astype(float)
        data = data.sort_values(by=[factor_name], ascending=True)
        data[factor_name] = data[factor_name].astype(str)
        data = data.reset_index(drop=True)
        return data
    else:
        return pd.DataFrame()


def verify_df_two(date_df, flag_name):
    date_df = date_df.drop(date_df[date_df[flag_name].isnull()].index)
    check = date_df[date_df[flag_name] > 1]
    if check.shape[0] != 0:
        print('Error: there exits the number bigger than one in the data')
        date_df = pd.DataFrame()
        return date_df
    elif date_df.shape[0] != 0:
        date_df[flag_name] = date_df[flag_name].astype(int)
        return date_df
    else:
        print('Error: the data is wrong')
        date_df = pd.DataFrame()
        return date_df


def getAllindex(tar_list, item):
    return list([a for a in range(0, len(tar_list)) if tar_list[a] == item])


# the split function part
# since there are the restriction to the position of the best ks ,thus we need to add some transformation to the start
# and end knot in the current function
def best_KS_knot_calculator(data, good_name, bad_name, start_knot, end_knot, rate):
    # this part may seems to be redundant, you can replace it by passing the total_len in the input part of the function
    total_len = sum(data[good_name]) + sum(data[bad_name])
    temp_df = data.loc[start_knot:end_knot]
    # the temp len here represents the length of the unique values in the raw data, not only the length of the temp_df
    temp_len = sum(temp_df[good_name]) + sum(temp_df[bad_name])
    # since we'd want to make sure the number of elements in each tiny bin should always greater than 5% of the raw
    # data's length, we need to add some restrictions to the start_knot and the end_knot
    start_add_num = sum(np.cumsum(temp_df[good_name] + temp_df[bad_name]) < rate * total_len)
    # print start_add_num
    end_add_num = sum(np.cumsum(temp_df[good_name] + temp_df[bad_name]) <= temp_len - rate * total_len)
    # print end_add_num
    processed_start_knot = start_knot + start_add_num
    # print processed_start_knot
    processed_end_knot = start_knot + end_add_num - 1
    # print processed_end_knot
    if processed_end_knot >= processed_start_knot:
        if sum(temp_df[bad_name]) != 0 and sum(temp_df[good_name]) != 0:
            default_CDF = np.cumsum(temp_df[bad_name]) / temp_df[bad_name].sum()
            # print default_CDF
            undefault_CDF = np.cumsum(temp_df[good_name]) / temp_df[good_name].sum()
            # print undefault_CDF
            ks_value = max(abs(default_CDF - undefault_CDF).loc[processed_start_knot:processed_end_knot])
            # ks_value
            # the index find here is not the final result, we should find the data's position in the outer data set
            index = getAllindex(list(abs(default_CDF - undefault_CDF)), ks_value)
            # print index
            return temp_df.index[max(index)]#取KS最大点做切分
        else:
            return None
    else:
        return None

#递归函数，迭代切分
def best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot, end_knot, current_time):
    # define the base case
    # here, we should first find out the total length of the raw data. Since the elements in the input data
    # represent the count of unique item in the raw data, thus we need to do some transformation to it to find
    # out the length that we need
    # total_len = sum(data[good_name]) + sum(data[bad_name])
    # print(start_knot)
    # print(end_knot)
    temp_df = data.loc[start_knot:end_knot]
    temp_len = sum(temp_df[good_name]) + sum(temp_df[bad_name])
    # due to the restriction to the number of elements in the tiny bin
    #限制箱内最小样本数
    if temp_len < rate * total_len * 2 or current_time >= max_times:
        return []
    new_knot = best_KS_knot_calculator(data, good_name, bad_name, start_knot, end_knot, rate)
    if new_knot is not None:
        # upper_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot,
        #                                     new_knot - 1, current_time + 1)
        upper_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot,
                                            new_knot, current_time + 1)
        lower_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, new_knot + 1,
                                            end_knot, current_time + 1)
    else:
        upper_result = []
        lower_result = []
    return upper_result + [new_knot] + lower_result


def new_ks_auto(data_df, total_rec, piece, rate, good_name, bad_name):
    # call the helper function to finish the following recursion part
    temp_result_list = best_ks_knots_helper(data_df, total_rec, piece, good_name, bad_name, rate, 0, data_df.shape[0],
                                            0)
    # since we are gonna to use reconstruct the whole function, thus I choose not to add the two marginal points to the
    # result list
    # temp_result_list = temp_result_list + [0, len(data_df)-1]
    temp_result_list=[x for x in temp_result_list if x is not None]#剔除none切分点
    temp_result_list.sort()
    return temp_result_list


# the merge function part
# first, we need to define a IV calculator function. The input should be the knots list and the data with unique value's
# black flag count. The result returned by the function will be the IV list of these tiny bins
def  IV_calculator(data_df, good_name, bad_name, knots_list):
    # to improve the efficiency of the calculation, I first split the df into a bunch of smaller data frames and put
    # them into a list. Then I Use the map function to do some transformation to calculate the IV value for each small bin
    temp_df_list = []
    for i in range(1, len(knots_list)):
        if i == 1:
            # the data range here we chose to use such format (start, end], since the calculation of CDF is left
            # continuous, thus we need to include the right margin and the left margin should be not included
            # attention: the pd.Series[start:end] is different from pd.DataFrame.loc[start:end]. The previous will not
            # include the end point but the later one will include
            temp_df_list.append(data_df.loc[knots_list[i - 1]:knots_list[i]])
        else:
            temp_df_list.append(data_df.loc[knots_list[i - 1] + 1:knots_list[i]])
    total_good = sum(data_df[good_name])
    total_bad = sum(data_df[bad_name])
    good_percent_series = pd.Series(list([float(sum(x[good_name])) / total_good for x in temp_df_list]))
    bad_percent_series = pd.Series(list([float(sum(x[bad_name])) / total_bad for x in temp_df_list]))
    # the woe_list here is used for debugging
    woe_list = list(np.log(good_percent_series / bad_percent_series))
    # here, since we want to make sure the woe of the result bins is monotonic, thus we add a justification statement
    # here, if it is not monotonic, then it will be discarded and the return will be None
    if sorted(woe_list) != woe_list and sorted(woe_list, reverse=True) != woe_list:#判断是否单调
        return None
    IV_series = (good_percent_series - bad_percent_series) * np.log(good_percent_series / bad_percent_series)
    if np.inf in list(IV_series) or -np.inf in list(IV_series):
        return None
    else:
        return sum(IV_series)


# combination_helper function
def combine_helper(data_df, good_name, bad_name, piece_num, cut_off_list):
    knots_list = list(combinations(cut_off_list, piece_num - 1))#切分点组合
    # here we do some transformation to the knots list, add the start knot and end knot to all the elements in the list
    # knots_list = map(lambda x: sorted(tuple(set(x + (0, len(data_df) - 1)))), knots_list)
    knots_list = [sorted(x + (0, len(data_df) - 1)) for x in knots_list]
    IV_for_bins = [IV_calculator(data_df, good_name, bad_name, x) for x in knots_list]#计算iv，并且判断woe是否单调，只输出单调的切割方式
    filtered_IV = list([x for x in IV_for_bins if x is not None])
    if len(filtered_IV) == 0:
        print(('There are no suitable division for the data set with ' + str(piece_num) + ' pieces'))
        return None
    else:
        if len(getAllindex(IV_for_bins, max(filtered_IV))) > 0:#取iv最大的组合
            target_index = getAllindex(IV_for_bins, max(filtered_IV))[0]
            return knots_list[target_index]
        else:
            return None


# the cut_off_list here should not contain the start knot 0 and end knot len(data_df)-1, since these knots will be added
# in the later process in function's map part
def combine_tiny_bins(data_df, good_name, bad_name, max_piece_num, cut_off_list):
    return_piece_num = min(max_piece_num, len(cut_off_list) + 1)
    if return_piece_num == 1:
        return cut_off_list
    for current_piece_num in sorted(list(range(2, return_piece_num + 1)), reverse=True):
        result_knots_list = combine_helper(data_df, good_name, bad_name, current_piece_num, cut_off_list)
        # here we obey the rule that, the function will return the maximum number of bins with maximum IV value, thus if
        # there is no suitable cut_off_list for the current piece, the number of bins will be minus one
        if result_knots_list is not None:
            return result_knots_list
    print("sry, there isn't any suitable division for this data set with the column that you give :(")
    return [0, len(data_df) - 1]


# this function is used for return the import statistical indicators for the binning result
def important_indicator_calculator(data_df, good_name, bad_name, factor_name, knots_list, na_df):
    if len(na_df) != 0:
        total_good = sum(data_df[good_name]) + sum(na_df[good_name])
        total_bad = sum(data_df[bad_name]) + sum(na_df[bad_name])
        default_CDF = np.cumsum(na_df[bad_name]) / total_bad
        undefault_CDF = np.cumsum(na_df[good_name]) / total_good
        ks_list = list(abs(default_CDF - undefault_CDF))
        na_df['total'] = na_df[good_name] + na_df[bad_name]
        na_df['good_percnet_series'] = na_df[good_name] / total_good
        na_df['bad_percnet_series'] = na_df[bad_name] / total_bad
        na_df['woe_list'] = np.log(na_df['good_percnet_series'] / na_df['bad_percnet_series'])
        na_df['IV_list'] = (na_df['good_percnet_series'] - na_df['bad_percnet_series']) * na_df['woe_list']
        na_df['total'] = na_df[good_name] + na_df[bad_name]
        na_df['bad_rate'] = na_df[bad_name] / na_df['total']
        na_indicator = pd.DataFrame({'Bin': na_df[factor_name].tolist(), 'bad': na_df[bad_name].tolist(),
                                     'good': na_df[good_name].tolist(),
                                     'KS': ks_list, 'WOE': na_df['woe_list'].tolist(),
                                     'IV': na_df['IV_list'].tolist(),
                                     'total_count': na_df['total'].tolist(),
                                     'bad_rate': na_df['bad_rate'].tolist()})
    else:
        total_good = sum(data_df[good_name])
        total_bad = sum(data_df[bad_name])
        na_indicator = pd.DataFrame()
    default_CDF = np.cumsum(data_df[bad_name]) / total_bad
    undefault_CDF = np.cumsum(data_df[good_name]) / total_good
    ks_list = list(abs(default_CDF - undefault_CDF).loc[knots_list[:len(knots_list) - 1]])
    temp_df_list = []
    bin_list = []
    for i in range(1, len(knots_list)):
        if i == 1:
            temp_df_list.append(data_df.loc[knots_list[i - 1]:knots_list[i]])
            bin_list.append('(-inf, ' + data_df[factor_name][knots_list[i]] + ']')
        else:
            temp_df_list.append(data_df.loc[knots_list[i - 1] + 1:knots_list[i]])
            if i == len(knots_list) - 1:
                bin_list.append('(' + data_df[factor_name][knots_list[i - 1]] + ', inf)')
            else:
                bin_list.append(
                    '(' + data_df[factor_name][knots_list[i - 1]] + ', ' + str(
                        data_df[factor_name][knots_list[i]]) + ']')
    good_percent_series = pd.Series(list([float(sum(x[good_name])) / total_good for x in temp_df_list]))
    bad_percent_series = pd.Series(list([float(sum(x[bad_name])) / total_bad for x in temp_df_list]))
    woe_list = list(np.log(good_percent_series / bad_percent_series))
    IV_list = list((good_percent_series - bad_percent_series) * np.log(good_percent_series / bad_percent_series))
    total_list = list([sum(x[good_name]) + sum(x[bad_name]) for x in temp_df_list])
    bad_num = list([sum(x[bad_name]) for x in temp_df_list])
    good_num = list([sum(x[good_name]) for x in temp_df_list])
    bad_rate_list = list([float(sum(x[bad_name])) / (sum(x[good_name]) + sum(x[bad_name])) for x in temp_df_list])
    non_na_indicator = pd.DataFrame({'Bin': bin_list, 'bad': bad_num, 'good': good_num,
                                     'KS': ks_list, 'WOE': woe_list, 'IV': IV_list,
                                     'total_count': total_list, 'bad_rate': bad_rate_list})
    result_indicator = pd.concat([non_na_indicator, na_indicator], axis=0).reset_index(drop=True)
    return result_indicator


# the most significant difference between the two function is the all_information part
# let's redefine this function
def all_information(data_df, na_df, total_rec, piece, rate, factor_name, bad_name, good_name):
    # to get the final result of the binning part, we need two process: split and merge
    split_knots = new_ks_auto(data_df, total_rec, piece, rate, good_name, bad_name) #分箱
    best_knots = combine_tiny_bins(data_df, good_name, bad_name, piece, split_knots) #合并分箱
    return important_indicator_calculator(data_df, good_name, bad_name, factor_name, best_knots, na_df)


# here is the final_outer function
def Best_KS_Bin(flag_name, factor_name, data=pd.DataFrame(), bad_name='bad', good_name='good',
                piece=5, rate=0.01, min_bin_size=50, not_in_list=[-99998, '-99998', '-99998.0', -99998.0]):
    # print(time.localtime(time.time()))
    # in order to avoid revising the raw data, we choose to copy the current data and contain only the factor and flag
    # columns
    if len(data) == 0:
        print('Error: there is no data')
        return pd.DataFrame()
    work_data = data.loc[data.index, [factor_name, flag_name]]
    # since the data without flag is meaningless thus, we can use the helper function to help us filer these data
    work_data = verify_df_two(work_data, flag_name)#检查label数据，并转成int
    if len(work_data) == 0:
        return pd.DataFrame
    # after that, we want to separate the current df into two parts, NA and non-NA
    # the very first thing here should be transforming the type of value in factor_name column into str type
    work_data[factor_name] = work_data[factor_name].astype(str)#将特征转换成str，以便区分none值
    # since there will be the None and nan be transformed into the str type, thus we need to add some default value into
    # the not_in_list, the set here may be redundant
    not_in_list = not_in_list + ['None', 'nan']
    na_df = work_data.loc[work_data[factor_name].apply(lambda x: x in not_in_list)]
    non_na_df = work_data.loc[work_data[factor_name].apply(lambda x: x not in not_in_list)]
    na_df = group_by_df(na_df, flag_name, factor_name, bad_name, good_name, True)
    non_na_df = group_by_df(non_na_df, flag_name, factor_name, bad_name, good_name, False)
    #     return non_na_df,na_df
    # print factor_name
    if len(non_na_df) == 0:
        print('sry, there are no data available for separate process :(')
        return pd.DataFrame()
    total_rec = work_data.shape[0]
    min_bin_size_rate = min_bin_size / float(total_rec)
    min_bin_size_r = min(rate, min_bin_size_rate)
    #区分nadf和nonnadf，nadf不进入分箱
    result = all_information(non_na_df, na_df, total_rec, piece, min_bin_size_r, factor_name, bad_name, good_name)
    return result


# Calculate variable WOE and replace variable value with W
# res=Best_KS_Bin('label','all_check_12month',data)


####map woe
# Create a dataframe with the variable values replaced by woe
from math import isnan



def var_woe(x, bin_dic, not_in_list):
    val = None
    if pd.isnull(x) or isnan(x) or x in not_in_list:
        for woe in bin_dic:
            if bin_dic[woe] in ['nan', 'NaN']:
                val = woe
    elif x in not_in_list:
        for woe in bin_dic:
            end_points = bin_dic[woe]
            len_points = len(end_points)
            if len_points == 1 and end_points[0] != 'nan':
                if x == float(end_points[0]):
                    val = woe
    else:
        for woe in bin_dic:
            end_points = bin_dic[woe]
            if len(end_points) == 2 and end_points[0] != 'nan':
                if end_points[0] == '-inf':
                    if x <= float(end_points[1]):
                        val = woe
                elif end_points[1] == 'inf':
                    if x > float(end_points[0]):
                        val = woe
                elif (x > float(end_points[0])) & (x <= float(end_points[1])):
                    val = woe
            elif len(end_points) > 2:
                if end_points[0] == '-inf' and end_points[-1] == 'inf':
                    val = woe
                elif end_points[0] == '-inf':
                    if x <= float(end_points[-1]):
                        val = woe
                elif end_points[-1] == 'inf':
                    if x > float(end_points[0]):
                        val = woe
                elif (x > float(end_points[0])) & (x <= float(end_points[-1])):
                    val = woe
    return val


def agg_func(x):
    str_value = ""
    for value in x:
        str_value += value
        str_value += ", "
    return str_value


def df_woe1(filepath, output_file, flag_name, data_test, data_train=pd.DataFrame(), not_var_list=[], not_in_list=[],
           target_var_list=[],
           flag_var_list=['custr_nbr'],Bin_rate=0.05,Bin_max_piece=5):
    """
    :param flag_name: bad flag
    :param data_test/data_train: target data
    :param data_train: input data to calculate the bin
    :param piece: number of bins in final result
    :param rate: number of minimum percent in one bin
    :param not_in_list: special values should be treated
    :param not_var_list: special variables should be treated
    :param Bin_rate: min rate of smaple in each Bin
    :return:
    """
    # 检验数据
    output_filename = filepath + "/" + output_file
    writer = pd.ExcelWriter("{}.xlsx".format(output_filename))
    flag_var_list.append(flag_name)
    if data_test.any().any():
        data_woe_test = data_test[flag_var_list]
    data_woe_train = data_train[flag_var_list]
    data_bin = pd.DataFrame()
    if len(data_train) == 0:
        print('Original input data is empty')
        return pd.DataFrame()
    var_list = data_train.columns
    if target_var_list:
        target_var_list = target_var_list
    else:
        not_var_list.extend(flag_var_list)
        not_var_list.append(flag_name)
        not_var_list.append('time_stamp')
        not_in_list.append(['None', 'nan'])
        target_var_list = list(set(var_list) - set(not_var_list))
    iv_list = []
    ks_list = []
    if len(target_var_list) == 0:
        print('No variable available for analysis')
        return pd.DataFrame()

    # 循环分箱
    iter = 0
    i = 0
    for var in target_var_list:
        # print(0)
        var_stat = Best_KS_Bin(flag_name, var, data_train,rate=Bin_rate,piece=Bin_max_piece) #bestKS分箱
        # print(1)
        var_stat['var'] = var
        var_stat.to_excel(writer, 'bin', startrow=i)
        len_df = var_stat.shape[0] + 1
        i += len_df + 2
        print(var_stat)
        if len(var_stat) > 0:
            var_stat_new = var_stat['Bin'].groupby(var_stat['WOE']).apply(lambda x: agg_func(x)).reset_index()
            bin_dic = dict(list(zip(var_stat_new['WOE'], var_stat_new['Bin'])))
            for woe in bin_dic:
                match_case = re.compile("\(|\)|\[|\]")
                end_points = match_case.sub('', bin_dic[woe]).split(', ')
                bin_dic[woe] = [x for x in end_points if x]
            if data_test.any().any():
                data_woe_test[var] = [var_woe(x, bin_dic, not_in_list) for x in data_test[var].map(lambda x: float(x))]
            data_woe_train[var] = [var_woe(x, bin_dic, not_in_list) for x in data_train[var].map(lambda x: float(x))]
            iv = sum(var_stat['IV'])
            ks = max(var_stat['KS'])
            data_bin = pd.concat([data_bin, var_stat])
            iv_list.append(iv)
            ks_list.append(ks)
            iter += 1
        else:
            print(var, ' Should be checked')

    writer.save()
    writer.close()
    for i in not_var_list:
        data_woe_train[i]=data_train[i]
    data_woe_train.to_csv('{}/train_woe_{}.csv'.format(filepath, output_file), index=False)
    if data_test.any().any():
        for i in not_var_list:
            data_woe_test[i]=data_test[i]
        data_woe_test.to_csv('{}/test_woe_{}.csv'.format(filepath, output_file), index=False)
        return  data_woe_train,data_woe_test
    else:
        return data_woe_train


def dist_col_stat(df, key_, label, output_filename, dist_list, value_count=20, del_col=[], test_df=pd.DataFrame()):
    bad_count = df[label].sum()
    df_len = df.shape[0]
    del_list = []
    writer = pd.ExcelWriter("{}_dist.xlsx".format(output_filename),)
    j = 0
    for i in dist_list:
        locals()[i+'_1'] = df[[i, label, key_]]
        locals()[i+'_2'] = locals()[i+'_1'][label].groupby(locals()[i+'_1'][i]).sum().rename('bad').reset_index()
        locals()[i+'_3'] = locals()[i+'_1'][key_].groupby(locals()[i+'_1'][i]).count().rename(i+'_count').reset_index()
        locals()[i+'_4'] = pd.merge(locals()[i+'_3'], locals()[i+'_2'], on=i, how='left')
        locals()[i+'_4']['bad_rate'] = locals()[i+'_4']['bad']/locals()[i+'_4'][i+'_count']
        locals()[i+'_4']['good_percent'] = (locals()[i+'_4'][i+'_count']-locals()[i+'_4']['bad'])/(df_len-bad_count)
        locals()[i+'_4']['bad_percent'] = locals()[i+'_4']['bad']/bad_count
        locals()[i+'_4']['WOE'] = np.log(locals()[i+'_4']['good_percent'] / locals()[i+'_4']['bad_percent'])
        locals()[i+'_4']['rank_'+i]=locals()[i+'_4'].bad_rate.rank(axis=0, method='first')
        locals()[i+'_4']['KS'] = 0
        locals()[i+'_4']['var'] = i
        locals()[i+'_4']['good'] = locals()[i+'_4'][i+'_count']-locals()[i+'_4']['bad']
        locals()[i] = locals()[i+'_4'][[i,'rank_'+i]]
        if locals()[i+'_4'].shape[0] > value_count: #删除离散属性值过多的特征
            df[i]=df.merge(locals()[i], how='left', on=i)['rank_'+i]
            del_list.append(i)
            try:
                test_df[i]=test_df.merge(locals()[i],how='left',on=i)['rank_'+i]
            except:
                pass
        else:
            locals()[i+'_4']['IV']=(locals()[i+'_4']['good_percent']-locals()[i+'_4']['bad_percent'])*locals()[i+'_4']['WOE']
            df[i] = df.merge(locals()[i+'_4'], how='left', on=i)['WOE']
            try:
                test_df[i] = test_df.merge(locals()[i+'_4'], how='left', on=i)['WOE']
            except:
                pass
            locals()[i+'_4'].sort_values('rank_'+i).drop(['rank_'+i, 'good_percent', 'bad_percent'], axis=1).rename(columns ={i: 'Bin', i+'_count': 'total_count'}).to_excel(writer, 'dist_bin', startrow=j)
            j += locals()[i+'_4'].shape[0]+2
    
    writer.save()        
    writer.close()
    print(del_list)
    print('Those features have too many types!!!')
    return df, test_df


def get_woe_df(df, label, output_path, output_file, dist_col, serial_col, del_col, df_test=pd.DataFrame()):
    
    #填充
    df = df[del_col+dist_col+serial_col]
    if df_test.shape[0] != 0:
        df_test = df_test[del_col+dist_col+serial_col]
        df_test = df_test.reset_index()
    df.fillna(-99998, inplace=True)
    df_test.fillna(-99998, inplace=True)

    df = df.reset_index()

    if len(dist_col) > 0:
        df, df_test = dist_col_stat(df, 'index', label, output_path+'/'+output_file, dist_col, value_count=20, del_col=del_col, test_df=df_test)

    if len(serial_col) > 0:
        print('Cutting bins...')
        if df_test.shape[0]==0:
            df=df_woe1(output_path,output_file,label,df_test,df,not_var_list=del_col+dist_col+['index'],not_in_list=['-99998','-99998.0',-99998,-99998.0], target_var_list=serial_col,flag_var_list=[])        
        else:
            df,df_test=df_woe1(output_path,output_file,label,df_test,df,not_var_list=del_col+dist_col,not_in_list=['-99998','-99998.0',-99998,-99998.0], target_var_list=serial_col,flag_var_list=[])
        print(df.shape) 
         
    return df, df_test

def concat_file(output_path,output_file,dist_col,serial_col):
    #合并
    writer1=pd.ExcelWriter(output_path+'/'+output_file+'3.xlsx')
    if len(dist_col)>0:
        iv_info1=pd.read_excel(output_path+'/'+output_file+'_dist.xlsx')
        iv_info1['var']=iv_info1['var'].apply(lambda x:str(x))
        iv_info_ss1=iv_info1[(iv_info1['var']!='nan') & (iv_info1['var']!='var')]
        iv_info_ss12=iv_info1[(iv_info1['var']!='nan') & (iv_info1['var']!='var')&(iv_info1['Bin']!='-99998.0')&(iv_info1['Bin']!=-99998)]
        iv_info_ss1.replace('inf',0,inplace=True)
        fg1=iv_info_ss12[['total_count','var']].groupby(['var']).sum().reset_index()
#       woe_max1=iv_info_ss1.loc[iv_info_ss1.groupby(['var'])['WOE'].idxmax().tolist(),['var','Bin','WOE']]
#       woe_min1=iv_info_ss1.loc[iv_info_ss1.groupby(['var'])['WOE'].idxmin().tolist(),['var','Bin','WOE']]
        iv1=iv_info_ss1['IV'].groupby(iv_info_ss1['var']).sum().reset_index()
        ks1=iv_info_ss1['KS'].groupby(iv_info_ss1['var']).max().reset_index()
        iv_ks_info1=iv1.merge(ks1,on='var',how='inner').merge(fg1,on='var',how='inner')#.merge(woe_max1,on='var',how='inner').merge(woe_min1,on='var',how='inner')
        iv_info1[['Bin','IV','KS','WOE','bad','bad_rate','good','total_count','var']].to_excel(writer1,'bin_detail')
    else:
        iv_info1=pd.DataFrame()
    if len(serial_col)>0:
        iv_info=pd.read_excel(output_path+'/'+output_file+'.xlsx')
        iv_info['var']=iv_info['var'].apply(lambda x:str(x))
        iv_info_ss=iv_info[(iv_info['var']!='nan') & (iv_info['var']!='var')]
        iv_info_ss2=iv_info[(iv_info['var']!='nan') & (iv_info['var']!='var')&(iv_info['Bin']!='-99998.0')]
        iv_info_ss.replace('inf',0,inplace=True)
        fg=iv_info_ss2[['total_count','var']].groupby(['var']).sum().reset_index()
#       woe_max=iv_info_ss.loc[iv_info_ss.groupby(['var'])['WOE'].idxmax().tolist(),['var','Bin','WOE']]
#       woe_min=iv_info_ss.loc[iv_info_ss.groupby(['var'])['WOE'].idxmin().tolist(),['var','Bin','WOE']]
        iv=iv_info_ss['IV'].groupby(iv_info_ss['var']).sum().reset_index()
        ks=iv_info_ss['KS'].groupby(iv_info_ss['var']).max().reset_index()
        iv_ks_info=iv.merge(ks,on='var',how='inner').merge(fg,on='var',how='inner')#.merge(woe_max,on='var',how='inner').merge(woe_min,on='var',how='inner')
        iv_info[['Bin','IV','KS','WOE','bad','bad_rate','good','total_count','var']].to_excel(writer1,'bin_detail',startrow=iv_info1.shape[0]+3)
    else:
        iv_info=pd.DataFrame()

    #iv_ks_info=pd.merge(iv,ks,on='var',how='inner')
    
    #iv_ks_info1=pd.merge(iv1,ks1,on='var',how='inner')
    
    if len(serial_col)>0 and len(dist_col)>0:
        iv_ks_info_sort=pd.concat([iv_ks_info,iv_ks_info1])
    elif len(serial_col)>0:
        iv_ks_info_sort=iv_ks_info
    elif len(dist_col)>0:
        iv_ks_info_sort=iv_ks_info1

        
    iv_ks_info_sort=iv_ks_info_sort.sort_values(by=['IV'],ascending=[False])
    
    #writer1=pd.ExcelWriter(output_path+'/'+output_file+'3.xlsx')
    iv_ks_info_sort.to_excel(writer1,'bin_summary')
    #iv_info1[['Bin','IV','KS','WOE','bad','bad_rate','good','total_count','var']].to_excel(writer1,'bin_detail')
    #iv_info.to_excel(writer1,'bin_detail',startrow=iv_info1.shape[0]+3)
    writer1.save()
    writer1.close() 

#µÚ¶þ²½£ºmap woe value
from math import isnan
def var_woe(x, bin_dic, not_in_list):
    val = None
    if pd.isnull(x) or isnan(x):
        for woe in bin_dic:
            if bin_dic[woe] in ['nan', 'NaN']:
                val = woe
    elif x in not_in_list:
        for woe in bin_dic:
            end_points = bin_dic[woe]
            len_points = len(end_points)
            if len_points == 1 and end_points[0] != 'nan':
                if x == float(end_points[0]):
                    val = woe
    else:
        for woe in bin_dic:
            end_points = bin_dic[woe]
            if len(end_points) == 2 and end_points[0] != 'nan':
                if end_points[0] == '-inf':
                    if x <= float(end_points[1]):
                        val = woe
                elif end_points[1] == 'inf':
                    if x > float(end_points[0]):
                        val = woe
                elif (x > float(end_points[0])) & (x <= float(end_points[1])):
                    val = woe
            elif len(end_points) > 2:
                if end_points[0] == '-inf' and end_points[-1] == 'inf':
                    val = woe
                elif end_points[0] == '-inf':
                    if x <= float(end_points[-1]):
                        val = woe
                elif end_points[-1] == 'inf':
                    if x > float(end_points[0]):
                        val = woe
                elif (x > float(end_points[0])) & (x <= float(end_points[-1])):
                    val = woe
    return val


def agg_func(x):
    str_value = ""
    for value in x:
        str_value += value
        str_value += ", "
    return str_value


def df_woe2(filepath, output_file, flag_name, bin_file, data_test, not_var_list=[], not_in_list=[], target_var_list=[],
           flag_var_list=['custr_nbr']):
    """
    :param flag_name: bad flag
    :param data_test/data_train: target data
    :param data_train: input data to calculate the bin
    :param piece: number of bins in final result
    :param rate: number of minimum percent in one bin
    :param not_in_list: special values should be treated
    :param not_var_list: special variables should be treated
    :return:
    """
    output_filename = filepath + "/" + output_file
    #writer = pd.ExcelWriter("{}.xlsx".format(output_filename))
    flag_var_list.append(flag_name)
    data_woe_test = data_test[flag_var_list]
    var_list = data_test.columns.tolist()
    var_left = list(set(var_list) - set(flag_var_list) - set([flag_name]))
    if target_var_list:
        target_var_list = target_var_list
    else:
        not_var_list.extend(flag_var_list)
        not_var_list.append(flag_name)
        not_var_list.append('time_stamp')
        not_in_list.append(['None', 'nan'])
        target_var_list = list(set(var_list) - set(not_var_list))
    if len(target_var_list) == 0:
        print('No variable available for analysis')
        return pd.DataFrame()
    bin_info = pd.read_excel(bin_file, 'bin')
    iter = 0
    i = 0
    for var in var_left:
        #         print(var)
        #         try:
        var_stat = bin_info[bin_info['var'] == var][['var', 'WOE', 'Bin']]
        #             var_stat.to_excel(writer,'bin',startrow=i)
        #             len_df=var_stat.shape[0]+1
        #             i+=len_df+2
        print(var_stat)
        if len(var_stat) > 0:
            var_stat_new = var_stat['Bin'].groupby(var_stat['WOE']).apply(lambda x: agg_func(x)).reset_index()
            bin_dic = dict(list(zip(var_stat_new['WOE'], var_stat_new['Bin'])))
            for woe in bin_dic:
                match_case = re.compile("\(|\)|\[|\]")
                end_points = match_case.sub('', bin_dic[woe]).split(', ')
                bin_dic[woe] = [x for x in end_points if x]
            #                 print bin_dic
            data_woe_test[var] = [var_woe(x, bin_dic, not_in_list) for x in data_test[var].map(lambda x: float(x))]
            #                 data_bin = pd.concat([data_bin, var_stat])
            iter += 1
        else:
            print(var, ' Should be checked')
    data_woe_test.to_csv('{}/test_woe_{}.csv'.format(filepath, output_file), index=False)
    return data_woe_test



#map woe
#df=pd.read_csv('/home/bangsun/RoyGao/widetable.csv')
#df=df.fillna(-99998)
#df_woe('.','wide_table','flag_ever_d30','info_wide_table.xlsx', df,not_var_list=['id_card','id_name','id_phone'],
#       not_in_list=['-99998','-99998.0',-99998,-99998.0], target_var_list=[],
#             flag_var_list=['loan_id'])


###step 3£ºmodel selection
####encoding:utf-8
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import pickle
import math


def direction_ar_auc(df, list_var, y, name):
    """
    :param df:
    :param not_var:
    :param y:
    :param name:
    :return:
    """
    # var_set = set(list(df.columns))
    # not_var_set = set(not_var)
    # list_var = var_set - not_var_set
    data_var_info = []
    for val in list_var:
        dic_value = {}
        x_auc = roc_auc_score(df[y], df[val].map(lambda x: float(x)))
        x_ar = x_auc * 2 - 1
        dic_value['var'] = val
        dic_value['ar_' + str(name)] = x_ar
        if x_ar > 0:
            direction = 'great'
        elif x_ar < 0:
            direction = 'less'
        else:
            direction = 'equal'
        dic_value['direction_' + str(name)] = direction
        data_var_info.append(dic_value)
    pd_var = pd.DataFrame(data_var_info)
    return pd_var


def compare_ar_direction(df_train, df_test, flag_name, list_var):
    """
    :param df_train:
    :param df_test:
    :param filepath:
    :param filename:
    :param flag_name:
    :param not_ar_var_list:
    :return:
    """
    # not_ar_var_list.append(flag_name)
    res_train = direction_ar_auc(df_train, list_var, flag_name, 'train')
    res_test = direction_ar_auc(df_test, list_var, flag_name, 'test')
    res_full = pd.merge(res_train, res_test, on='var', how='inner')
    res_full_s = res_full[
        (res_full['direction_train'] == res_full['direction_test']) & (res_full['direction_test'] != 'equal')]
    # res_full_s.to_csv('{}/info_direction.csv'.format(filepath),index=False)
    return list(res_full_s['var'])


def delete_inconsistent_coefficient(df_train, var_list, flag_name):
    not_var2 = []
    var_count = 1
    x_train = df_train[var_list]
    x_train = sm.add_constant(x_train)
    y_train = df_train[flag_name]
    logit = sm.Logit(y_train, x_train)
    result = logit.fit()
    var_coef = result.params
    var_coef = var_coef.reset_index()
    var_coef.columns = ['var', 'coef']
    # var_ivks.columns = ['var','iv','ks']
    # train_ivcoef = pd.merge(var_coef,var_ivks,on='var',how='left')
    var_p = result.pvalues
    var_p = pd.DataFrame(var_p)
    var_p = var_p.reset_index()
    var_p.columns = ['var', 'p']
    var_coef2 = var_coef.loc[var_coef['coef'] > 0]
    var_count = len(var_coef2['var'])
    if var_count != 0:
        not_var_p = list(var_coef2['var'])
    else:
        not_var_p = []
    not_var2 = not_var2 + not_var_p
    var = list(set(var_list) - set(not_var_p))
    return var


def variance_inflation_factor(exog, exog_idx):
    '''variance inflation factor, VIF, for one exogenous variable

    The variance inflation factor is a measure for the increase of the
    variance of the parameter estimates if an additional variable, given by
    exog_idx is added to the linear regression. It is a measure for
    multicollinearity of the design matrix, exog.

    One recommendation is that if VIF is greater than 5, then the explanatory
    variable given by exog_idx is highly collinear with the other explanatory
    variables, and the parameter estimates will have large standard errors
    because of this.

    Parameters
    ----------
    exog : ndarray, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression
    exog_idx : int
        index of the exogenous variable in the columns of exog

    Returns
    -------
    vif : float
        variance inflation factor

    Notes
    -----
    This function does not save the auxiliary regression.

    See Also
    --------
    xxx : class for regression diagnostics  TODO: doesn't exist yet

    References
    ----------
    http://en.wikipedia.org/wiki/Variance_inflation_factor

    '''
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


def get_vif(df_train, var_list, dic_iv_rank):
    x_train = df_train[var_list].iloc[:, :].values
    dic_new = {}
    for i in range(len(var_list)):
        vif_value = variance_inflation_factor(x_train, i)
        dic_new[var_list[i]] = vif_value
    df_vif = pd.DataFrame({'var': list(dic_new.keys()), 'vif': list(dic_new.values())})
    # df_vif.columns = ['var', 'vif']
    df_iv = pd.DataFrame({'var': list(dic_iv_rank.keys()), 'IV': list(dic_iv_rank.values())})
    # df_iv.columns = ['var', 'IV']
    df_vif_iv = pd.merge(df_vif, df_iv, on='var', how='inner')
    vif_iv_sort = df_vif_iv.sort_values(by=['vif', 'IV'], ascending=[False, True])
    vif_iv_sort = vif_iv_sort.reset_index()
    return vif_iv_sort


def cal_vif_value(var_list, df_train, vif_threshold, dic_iv_rank):
    """
    :param var_list:
    :param df_train:
    :param vif_threshold:
    :param dic_iv_rank:
    :return:
    """
    delete_list = []
    vif_iv_sort = get_vif(df_train, var_list, dic_iv_rank)
    vif_max = vif_iv_sort.loc[0, 'vif']
    vif_need_var = vif_iv_sort['var'].tolist()
    while vif_max >= vif_threshold:
        select_var_all = vif_iv_sort[vif_iv_sort['vif'] >= vif_threshold]
        select_var_s = select_var_all.sort_values('IV', ascending=True)
        delete_var = select_var_s['var'].tolist()[0]
        # if delete_var in vif_need_var:
        vif_need_var.remove(delete_var)
        # print delete_var, vif_need_var
        # else:
        #     vif_need_var=vif_need_var
        if len(vif_need_var) >= 2:
            # return vif_need_var
            vif_iv_sort = get_vif(df_train, vif_need_var, dic_iv_rank)
            # print  vif_iv_sort
            vif_max = vif_iv_sort.loc[0, 'vif']
    return vif_need_var


def func_sort_col(col, df_, code):
    cols = col[:]
    x_train = df_[col]
    x_train = sm.add_constant(x_train)
    logit_1 = sm.Logit(df_[code], x_train)
    result_1 = logit_1.fit()
    wald_chi2 = np.square((result_1.params) / np.square(result_1.bse))
    a = pd.DataFrame(wald_chi2, columns=['value'])
    b = a.sort_values('value', ascending=False)
    sorted_cols = b.index.tolist()
    if 'const' in sorted_cols:
        sorted_cols.remove('const')
    return sorted_cols


import re


def func_stepwise_1(cols, df_, code, sort):
    train_cols = []
    if sort:
        sorted_cols = func_sort_col(cols, df_, code)
    else:
        sorted_cols = cols[:]
    sorted_cols_filter = [i for i in sorted_cols if re.match('^WOE_i', i) or re.match('^WOE_m', i)]
    sorted_cols_leave = list(set(sorted_cols) - set(sorted_cols_filter))
    sorted_cols_filter.extend(sorted_cols_leave)
    #print sorted_cols_filter
    for i in sorted_cols_filter:
        # if re.match('^WOE_i',i) or re.match('^WOE_m',i):
        # print 'has looped to '+str(i)
        train_cols.append(i)
        if 'const' in train_cols:
            train_cols.remove('const')
        else:
            train_cols = train_cols
        x_train = df_[train_cols]
        x_train = sm.add_constant(x_train)
        logit = sm.Logit(df_[code], x_train)
        result = logit.fit()
        train_cols = result.pvalues[result.pvalues < 0.05].index.tolist()
    if 'const' in train_cols:
        train_cols.remove('const')
    else:
        train_cols = train_cols
    x_train = df_[train_cols]
    x_train = sm.add_constant(x_train)
    logit = sm.Logit(df_[code], x_train)
    result = logit.fit()
    #print((result.summary()))
    #print(('aic:' + str(result.aic)))
    return train_cols


def cal_score(yp_train, flag, base_score, double_score, odds):
    p = yp_train
    rs = pd.DataFrame({'p': yp_train, 'flag': flag})
    B = double_score / math.log(2)
    A = base_score + B * math.log(odds)
    print((A, B))
    p_ = []
    for i in range(len(p)):
        p_.append(1 - p[i])
    df_rs = rs.copy()
    df_rs.loc[:, 'p_'] = p_
    df_rs.loc[:, 'log_odds'] = np.log(df_rs.loc[:, 'p'] / df_rs.loc[:, 'p_'])
    df_rs.loc[:, 'score'] = np.array([A for _ in range(len(df_rs))]) - np.array([B for _ in range(len(df_rs))]) * np.log(
        df_rs.loc[:, 'p'] / df_rs.loc[:, 'p_'])
    df_rs['score'] = df_rs['score'].apply(lambda x: int(x))
    return df_rs, A, B


def binning_person(col, cut_num):
    colBin = pd.qcut(col, cut_num,duplicates='drop')
    return colBin


# def map_value(x):
#     value=x.replace('[','').replace(']','')
#     value[0]

def get_data_4_ar_ks(df_train_score, cut_num, score_trigger):
    """
    :param result:
    :param x_train:
    :param y_train:
    :param cut_num:
    :return:
    """
    flag = df_train_score['flag'].tolist()
    if score_trigger:
        res_1 = binning_person(df_train_score['score'].tolist(), cut_num)
    else:
        res_1 = binning_person(df_train_score['log_odds'].tolist(), cut_num)
    df_s1 = pd.DataFrame({'score': res_1, 'y': flag})
    res_s1 = pd.crosstab(df_s1.score, df_s1.y, rownames=['score'], colnames=['y'])
    res_train = res_s1.reset_index()
    #     res_train['score']=res_train['score'].apply(lambda x:str([x.left,x.right]))
    return res_train


def get_AR_KS(res):
    """
    :param res:
    :return:
    """
    res.columns = ['score', 'good', 'bad']
    total_good = res['good'].sum()
    total_bad = res['bad'].sum()
    res['good_per'] = res['good'] / total_good
    res['bad_per'] = res['bad'] / total_bad
    res['cum_good_per'] = np.cumsum(res['good_per'], axis=0)
    res['cum_bad_per'] = np.cumsum(res['bad_per'], axis=0)
    res['total'] = res['good'] + res['bad']
    res['bad_rate'] = res['bad'] / res['total']
    cum_bad_rate = []
    cum_good_rate = []
    for i in range(len(res)):
        cum_bad_rate.append(float(sum(res['bad'].loc[i:len(res)])) / float(total_bad))
        cum_good_rate.append(float(sum(res['good'].loc[i:len(res)])) / float(total_good))
    res['cum_bad_rate'] = pd.Series(cum_bad_rate, index=res.index)
    res['cum_good_rate'] = pd.Series(cum_good_rate, index=res.index)
    res['KS'] = abs(res['cum_bad_per'] - res['cum_good_per'])
    AUC = []
    auc_0 = 0.5 * res['cum_bad_per'].loc[0] * res['cum_good_per'].loc[0]
    AUC.append(auc_0)
    for i in range(1, len(res)):
        value = 0.5 * float(res['cum_bad_per'].loc[i] + res['cum_bad_per'].loc[i - 1]) * float(
            res['cum_good_per'].loc[i] - res['cum_good_per'].loc[i - 1])
        AUC.append(value)
    res['AUC'] = pd.Series(AUC, index=res.index)
    return res


def map_cut_bin(x, score_value_list):
    list_content = []
    for i in range(len(score_value_list)):
        if i == 0:
            score_value = score_value_list[0]
            if x <= score_value[1]:
                list_content.append(i)
        elif i == len(score_value_list) - 1:
            score_value = score_value_list[i]
            if x > score_value[0]:
                list_content.append(i)
        else:
            score_value = score_value_list[i]
            if score_value[0] <= x and x <= score_value[1]:
                list_content.append(i)
    if list_content:
        return list_content[0]
    else:
        return x


def get_data_for_psi(res_train, name):
    """
    :param res_train:
    :param cut_num:
    :param name:
    :return:
    """
    train_value = pd.DataFrame({'total': res_train['total']})
    train_all = train_value['total'].sum()
    train_value['per_' + str(name)] = train_value['total'] / train_all
    return train_value


def cal_model_psi(info_train, info_test, cut_num):
    """
    :param df_train:
    :param df_test:
    :param cut_num:
    :param var_list:
    :param result:
    :return:
    """
    df_train_res = get_data_for_psi(info_train, 'train')
    df_test_res = get_data_for_psi(info_test, 'test')
    df_full = pd.concat([df_train_res, df_test_res], axis=1).reset_index(drop=True)
    if df_full.shape[0] != cut_num:
        print('merge data in psi calculate is error')
    else:
        df_full['per_series'] = df_full['per_test'] - df_full['per_train']
        df_full['ln_series'] = np.log(df_full['per_test'] / df_full['per_train'])
        df_full['psi_value'] = df_full['per_series'] * df_full['ln_series']
        psi_total = df_full['psi_value'].sum()
        return psi_total


# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import os
import matplotlib as mpl
import os

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy  as  np


def KS_curve(df, data_name, cut_name):
    plt.figure(data_name)
    fig, ax = plt.subplots()
    x = df['score'].tolist()
    y_ks = df['KS'].tolist()
    y_g = df['cum_good_per'].tolist()
    y_b = df['cum_bad_per'].tolist()
    ks_max = np.max(y_ks)
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 20,
            }
    xticks = list(range(0, len(x)))
    xlabels = [x[index] for index in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=40)
    plt.plot(y_ks)
    plt.plot(y_g)
    plt.plot(y_b)
    plt.grid()
    plt.show()
    plt.plot(y_ks, 'c-.', label='Kolmogorov-Smirnov (KS): %0.2f' % ks_max)
    plt.plot(y_g, 'k-', label='cumulate good per')
    plt.plot(y_b, 'b-', label='cumulate good per')
    # plt.title('KS curve',size=20)
    plt.title('Kolmogorov-Smirnov curve', fontdict=font)
    # plt.legend(prop={'family':'SimHei','size':8} ,loc="best",shadow=True)
    # plt.legend(loc="lower right")  # set legend location
    # plt.ylabel('cumulate percentage',size=15)  # set ystick label
    # plt.xlabel('score',size=15)  # set xstck label
    plt.grid()
    plt.savefig('KS_curve_ss_{}_{}.png'.format(data_name, cut_name), bbox_inches='tight')


# reload(sys)
# sys.setdefaultencoding('utf8')
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def get_roc_curve(df_train_ss, flag_name, pred_var, data_name, cut_name):
    plt.figure(data_name)
    y_train = df_train_ss[flag_name].values
    y_score = df_train_ss[pred_var].values
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_train, y_score)
    roc_auc = auc(fpr, tpr)
    # # Compute micro-average ROC curve and ROC area
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(ROC) curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC_cut_on_ss_{}_{}.png'.format(data_name, cut_name))


def get_AR_KS_res_amount(df_train, df_test, var_list, flag_name, cut_num, score_trigger, base_score, double_score,
                         odds_value):
    """
    :param df_train:
    :param df_test:
    :param yp_train:
    :param yp_test:
    :param flag_name:
    :param cut_num:
    :param score_trigger:
    :param base_score:
    :param double_score:
    :return:
    """
    if not odds_value:
        y_1 = df_train[df_train[flag_name] == 1]
        y_0 = df_train[df_train[flag_name] == 0]
        odds = float(y_1.shape[0]) / float(y_0.shape[0])
    else:
        odds = odds_value
    x_train = df_train[var_list]
    x_train = sm.add_constant(x_train)
    y_train = df_train[flag_name]
    logit = sm.Logit(y_train, x_train)
    result = logit.fit()
    param_res = result.params
    yp_train = result.predict(x_train).tolist()
    flag_train = y_train.tolist()
    df_train_score, A, B = cal_score(yp_train, flag_train, base_score, double_score, odds)
    res_train = get_data_4_ar_ks(df_train_score, cut_num, score_trigger)
    info_train = get_AR_KS(res_train)
    score_list = info_train['score'].tolist()
    # return score_list
    #     score_value=[x.replace('[','').replace(']','').split(',') for x in score_list]
    score_value = [str(x).replace('(', '').replace('[', '').replace(')', '').replace(']', '').split(',') for x in score_list]
    score_value_list = [[float(mm) for mm in x] for x in score_value]
    ks_max_train = info_train['KS'].max()
    auc_train = info_train['AUC'].sum()
    df_info = pd.DataFrame({'train_auc': [auc_train], 'train_ks': [ks_max_train]})
    # KS_curve(info_train,'train','amount')
    # get_roc_curve(df_train_score, 'flag', 'p', 'train', 'amount')
    if df_test.any().any():
        flag_test = df_test[flag_name].tolist()
        x_test = df_test[var_list].iloc[:, :].values
        x_test = sm.add_constant(x_test)
        yp_test = result.predict(x_test).tolist()
        df_test_score, A, B = cal_score(yp_test, flag_test, base_score, double_score, odds)
        if score_trigger:
            df_test_score['score'] = df_test_score['score'].apply(lambda x: map_cut_bin(x, score_value_list))
        else:
            df_test_score['score'] = df_test_score['log_odds'].apply(lambda x: map_cut_bin(x, score_value_list))
        # df_test_score['score']=info_train['score']
        df_test_score_res = pd.DataFrame({'score': df_test_score['score'].tolist(), 'y': flag_test})
        test_score = pd.crosstab(df_test_score_res.score, df_test_score_res.y, rownames=['score'], colnames=['y'])
        test_score = test_score.reset_index()
        info_test = get_AR_KS(test_score)
        info_test['score'] = info_train['score']
        ks_max_test = info_test['KS'].max()
        auc_test = info_test['AUC'].sum()
        psi_total = cal_model_psi(info_train, info_test, cut_num)
        df_info = pd.DataFrame({'train_auc': [auc_train], 'train_ks': [ks_max_train],
                                'test_auc': [auc_test], 'test_ks': [ks_max_test], 'psi': [psi_total]})
        KS_curve(info_test, 'test', 'amount')
        get_roc_curve(df_test_score, 'flag', 'p', 'test', 'amount')
    if df_test.any().any():
        return A, B, param_res, info_train, info_test, df_info
    else:
        return A, B, param_res, info_train, df_info


def binning_score(col, break_points):
    colBin = pd.cut(col, bins=break_points, include_lowest=True)
    return colBin


def get_data_4_ar_ks_score(df_train_score, cut_num, score_trigger):
    """
    :param result:
    :param x_train:
    :param y_train:
    :param cut_num:
    :return:
    """
    flag = df_train_score['flag'].tolist()
    if score_trigger:
        res_1 = binning_score(df_train_score['score'].tolist(), cut_num)
    else:
        res_1 = binning_score(df_train_score['log_odds'].tolist(), cut_num)
    df_s1 = pd.DataFrame({'score': res_1, 'y': flag})
    res_s1 = pd.crosstab(df_s1.score, df_s1.y, rownames=['score'], colnames=['y'])
    res_train = res_s1.reset_index()
    #     res_train['score']=res_train['score'].apply(lambda x:str([int(x.left),int(x.right)]))
    return res_train


def get_AR_KS_res_score(df_train, df_test, var_list, flag_name, cut_num, score_trigger, base_score, double_score,
                        odds_value=False):
    """
    :param df_train:
    :param df_test:
    :param yp_train:
    :param yp_test:
    :param flag_name:
    :param cut_num:
    :param score_trigger:
    :param base_score:
    :param double_score:
    :return:
    """
    if not odds_value:
        y_1 = df_train[df_train[flag_name] == 1]
        y_0 = df_train[df_train[flag_name] == 0]
        odds = float(y_1.shape[0]) / float(y_0.shape[0])
    else:
        odds = odds_value
    x_train = df_train[var_list]
    x_train = sm.add_constant(x_train)
    y_train = df_train[flag_name]
    logit = sm.Logit(y_train, x_train)
    result = logit.fit()
    res_param = result.params
    yp_train = result.predict(x_train).tolist()
    flag_train = y_train.tolist()
    df_train_score, A, B = cal_score(yp_train, flag_train, base_score, double_score, odds)
    res_train = get_data_4_ar_ks_score(df_train_score, cut_num, score_trigger)
    info_train = get_AR_KS(res_train)
    score_list = info_train['score'].tolist()
    score_value = [str(x).replace("(", "").replace(")", "").replace('[', '').replace(']', '').split(',') for x in score_list]
    score_value_list = [[float(mm) for mm in x] for x in score_value]
    ks_max_train = info_train['KS'].max()
    auc_train = info_train['AUC'].sum()
    df_info = pd.DataFrame({'train_auc': [auc_train], 'train_ks': [ks_max_train]})
    KS_curve(info_train, 'train', 'score')
    get_roc_curve(df_train_score, 'flag', 'p', 'train', 'score')
    if df_test.any().any():
        flag_test = df_test[flag_name].tolist()
        x_test = df_test[var_list].iloc[:, :].values
        x_test = sm.add_constant(x_test)
        yp_test = result.predict(x_test).tolist()
        df_test_score, A, B = cal_score(yp_test, flag_test, base_score, double_score, odds)
        if score_trigger:
            df_test_score['score'] = df_test_score['score'].apply(lambda x: map_cut_bin(x, score_value_list))
        else:
            df_test_score['score'] = df_test_score['log_odds'].apply(lambda x: map_cut_bin(x, score_value_list))
        df_test_score_res = pd.DataFrame({'score': df_test_score['score'].tolist(), 'y': flag_test})
        test_score = pd.crosstab(df_test_score_res.score, df_test_score_res.y, rownames=['score'], colnames=['y'])
        test_score = test_score.reset_index()
        info_test = get_AR_KS(test_score)
        ks_max_test = info_test['KS'].max()
        auc_test = info_test['AUC'].sum()
        psi_total = cal_model_psi(info_train, info_test, cut_num)
        df_info = pd.DataFrame({'train_auc': [auc_train], 'train_ks': [ks_max_train],
                                'test_auc': [auc_test], 'test_ks': [ks_max_test], 'psi': [psi_total]})
        KS_curve(info_test, 'test', 'score')
        get_roc_curve(df_test_score, 'flag', 'p', 'test', 'score')
    if df_test.any().any():
        return A, B, res_param, info_train, info_test, df_info
    else:
        return A, B, res_param, info_train, df_info


def delete_none_p_var(df_train, var_list, flag_name):
    not_var3 = []
    x_train = df_train[var_list]
    x_train = sm.add_constant(x_train)
    y_train = df_train[flag_name]
    logit = sm.Logit(y_train, x_train)
    result = logit.fit()
    var_p = result.pvalues
    var_p = var_p.reset_index()
    var_p.columns = ['var', 'varp']
    # return var_p
    var_null = list(var_p[pd.isnull(var_p['varp'])]['var'])
    not_var_p = list(var_p[(var_p['var'] != 'const')]['var'])
    if not_var_p:
        all_var3 = not_var3 + not_var_p
        var = list(set(not_var_p) - set(var_null))
    return var


def get_dic_iv_rank(iv_res):
    iv_res_sort = iv_res.sort_values('IV', ascending=False)
    iv_res_sort_s = iv_res_sort.reset_index()
    iv_res_sort_s['rank'] = iv_res_sort_s.index
    iv_res_sort_ss = iv_res_sort_s.drop(['index', 'IV'], 1)
    dic_iv_rank = iv_res_sort_ss.to_dict(orient='records')
    list_iv_rank = [(str(x['var']), x['rank']) for x in dic_iv_rank]
    dic_iv_rank = dict(list_iv_rank)
    return dic_iv_rank


def random_forest_result(df_train_s, df_test_s, fea_list, flag_name, top_num):
    x_train = df_train_s[fea_list]
    y_train = df_train_s[flag_name]
    if df_test_s.any().any():
        x_test = df_test_s[fea_list]
        y_test = df_test_s[flag_name]
    max_ROC = 0
    sample_leaf_options = [20]
    select_fea = ['sqrt']
    weight_class = ['balanced']
    for method in select_fea:
        for leaf_size in sample_leaf_options:
            for weight in weight_class:
                clf = RandomForestClassifier(bootstrap=True, class_weight=weight, criterion='gini',
                                             max_depth=None, max_features=method, max_leaf_nodes=None,
                                             min_samples_leaf=leaf_size, min_samples_split=2,
                                             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                                             oob_score=False, random_state=0, verbose=0, warm_start=False)
                clf.fit(x_train, y_train)
                train_roc = roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1])
                final_clf = clf
    final_clf.fit(x_train, y_train)
    importances = final_clf.feature_importances_
    df_rf = pd.DataFrame({'fea': fea_list, 'import': importances})
    df_rf_s = df_rf.sort_values('import', ascending=False)
    var_list = df_rf_s[:top_num]['fea'].tolist()
    return df_rf_s, var_list


def filter_by_corr(df_, series_, corr_limit=0.7):
    series_ = [i for i in series_
               if i in df_.columns]
    drop_set, var_set = set(), set(series_)
    for i in series_:
        if i in var_set:
            var_set.remove(i)
        if i not in drop_set:
            drop_set |= {v for v in var_set if np.corrcoef(
                df_[i].values, df_[v].values)[0, 1] >= corr_limit}
            var_set -= drop_set
    return [i for i in series_ if i not in drop_set]




####deal with the variables
def select_var_part(df_woe_train, df_woe_test, filepath, output_file, flag_name, iv_file, vif_threshold,
                    score_trigger, base_score, double_score, exclude_var, target_var_list, odds_value, cut_num=10,
                    corr_limit=0.7):
    output_filename = filepath + '/' + output_file
    writer = pd.ExcelWriter(output_filename)
    iv_info_s = pd.read_excel(iv_file, 'bin_summary')
    dic_iv_rank = get_dic_iv_rank(iv_info_s)
    if target_var_list:
        var_list = target_var_list
    else:
        list_var = list(df_woe_train.columns)
        list_var.remove(flag_name)
        var_list = list(set(list_var) - set(exclude_var))
    series_ = list(set(iv_info_s['var']))
    df_train_ss = df_woe_train[var_list]
    #print var_list
    corr_filter_var = filter_by_corr(df_train_ss, series_, corr_limit)
    print('=====================================================================================================')
    print(('corr_var', corr_filter_var))
    print('=====================================================================================================')
    vif_selected_var = cal_vif_value(corr_filter_var, df_woe_train, vif_threshold, dic_iv_rank)
    #print vif_selected_var
    coef_selected_var = delete_inconsistent_coefficient(df_woe_train, vif_selected_var, flag_name)
    print('=====================================================================================================')
    print(('delete none positive', coef_selected_var))
    print('=====================================================================================================')
    final_selected_var = func_stepwise_1(coef_selected_var, df_woe_train, flag_name, True)
    print('=====================================================================================================')
    print(('final', final_selected_var))
    print('=====================================================================================================')
    ####step 4:calculate the final model indicatior
    res_final_model_amount = get_AR_KS_res_amount(df_woe_train, df_woe_test, final_selected_var, flag_name, cut_num,
                                                  score_trigger, base_score, double_score, odds_value)
    res_final_model_score = get_AR_KS_res_score(df_woe_train, df_woe_test, final_selected_var, flag_name, cut_num,
                                                score_trigger, base_score, double_score, odds_value)
    A = res_final_model_score[0]
    B = res_final_model_score[1]
    res_param = res_final_model_score[2]
    nn = res_param.shape[0]
    iv_detail = pd.read_excel(iv_file, 'bin_detail')
    iv_detail['WOE']=iv_detail['WOE'].replace('inf',0)
    iv_detail['WOE']=iv_detail['WOE'].replace('-inf',0)
    i = 0
    for var in final_selected_var:
        iv_var = iv_detail[iv_detail['var'] == var][['Bin', 'var', 'WOE', 'bad_rate', 'bad', 'good']]
        var_param = res_param[var]
        iv_var['score']= (A - B*res_param['const'])/ (nn-1) - var_param * iv_var['WOE'] * B
        iv_var.to_excel(writer, 'model_var_score', startrow=i)
        len_df = iv_var.shape[0] + 1
        i += len_df + 2
    df_woe = iv_info_s.loc[iv_info_s['var'].isin(final_selected_var),]
    # res_final_model_amount[2].to_excel(writer,'fin_model_indicator_amount')
    res_final_model_amount[3].to_excel(writer, 'model_trainset_indicator_amount')
    res_final_model_amount[4].to_excel(writer, 'model_tsetset_indicator_amount')
    # res_final_model_score[2].to_excel(writer,'fin_model_indicator_socre')
    res_final_model_score[3].to_excel(writer, 'model_trainset_indicator_score')
    res_final_model_score[4].to_excel(writer, 'model_tsetset_indicator_score')
    df_woe.to_excel(writer, 'bin_detail')
    # df_iv.to_excel(writer,'iv_psi_info')
    writer.save()        
    writer.close()
    return final_selected_var


def scorecard2sql(iv_file):
    df1=pd.read_excel(iv_file,sheet_name='model_var_score',index_col=0)
    df1=df1[(df1['Bin'].isna()==False)&(df1['Bin']!='Bin')]
    df1.reset_index(drop=True,inplace=True)
    for i in df1.index.tolist():
        if df1.loc[i, 1] != df1.loc[i-1, 1]:
            print(',case ', end='')
        if ',' in str(df1.loc[i, 0]):
            if '-inf' in str(df1.loc[i, 0]):
                print('when '+str(df1.loc[i, 'var'])+' <= '+df1.loc[i,0].strip('(]').split(',')[1]+' then '+str(df1.loc[i, 'score']))
            elif 'inf' in str(df1.loc[i, 0]):
                print('when '+str(df1.loc[i, 'var'])+' > '+df1.loc[i,0].strip('(]').split(',')[0]+' then '+str(df1.loc[i, 'score']))
            else:
                print('when '+str(df1.loc[i,'var'])+' between '+df1.loc[i,0].strip('(]').split(',')[0]+' and '+df1.loc[i, 0].strip('(]').split(',')[1] + ' then '+str(df1.loc[i,'score']))
        elif (df1.loc[i,'Bin'] == '-99998')|(df1.loc[i,'Bin']=='-99998.0')|(df1.loc[i, 'Bin']==-99998):
            print('when '+str(df1.loc[i,'var'])+' is null or  length('+str(df1.loc[i, 'var'])+')=0 then '+str(df1.loc[i, 'score']))
        else:
            print('when '+str(df1.loc[i,'var'])+" = '"+str(df1.loc[i, 'Bin'])+"' then "+str(df1.loc[i, 'score']))
        if i == df1.shape[0]-1:
            print('end')
            break
        if df1.loc[i, 1] != df1.loc[i+1, 1]:
            print('end')