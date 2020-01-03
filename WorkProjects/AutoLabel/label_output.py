#!/usr/bin/env python3
# encoding: utf-8
'''
@author: Cai
@contact: yuwei.chen@yunzhenxin.com
@application:
@time: 2019/10/14 15:06
@desc: 输出报告
'''

import pandas as pd
import numpy as np
import math


class LabelOutput(object):

    def __init__(self, cus_df, output_path):
        self.config_df = pd.read_csv(
            "/Users/cai/Desktop/pythonProjects/local_pyProjects/WorkProjects/AutoLabel/config_files/full_config.csv",
            sep='|')
        self.continuous_var = ['ft_app_ins_current_finance_cnt_avg'
            , 'ft_app_ins_current_finance_cnt_diff_m_llm'
            , 'ft_app_ins_current_finance_cnt_diff_m_lm'
            , 'ft_app_ins_current_finance_cnt_std'
            , 'ft_app_ins_current_micro_loan_cnt_avg'
            , 'ft_app_ins_current_micro_loan_cnt_diff_m_llm'
            , 'ft_app_ins_current_micro_loan_cnt_diff_m_lm'
            , 'ft_app_ins_current_micro_loan_cnt_std'
            , 'ft_app_ins_current_finance_cnt_m_rate'
            , 'ft_app_ins_current_debit_credit_cnt_m_rate'
            , 'ft_app_ins_current_micro_loan_cnt_m_rate'
            , 'ft_app_act_30d_finance_cnt_m_rate'
            , 'ft_app_act_30d_finance_times_m_rate'
            , 'ft_app_act_30d_finance_days_m_rate'
            , 'ft_app_act_30d_debit_credit_cnt_m_rate'
            , 'ft_app_act_30d_debit_credit_times_m_rate'
            , 'ft_app_act_30d_debit_credit_days_m_rate'
            , 'ft_app_act_30d_micro_loan_cnt_m_rate'
            , 'ft_app_act_30d_micro_loan_times_m_rate'
            , 'ft_app_act_30d_micro_loan_days_m_rate'
            , 'ft_app_ins_current_finance_cnt_cv'
            , 'ft_app_ins_current_debit_credit_cnt_cv'
            , 'ft_app_ins_current_micro_loan_cnt_cv'
            , 'ft_app_act_30d_finance_cnt_cv'
            , 'ft_app_act_30d_finance_times_cv'
            , 'ft_app_act_30d_finance_days_cv'
            , 'ft_app_act_30d_debit_credit_cnt_cv'
            , 'ft_app_act_30d_debit_credit_times_cv'
            , 'ft_app_act_30d_debit_credit_days_cv'
            , 'ft_app_act_30d_micro_loan_cnt_cv'
            , 'ft_app_act_30d_micro_loan_times_cv'
            , 'ft_app_act_30d_micro_loan_days_cv'
            , 'ft_app_ins_current_risk_cnt'
            , 'ft_app_ins_history_risk_cnt'
            , 'ft_app_ins_current_cheating_cnt'
            , 'ft_app_ins_history_cheating_cnt'
            , 'ft_app_ins_current_imei_alter_cnt'
            , 'ft_app_ins_history_imei_alter_cnt'
            , 'ft_app_ins_current_malicious_cnt'
            , 'ft_app_ins_history_malicious_cnt'
            , 'ft_dev_battery_use_30d_whole_nighttime_hours'
            , 'ft_dev_battery_use_30d_weekday_allday_hours'
            , 'ft_dev_battery_use_30d_weekday_worktime_hours'
            , 'ft_dev_battery_use_30d_weekday_closetime_hours'
            , 'ft_dev_battery_use_30d_weekday_nighttime_hours'
            , 'ft_dev_battery_use_30d_weekend_allday_hours'
            , 'ft_dev_battery_all_90d_whole_allday_hours'
            , 'ft_dev_battery_full_90d_whole_allday_hours'
            , 'ft_dev_battery_full_90d_weekday_allday_hours'
            , 'ft_dev_battery_full_90d_weekend_allday_hours'
            , 'ft_dev_battery_charge_90d_whole_allday_hours'
            , 'ft_dev_battery_charge_90d_weekday_allday_hours'
            , 'ft_dev_battery_charge_90d_weekend_allday_hours'
            , 'ft_dev_battery_use_90d_whole_allday_hours'
            , 'ft_dev_battery_use_90d_whole_worktime_hours'
            , 'ft_dev_battery_use_90d_whole_closetime_hours'
            , 'ft_dev_battery_use_90d_whole_nighttime_hours'
            , 'ft_dev_battery_use_90d_weekday_allday_hours'
            , 'ft_dev_battery_use_90d_weekday_worktime_hours'
            , 'ft_dev_battery_use_90d_weekday_closetime_hours'
            , 'ft_dev_battery_use_90d_weekday_nighttime_hours'
            , 'ft_dev_battery_use_90d_weekend_allday_hours'
            , 'ft_dev_ip_30d_whole_allday_cnt'
            , 'ft_dev_wifimac_30d_whole_allday_cnt'
            , 'ft_dev_wifimac_30d_whole_allday_days'
            , 'ft_dev_ip_30d_whole_worktime_cnt'
            , 'ft_dev_ip_30d_whole_closetime_cnt'
            , 'ft_dev_ip_30d_whole_nighttime_cnt'
            , 'ft_dev_ip_30d_weekday_allday_cnt'
            , 'ft_dev_ip_30d_weekday_worktime_cnt'
            , 'ft_dev_ip_30d_weekday_closetime_cnt'
            , 'ft_dev_ip_30d_weekday_nighttime_cnt'
            , 'ft_dev_ip_30d_weekend_allday_cnt'
            , 'ft_dev_wifimac_30d_whole_worktime_cnt'
            , 'ft_dev_wifimac_30d_whole_closetime_cnt'
            , 'ft_dev_wifimac_30d_whole_nighttime_cnt'
            , 'ft_dev_wifimac_30d_weekday_allday_cnt'
            , 'ft_dev_wifimac_30d_weekday_worktime_cnt'
            , 'ft_dev_wifimac_30d_weekday_closetime_cnt'
            , 'ft_dev_wifimac_30d_weekday_nighttime_cnt'
            , 'ft_dev_wifimac_30d_weekend_allday_cnt'
            , 'ft_dev_wifimac_30d_whole_worktime_days'
            , 'ft_dev_wifimac_30d_whole_closetime_days'
            , 'ft_dev_wifimac_30d_whole_nighttime_days'
            , 'ft_dev_wifimac_30d_weekday_allday_days'
            , 'ft_dev_wifimac_30d_weekday_worktime_days'
            , 'ft_dev_wifimac_30d_weekday_closetime_days'
            , 'ft_dev_wifimac_30d_weekday_nighttime_days'
            , 'ft_dev_wifimac_30d_weekend_allday_days'
            , 'ft_act_2week_rest_active_period_ls_hour0_4'
            , 'ft_act_2week_rest_active_period_ls_hour12_13'
            , 'ft_act_2week_rest_active_period_ls_hour13_18'
            , 'ft_act_2week_rest_active_period_ls_hour18_23'
            , 'ft_act_2week_rest_active_period_ls_hour5_7'
            , 'ft_act_2week_rest_active_period_ls_hour8_11'
            , 'ft_act_2week_work_active_period_ls_hour0_4'
            , 'ft_act_2week_work_active_period_ls_hour12_13'
            , 'ft_act_2week_work_active_period_ls_hour13_18'
            , 'ft_act_2week_work_active_period_ls_hour18_23'
            , 'ft_act_2week_work_active_period_ls_hour5_7'
            , 'ft_act_2week_work_active_period_ls_hour8_11'
            , 'ft_act_2week_rest_less_act_times'
            , 'ft_act_2week_rest_most_act_times'
            , 'ft_act_2week_rest_sum_act_times'
            , 'ft_act_2week_work_less_act_times'
            , 'ft_act_2week_work_most_act_times'
            , 'ft_act_2week_work_sum_act_times'
            , 'ft_app_act_30d_allcate_cnt'
            , 'ft_app_act_30d_allcate_times'
            , 'ft_app_act_30d_allcate_days'
            , 'ft_app_act_30d_fitness_cnt'
            , 'ft_app_act_30d_fitness_times'
            , 'ft_app_act_30d_fitness_days'
            , 'ft_app_act_30d_office_business_cnt'
            , 'ft_app_act_30d_office_business_times'
            , 'ft_app_act_30d_office_business_days'
            , 'ft_app_act_30d_video_cnt'
            , 'ft_app_act_30d_video_times'
            , 'ft_app_act_30d_video_days'
            , 'ft_app_act_30d_photo_cnt'
            , 'ft_app_act_30d_insurance_cnt'
            , 'ft_app_act_30d_insurance_times'
            , 'ft_app_act_30d_insurance_days'
            , 'ft_app_act_30d_credit_card_cnt'
            , 'ft_app_act_30d_credit_card_times'
            , 'ft_app_act_30d_credit_card_days'
            , 'ft_app_act_30d_debit_credit_cnt'
            , 'ft_app_act_30d_debit_credit_times'
            , 'ft_app_act_30d_debit_credit_days'
            , 'ft_app_act_30d_lottery_cnt'
            , 'ft_app_act_30d_lottery_times'
            , 'ft_app_act_30d_lottery_days'
            , 'ft_app_act_30d_invest_cnt'
            , 'ft_app_act_30d_invest_times'
            , 'ft_app_act_30d_invest_days'
            , 'ft_app_act_30d_payment_cnt'
            , 'ft_app_act_30d_payment_times'
            , 'ft_app_act_30d_payment_days'
            , 'ft_app_act_30d_money_management_cnt'
            , 'ft_app_act_30d_money_management_times'
            , 'ft_app_act_30d_money_management_days'
            , 'ft_app_act_30d_tally_cnt'
            , 'ft_app_act_30d_tally_times'
            , 'ft_app_act_30d_tally_days'
            , 'ft_app_act_30d_bank_cnt'
            , 'ft_app_act_30d_bank_times'
            , 'ft_app_act_30d_bank_days'
            , 'ft_app_act_30d_online_gamble_cnt'
            , 'ft_app_act_30d_online_gamble_times'
            , 'ft_app_act_30d_online_gamble_days'
            , 'ft_app_act_30d_micro_loan_cnt'
            , 'ft_app_act_30d_micro_loan_times'
            , 'ft_app_act_30d_micro_loan_days'
            , 'ft_app_act_30d_finance_cnt'
            , 'ft_app_act_30d_finance_times'
            , 'ft_app_act_30d_finance_days'
            , 'ft_finance_score_c'
            , 'ft_finance_score_d'
            , 'ft_finance_score_e'
            , 'ft_finance_score_f'
            , 'ft_app_ins_current_gpscheat_cnt'
            , 'ft_app_ins_history_gpscheat_cnt'
            , 'ft_app_ins_current_simulator_cnt'
            , 'ft_app_ins_history_simulator_cnt'
            , 'ft_app_act_30d_risk_cnt'
            , 'ft_app_act_30d_risk_times'
            , 'ft_app_act_30d_risk_days'
            , 'ft_app_act_30d_cheating_cnt'
            , 'ft_app_act_30d_cheating_times'
            , 'ft_app_act_30d_cheating_days'
            , 'ft_app_act_30d_imei_alter_cnt'
            , 'ft_app_act_30d_imei_alter_times'
            , 'ft_app_act_30d_imei_alter_days'
            , 'ft_app_act_30d_malicious_cnt'
            , 'ft_app_act_30d_malicious_times'
            , 'ft_app_act_30d_malicious_days'
            , 'ft_app_act_30d_gpscheat_cnt'
            , 'ft_app_act_30d_gpscheat_times'
            , 'ft_app_act_30d_gpscheat_days'
            , 'ft_app_act_30d_simulator_cnt'
            , 'ft_app_act_30d_simulator_times'
            , 'ft_app_act_30d_simulator_days'
            , 'ft_dev_storage'
            , 'ft_dev_free_storage'
            , 'ft_dev_storage_usage_ratio'
            , 'ft_gezhen_multi_loan_score'
            , 'ft_safe_score_c'
            , 'ft_safe_score_d'
            , 'ft_safe_score_e'
            , 'ft_safe_score_f'
            , 'ft_dev_battery_full_30d_whole_allday_hours'
            , 'ft_dev_battery_charge_30d_whole_allday_hours'
            , 'ft_dev_battery_use_30d_whole_allday_hours'
            , 'ft_dev_battery_all_30d_whole_allday_hours'
            , 'ft_dev_battery_full_30d_weekday_allday_hours'
            , 'ft_dev_battery_full_30d_weekend_allday_hours'
            , 'ft_dev_battery_charge_30d_weekday_allday_hours'
            , 'ft_dev_battery_charge_30d_weekend_allday_hours'
            , 'ft_dev_battery_use_30d_whole_worktime_hours'
            , 'ft_dev_battery_use_30d_whole_closetime_hours'
            , 'ft_app_ins_current_money_management_cnt'
            , 'ft_app_ins_history_money_management_cnt'
            , 'ft_app_ins_current_tally_cnt'
            , 'ft_app_ins_history_tally_cnt'
            , 'ft_app_ins_current_bank_cnt'
            , 'ft_app_ins_history_bank_cnt'
            , 'ft_app_ins_current_online_gamble_cnt'
            , 'ft_app_ins_history_online_gamble_cnt'
            , 'ft_app_ins_current_micro_loan_cnt'
            , 'ft_app_ins_history_micro_loan_cnt'
            , 'ft_app_ins_history_shortterm_loan'
            , 'ft_app_ins_current_shortterm_loan'
            , 'ft_app_act_30d_gamble_cnt'
            , 'ft_app_act_30d_gamble_times'
            , 'ft_app_act_30d_gamble_days'
            , 'ft_act_month_rest_active_period_ls_hour0_4'
            , 'ft_act_month_rest_active_period_ls_hour12_13'
            , 'ft_act_month_rest_active_period_ls_hour13_18'
            , 'ft_act_month_rest_active_period_ls_hour18_23'
            , 'ft_act_month_rest_active_period_ls_hour5_7'
            , 'ft_act_month_rest_active_period_ls_hour8_11'
            , 'ft_act_month_rest_less_act_times'
            , 'ft_act_month_rest_most_act_times'
            , 'ft_act_month_rest_sum_act_times'
            , 'ft_act_month_work_active_period_ls_hour0_4'
            , 'ft_act_month_work_active_period_ls_hour8_11'
            , 'ft_act_month_work_active_period_ls_hour12_13'
            , 'ft_act_month_work_active_period_ls_hour13_18'
            , 'ft_act_month_work_active_period_ls_hour18_23'
            , 'ft_act_month_work_active_period_ls_hour5_7'
            , 'ft_act_month_work_less_act_times'
            , 'ft_act_month_work_most_act_times'
            , 'ft_act_month_work_sum_act_times'
            , 'ft_app_ins_current_fitness_cnt'
            , 'ft_app_ins_history_fitness_cnt'
            , 'ft_app_ins_current_photo_cnt'
            , 'ft_app_ins_history_photo_cnt'
            , 'ft_app_ins_current_life_casual_cnt'
            , 'ft_app_ins_history_life_casual_cnt'
            , 'ft_app_ins_current_office_business_cnt'
            , 'ft_app_ins_history_office_business_cnt'
            , 'ft_app_ins_current_video_cnt'
            , 'ft_app_ins_history_video_cnt'
            , 'ft_app_ins_current_news_reading_cnt'
            , 'ft_app_ins_history_news_reading_cnt'
            , 'ft_app_ins_current_game_cnt'
            , 'ft_app_ins_history_game_cnt'
            , 'ft_app_ins_current_examination_cnt'
            , 'ft_app_ins_history_examination_cnt'
            , 'ft_app_ins_current_child_rearing_cnt'
            , 'ft_app_ins_history_child_rearin_cnt'
            , 'ft_app_ins_current_messaging_social_cnt'
            , 'ft_app_ins_history_messaging_social_cnt'
            , 'ft_app_ins_current_short_video_cnt'
            , 'ft_app_ins_history_short_video_cnt'
            , 'ft_app_ins_current_live_cnt'
            , 'ft_app_ins_history_live_cnt'
            , 'ft_act_cate_num'
            , 'ft_act_max_act_time'
            , 'ft_act_pkg_num'
            , 'ft_ins_cate_app_cnt_ls_100000'
            , 'ft_ins_cate_app_cnt_ls_120000'
            , 'ft_ins_cate_app_cnt_ls_130000'
            , 'ft_ins_cate_app_cnt_ls_140000'
            , 'ft_ins_cate_app_cnt_ls_150000'
            , 'ft_ins_cate_app_cnt_ls_160000'
            , 'ft_ins_cate_app_cnt_ls_170000'
            , 'ft_ins_cate_app_cnt_ls_190000'
            , 'ft_ins_cate_app_cnt_ls_200000'
            , 'ft_ins_cate_app_cnt_ls_210000'
            , 'ft_ins_cate_app_cnt_ls_220000'
            , 'ft_ins_cate_app_cnt_ls_230000'
            , 'ft_ins_cate_app_cnt_ls_240000'
            , 'ft_ins_cate_app_cnt_ls_250000'
            , 'ft_ins_cate_app_cnt_ls_270000'
            , 'ft_ins_cate_app_cnt_ls_290000'
            , 'ft_ins_cate_app_cnt_ls_300000'
            , 'ft_ins_cate_app_cnt_ls_900400'
            , 'ft_ins_cate_app_cnt_ls_900500'
            , 'ft_ins_cate_app_cnt_ls_900600'
            , 'ft_ins_cate_app_cnt_ls_900700'
            , 'ft_ins_category_cnt'
            , 'ft_ins_largest_cate_cnt'
            , 'ft_ins_pkg_cnt'
            , 'ft_lbs_city_stay_oneday_cnt'
            , 'ft_lbs_city_stay_twoday_cnt'
            , 'ft_stable_score_a'
            , 'ft_stable_score_c'
            , 'ft_stable_score_e'
            , 'ft_app_act_30d_debit_credit_cnt_avg'
            , 'ft_app_act_30d_debit_credit_cnt_diff_m_llm'
            , 'ft_app_act_30d_debit_credit_cnt_diff_m_lm'
            , 'ft_app_act_30d_debit_credit_cnt_std'
            , 'ft_app_act_30d_debit_credit_days_avg'
            , 'ft_app_act_30d_debit_credit_days_diff_m_llm'
            , 'ft_app_act_30d_debit_credit_days_diff_m_lm'
            , 'ft_app_act_30d_debit_credit_days_std'
            , 'ft_app_act_30d_debit_credit_times_avg'
            , 'ft_app_act_30d_debit_credit_times_diff_m_llm'
            , 'ft_app_act_30d_debit_credit_times_diff_m_lm'
            , 'ft_app_act_30d_debit_credit_times_std'
            , 'ft_app_act_30d_finance_cnt_avg'
            , 'ft_app_act_30d_finance_cnt_diff_m_llm'
            , 'ft_app_act_30d_finance_cnt_diff_m_lm'
            , 'ft_app_act_30d_finance_cnt_std'
            , 'ft_app_act_30d_finance_days_avg'
            , 'ft_app_act_30d_finance_days_diff_m_llm'
            , 'ft_app_act_30d_finance_days_diff_m_lm'
            , 'ft_app_act_30d_finance_days_std'
            , 'ft_app_act_30d_finance_times_avg'
            , 'ft_app_act_30d_finance_times_diff_m_llm'
            , 'ft_app_act_30d_finance_times_diff_m_lm'
            , 'ft_app_act_30d_finance_times_std'
            , 'ft_app_act_30d_micro_loan_cnt_avg'
            , 'ft_app_act_30d_micro_loan_cnt_diff_m_llm'
            , 'ft_app_act_30d_micro_loan_cnt_diff_m_lm'
            , 'ft_app_act_30d_micro_loan_cnt_std'
            , 'ft_app_act_30d_micro_loan_days_avg'
            , 'ft_app_act_30d_micro_loan_days_diff_m_llm'
            , 'ft_app_act_30d_micro_loan_days_diff_m_lm'
            , 'ft_app_act_30d_micro_loan_days_std'
            , 'ft_app_act_30d_micro_loan_times_avg'
            , 'ft_app_act_30d_micro_loan_times_diff_m_llm'
            , 'ft_app_act_30d_micro_loan_times_diff_m_lm'
            , 'ft_app_act_30d_micro_loan_times_std'
            , 'ft_app_ins_current_debit_credit_cnt_avg'
            , 'ft_app_ins_current_debit_credit_cnt_diff_m_llm'
            , 'ft_app_ins_current_debit_credit_cnt_diff_m_lm'
            , 'ft_app_ins_current_debit_credit_cnt_std'
            , 'ft_app_act_30d_market_days'
            , 'ft_app_ins_current_online_shopping_cnt'
            , 'ft_app_ins_history_online_shopping_cnt'
            , 'ft_app_ins_current_travel_cnt'
            , 'ft_app_ins_history_travel_cnt'
            , 'ft_app_ins_current_discounts_cnt'
            , 'ft_app_ins_history_discounts_cnt'
            , 'ft_app_ins_current_market_cnt'
            , 'ft_app_ins_history_market_cnt'
            , 'ft_lbs_home_cons_ls_high'
            , 'ft_lbs_home_cons_ls_low'
            , 'ft_lbs_home_cons_ls_middle'
            , 'ft_lbs_pwoi_all_mostoften_consume_high'
            , 'ft_lbs_pwoi_all_mostoften_consume_low'
            , 'ft_lbs_pwoi_all_mostoften_consume_middle'
            , 'ft_lbs_pwoi_all_often_consum_high'
            , 'ft_lbs_pwoi_all_often_consum_low'
            , 'ft_lbs_pwoi_all_often_consum_middle'
            , 'ft_lbs_work_cons_ls_high'
            , 'ft_lbs_work_cons_ls_low'
            , 'ft_lbs_work_cons_ls_middle'
            , 'ft_consumption_score_a'
            , 'ft_consumption_score_c'
            , 'ft_consumption_score_e'
            , 'ft_app_ins_current_finance_cnt'
            , 'ft_app_ins_history_finance_cnt'
            , 'ft_app_ins_current_gamble_cnt'
            , 'ft_app_ins_history_gamble_cnt'
            , 'ft_app_ins_current_insurance_cnt'
            , 'ft_app_ins_history_insurance_cnt'
            , 'ft_app_ins_current_credit_card_cnt'
            , 'ft_app_ins_history_credit_card_cnt'
            , 'ft_app_ins_current_debit_credit_cnt'
            , 'ft_app_ins_history_debit_credit_cnt'
            , 'ft_app_ins_current_lottery_cnt'
            , 'ft_app_ins_history_lottery_cnt'
            , 'ft_app_ins_current_invest_cnt'
            , 'ft_app_ins_history_invest_cnt'
            , 'ft_app_ins_current_payment_cnt'
            , 'ft_app_ins_history_payment_cnt'
            , 'ft_app_act_30d_photo_times'
            , 'ft_app_act_30d_photo_days'
            , 'ft_app_act_30d_news_reading_cnt'
            , 'ft_app_act_30d_news_reading_times'
            , 'ft_app_act_30d_news_reading_days'
            , 'ft_app_act_30d_game_cnt'
            , 'ft_app_act_30d_game_times'
            , 'ft_app_act_30d_game_days'
            , 'ft_app_act_30d_life_casual_cnt'
            , 'ft_app_act_30d_life_casual_times'
            , 'ft_app_act_30d_life_casual_days'
            , 'ft_app_act_30d_examination_cnt'
            , 'ft_app_act_30d_examination_times'
            , 'ft_app_act_30d_examination_days'
            , 'ft_app_act_30d_child_rearin_cnt'
            , 'ft_app_act_30d_child_rearin_times'
            , 'ft_app_act_30d_child_rearin_days'
            , 'ft_app_act_30d_messaging_social_cnt'
            , 'ft_app_act_30d_messaging_social_times'
            , 'ft_app_act_30d_messaging_social_days'
            , 'ft_app_act_30d_short_video_cnt'
            , 'ft_app_act_30d_short_video_times'
            , 'ft_app_act_30d_short_video_days'
            , 'ft_app_act_30d_live_cnt'
            , 'ft_app_act_30d_live_times'
            , 'ft_app_act_30d_live_days'
            , 'ft_asset_score_a'
            , 'ft_asset_score_c'
            , 'ft_asset_score_e'
            , 'ft_app_act_30d_online_shopping_cnt'
            , 'ft_app_act_30d_online_shopping_times'
            , 'ft_app_act_30d_online_shopping_days'
            , 'ft_app_act_30d_travel_cnt'
            , 'ft_app_act_30d_travel_times'
            , 'ft_app_act_30d_travel_days'
            , 'ft_app_act_30d_discounts_cnt'
            , 'ft_app_act_30d_discounts_times'
            , 'ft_app_act_30d_discounts_days'
            , 'ft_app_act_30d_market_cnt'
            , 'ft_app_act_30d_market_times']
        self.blacklist_var = ["ft_gz_black_list", "ft_gz_grey_list"]
        self.class_var = ['ft_act_2week_rest_less_act_period_ls_hour12_13'
            , 'ft_act_2week_rest_less_act_period_ls_hour5_7'
            , 'ft_cross_lbs_4wr_act_cnt_app_4w_unins'
            , 'ft_cross_lbs_staycity_tag_021700'
            , 'ft_cross_most_act_cate_lbs_staycity'
            , 'ft_cross_tag_travel_macao_cnt'
            , 'ft_dev_all_nighter'
            , 'ft_dev_deprecia_price'
            , 'ft_dev_gid2imei_2w_cnt'
            , 'ft_dev_gid2imsi_2w_cnt'
            , 'ft_dev_gid2mac_2w_cnt'
            , 'ft_dev_market_price'
            , 'ft_dev_market_year'
            , 'ft_dev_root'
            , 'ft_dev_root_blacklist'
            , 'ft_dev_trust'
            , 'ft_dev_virtual'
            , 'ft_dev_xposed'
            , 'ft_gezhen_multi_loan_level'
            , 'ft_gz_sensitive_area'
            , 'ft_lbs_entertainment_a_12w_days'
            , 'ft_lbs_entertainment_b_12w_days'
            , 'ft_lbs_entertainment_c_12w_days'
            , 'ft_lbs_entertainment_d_12w_days'
            , 'ft_lbs_entertainment_e_12w_days'
            , 'ft_lbs_entertainment_f_12w_days'
            , 'ft_lbs_hospital_12w_days'
            , 'ft_social_blackfriends'
            , 'ft_social_blackfriends_group'
            , 'ft_tag_banking'
            , 'ft_tag_bas_fraudster'
            , 'ft_tag_bookkeeping'
            , 'ft_tag_credit_card'
            , 'ft_tag_fitness'
            , 'ft_tag_health'
            , 'ft_tag_language_learing'
            , 'ft_tag_lottery'
            , 'ft_tag_online_learing'
            , 'ft_act_2week_rest_most_act_period_ls_hour13_18'
            , 'ft_act_2week_rest_most_act_period_ls_hour18_23'
            , 'ft_act_2week_rest_most_act_period_ls_hour8_11'
            , 'ft_act_2week_work_most_act_period_ls_hour13_18'
            , 'ft_act_2week_work_most_act_period_ls_hour18_23'
            , 'ft_act_2week_work_most_act_period_ls_hour8_11'
            , 'ft_act_cate_ls_100000'
            , 'ft_act_cate_ls_120000'
            , 'ft_act_cate_ls_140000'
            , 'ft_act_cate_ls_150000'
            , 'ft_act_cate_ls_160000'
            , 'ft_act_cate_ls_200000'
            , 'ft_act_cate_ls_210000'
            , 'ft_act_cate_ls_220000'
            , 'ft_act_cate_ls_230000'
            , 'ft_act_cate_ls_240000'
            , 'ft_act_cate_ls_270000'
            , 'ft_act_cate_ls_300000'
            , 'ft_act_month_rest_less_act_period_ls_hour0_4'
            , 'ft_act_month_rest_less_act_period_ls_hour12_13'
            , 'ft_act_month_rest_less_act_period_ls_hour5_7'
            , 'ft_act_month_rest_most_act_period_ls_hour13_18'
            , 'ft_act_month_rest_most_act_period_ls_hour18_23'
            , 'ft_act_month_rest_most_act_period_ls_hour8_11'
            , 'ft_act_month_work_less_act_period_ls_hour0_4'
            , 'ft_act_month_work_less_act_period_ls_hour12_13'
            , 'ft_act_month_work_less_act_period_ls_hour5_7'
            , 'ft_act_month_work_less_act_period_ls_hour8_11'
            , 'ft_act_month_work_most_act_period_ls_hour13_18'
            , 'ft_act_month_work_most_act_period_ls_hour18_23'
            , 'ft_act_month_work_most_act_period_ls_hour8_11'
            , 'ft_andr_t1_most_interest_ls_021000'
            , 'ft_app_install_category_160000'
            , 'ft_app_uninstall_category_160000'
            , 'ft_asset_score_b'
            , 'ft_asset_score_d'
            , 'ft_asset_score_f'
            , 'ft_consumption_score_b'
            , 'ft_consumption_score_d'
            , 'ft_consumption_score_f'
            , 'ft_cross_act_160000_tag_0844'
            , 'ft_cross_blackfriends_act_2w_cat_16000'
            , 'ft_cross_blackfriends_tag_p2p'
            , 'ft_cross_ins_credit_ins_largest_cate_cnt'
            , 'ft_cross_ins_loan_tag_business_travel'
            , 'ft_cross_ins_loan_tag_p2p'
            , 'ft_cross_ins_wifilocating_tag_credit_card'
            , 'ft_cross_lbs_staycity_ins_160000'
            , 'ft_cross_tag_city_level1_housing_price'
            , 'ft_cross_tag_city_level1_price'
            , 'ft_cross_tag_city_level2_housing_price'
            , 'ft_cross_tag_city_level2_price'
            , 'ft_cross_tag_city_level3_housing_price'
            , 'ft_cross_tag_city_level3_price'
            , 'ft_cross_tag_city_level4_housing_price'
            , 'ft_cross_tag_city_level4_price'
            , 'ft_cross_tag_college_student_tag_p2p'
            , 'ft_cross_tag_p2p_tag_lifeservice'
            , 'ft_cross_unins_app_ins_160000'
            , 'ft_dev_timezone'
            , 'ft_gz_car_price'
            , 'ft_gz_consumption_capacity'
            , 'ft_gz_housing_price_homeaddr_cnt'
            , 'ft_gz_income_1'
            , 'ft_ins_app_install_category_160000'
            , 'ft_ins_app_uninstall_category_120000'
            , 'ft_ins_app_uninstall_category_140000'
            , 'ft_ins_app_uninstall_category_150000'
            , 'ft_ins_app_uninstall_category_160000'
            , 'ft_ins_app_uninstall_category_190000'
            , 'ft_ins_app_uninstall_category_200000'
            , 'ft_ins_app_uninstall_category_210000'
            , 'ft_ins_app_uninstall_category_230000'
            , 'ft_ins_app_uninstall_category_240000'
            , 'ft_ins_app_uninstall_category_270000'
            , 'ft_ins_app_uninstall_category_300000'
            , 'ft_ins_app_uninstall_category_900400'
            , 'ft_ins_largest_cate_ls_120000'
            , 'ft_ins_largest_cate_ls_140000'
            , 'ft_ins_largest_cate_ls_150000'
            , 'ft_ins_largest_cate_ls_190000'
            , 'ft_ins_largest_cate_ls_200000'
            , 'ft_ins_largest_cate_ls_210000'
            , 'ft_ins_largest_cate_ls_270000'
            , 'ft_lbs_airport_4w_times'
            , 'ft_lbs_dis_label'
            , 'ft_lbs_family_12w_change_cnt'
            , 'ft_lbs_family_12w_cnt'
            , 'ft_lbs_geo7_unusual'
            , 'ft_lbs_hotel_4w_days'
            , 'ft_lbs_hotel_4w_level'
            , 'ft_lbs_house_price'
            , 'ft_lbs_property_fee'
            , 'ft_lbs_residence_stability'
            , 'ft_lbs_same_night_stay_wifimac_cnt'
            , 'ft_lbs_stable_stops'
            , 'ft_lbs_trainstation_4w_times'
            , 'ft_lbs_work_12w_change_cnt'
            , 'ft_lbs_work_12w_cnt'
            , 'ft_lbs_workplace_stability'
            , 'ft_most_act_cate_ls_160000'
            , 'ft_stable_imsi_active_days'
            , 'ft_stable_imsi_changes'
            , 'ft_stable_imsi_used_days'
            , 'ft_stable_score_b'
            , 'ft_stable_score_d'
            , 'ft_stable_score_f'
            , 'ft_tag_age'
            , 'ft_tag_asset_management'
            , 'ft_tag_bourgeois'
            , 'ft_tag_car_ownership'
            , 'ft_tag_city_tier'
            , 'ft_tag_consumption_level'
            , 'ft_tag_consumption_range'
            , 'ft_tag_education'
            , 'ft_tag_gender'
            , 'ft_tag_hotel'
            , 'ft_tag_industry_ad'
            , 'ft_tag_industry_financial'
            , 'ft_tag_industry_it'
            , 'ft_tag_marrige'
            , 'ft_tag_occupation_entrepreneur'
            , 'ft_tag_occupation_it'
            , 'ft_tag_occupation_medical'
            , 'ft_tag_occupation_office'
            , 'ft_tag_occupation_teacher'
            , 'ft_tag_p2p'
            , 'ft_tag_parent_0to2y'
            , 'ft_tag_parent_3to6y'
            , 'ft_tag_parent_k1tok6'
            , 'ft_tag_parent_k7tok12'
            , 'ft_tag_primary_purchaser'
            , 'ft_tag_property_owner'
            , 'ft_tag_purchaser'
            , 'ft_tag_residence'
            , 'ft_tag_shopping'
            , 'ft_tag_stock'
            , 'ft_tag_travel'
            , 'mother'
            , 'parent'
            , 'ft_dev_phone_brand'
            , 'ft_dev_phone_model'
            , 'ft_app_act_30d_debit_credit_cnt_change'
            , 'ft_app_act_30d_debit_credit_days_change'
            , 'ft_app_act_30d_debit_credit_times_change'
            , 'ft_app_act_30d_finance_cnt_change'
            , 'ft_app_act_30d_finance_days_change'
            , 'ft_app_act_30d_finance_times_change'
            , 'ft_app_act_30d_micro_loan_cnt_change'
            , 'ft_app_act_30d_micro_loan_days_change'
            , 'ft_app_act_30d_micro_loan_times_change'
            , 'ft_app_ins_current_debit_credit_cnt_change'
            , 'ft_app_ins_current_finance_cnt_change'
            , 'ft_app_ins_current_micro_loan_cnt_change'
            , 'ft_tag_city'
            , 'ft_tag_province']
        self.cus_df = cus_df
        self.cus_df["month"] = self.cus_df["create_date"].map(lambda x: str(x)[:6])
        self.output_path = output_path

    # 对大盘数据进行月份比例加减乘除
    def get_gz_result(self, month_feature="month"):
        month_dict = self.cus_df[month_feature].value_counts().to_dict()
        month_dict = {k: round(v / len(self.cus_df), 2) for k, v in month_dict.items()}
        monthList = ",".join([str(k) for k in month_dict])
        sql = "select * from chenyw_yzx.autolabel_map where month in ({})".format(monthList)
        df = spark.sql(sql)
        self.gz_df = df.toPandas()
        self.gz_df["tag"] = self.gz_df.tag.astype(str)
        self.gz_df["cnt"] = self.gz_df.cnt.astype(int)
        self.bin_result = pd.DataFrame()

        bin_ratio = self.gz_df['cnt'].groupby(by=[self.gz_df['month']]).agg(['sum'])
        bin_ratio['month'] = bin_ratio.index
        bin_ratio['total_cnt'] = bin_ratio['sum'].map(lambda x: x / 605)
        bin_ratio.index = [i for i in range(len(bin_ratio))]
        self.gz_df = pd.merge(self.gz_df, bin_ratio[['month', 'total_cnt']], how='left', on='month')
        self.gz_df['ratio'] = self.gz_df.apply(lambda x: round(month_dict[str(x.month)] * x.cnt / x.total_cnt, 4),
                                               axis=1)
        gz_bin_result = pd.pivot_table(self.gz_df, index=["feature", "tag"], values=["cnt", "ratio"], aggfunc=[sum])
        gz_bin_result.columns = ['gz_cnt', 'gz_ratio']
        gz_bin_result['feature'] = [i[0] for i in gz_bin_result.index]
        gz_bin_result['tag'] = [i[1] for i in gz_bin_result.index]
        gz_bin_result.index = [i for i in range(gz_bin_result.shape[0])]
        self.bin_result = gz_bin_result[['feature', 'tag', 'gz_cnt', 'gz_ratio']]

    # 连续变量需要引入配置文件
    def get_continuous_tag(self, feature, value):
        tmp_df = self.config_df[self.config_df["feature"] == feature]
        min_value = float(tmp_df.max_value.min())
        if value <= min_value:
            return '0'
        max_value = float(tmp_df.min_value.max())
        if value > max_value:
            return '4'
        interval = (max_value - min_value) / 3
        tag = np.ceil((value - min_value) / interval)
        return tag

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    # 对甲方爸爸数据进行处理
    def get_cus_result(self):
        for c in self.blacklist_var:
            self.cus_df[c] = self.cus_df[c].map(lambda x: 'blank' if x is None or x != x or x == '' else 1)
        for c in self.class_var:
            self.cus_df[c] = self.cus_df[c].map(lambda x: 'blank' if x is None or x != x or x == '' else x)
            # 甲方爸爸的离散变量tag值不一定存在于2KW中
            cus_tag_list = self.cus_df[c].unique().tolist()
            cus_tag_list = [float(i) for i in cus_tag_list if self.is_number(i)] + [i for i in cus_tag_list if not self.is_number(i)]
            cus_tag_list = [str(int(i)) for i in cus_tag_list if type(i) == float and float(i) == int(i)] + \
                           [str(i) for i in cus_tag_list if type(i) == float and float(i) != int(i)] + \
                           [str(i) for i in cus_tag_list if type(i) != float]
            gz_tag_list = self.gz_df[self.gz_df.feature == c].tag.tolist()
            delta_tag = list(set(cus_tag_list) - set(gz_tag_list))
            if len(delta_tag) == 0:
                continue
            delta_dict = {
                "feature": [c]*len(delta_tag),
                "tag": delta_tag,
                "gz_cnt": [0]*len(delta_tag),
                "gz_ratio": [0]*len(delta_tag)
            }
            self.bin_result = pd.concat([self.bin_result, pd.DataFrame(data=delta_dict,columns=['feature', 'tag', 'gz_cnt','gz_ratio'])])
        for c in self.continuous_var:
            tmp = self.config_df[self.config_df.feature == c]
            cut_points = sorted(list(set(tmp.min_value.to_list() + tmp.max_value.to_list())))
            self.cus_df[c] = pd.cut(self.cus_df[c], cut_points, labels=['0', '1', '2', '3', '4'])
            self.cus_df[c] = self.cus_df[c].cat.add_categories(['blank'])
            self.cus_df[c].fillna("blank", inplace=True)

        self.bin_result['cus_cnt'] = self.bin_result.apply(
            lambda x: len(self.cus_df[self.cus_df[x.feature] == str(x.tag)]), axis=1)
        self.bin_result['gz_cnt'] = self.bin_result['gz_cnt'].astype(int)
        self.bin_result['gz_ratio'] = self.bin_result['gz_ratio'].astype(float)
        self.bin_result = self.bin_result[(self.bin_result.gz_cnt > 0) | (self.bin_result.cus_cnt > 0)]
        self.bin_result['cus_ratio'] = self.bin_result["cus_cnt"].map(lambda x: round(x / len(self.cus_df), 2))
        self.bin_result['delta_ratio'] = self.bin_result.apply(
            lambda x: float("inf") if x["gz_cnt"] == 0 or x["gz_ratio"] == 0 else round(
                (x["cus_ratio"] - x["gz_ratio"]) / x["gz_ratio"], 4), axis=1)

    # 计算WOE
    def get_woe_iv(self):
        self.bin_result['woe'] = self.bin_result.apply(
            lambda x: float("-inf") if x["cus_ratio"] == 0 else float("inf") if x["gz_ratio"] == 0 else round(
                math.log(x["cus_ratio"] / x["gz_ratio"]), 4), axis=1)
        self.bin_result["iv"] = self.bin_result.apply(lambda x: (x.cus_ratio - x.gz_ratio) * x.woe, axis=1)
        self.bin_result = pd.concat([self.bin_result[~self.bin_result.iv.isin(["inf", "-inf"])].sort_values(by='iv', ascending=False),
                                self.bin_result[self.bin_result.iv.isin(["inf", "-inf"])]])
        self.bin_result = self.bin_result.sort_values(by='iv', ascending=False)
        miss_df = pd.read_csv("/Users/cai/Desktop/pythonProjects/github_FoolishCai/GeneralTools/configs/import_miss.txt", sep=',')
        self.bin_result = pd.merge(self.bin_result, miss_df[['feature', 'chinese_name']], how='left', on='feature')
        self.bin_result = self.bin_result[["feature", "chinese_name", "tag", "gz_cnt", "gz_ratio", "cus_cnt", "cus_ratio", "delta_ratio", "woe", "iv"]]
        self.bin_result.columns = ["特征", "中文名称", "属性", "随机样本数量", "随机样本占比", "客户样本数量", "客户样本占比", "差异比", "WOE", "IV"]

    def main(self):
        self.get_gz_result(month_feature='')
        self.get_cus_result()
        self.get_woe_iv()
        self.bin_result.to_excel(self.output_path, index=None)




cus_df = pd.read_csv(
    "/Users/cai/Desktop/pythonProjects/local_pyProjects/WorkProjects/AutoLabel/config_files/cus_df.csv", sep=',')
cus_df.columns = [i[i.index(".")+1:] for i in cus_df.columns]
gz_data = pd.read_csv(
    "/Users/cai/Desktop/pythonProjects/local_pyProjects/WorkProjects/AutoLabel/config_files/gz_data_reault.csv",
    sep=',')
gz_data.columns = [i[i.index(".")+1:] for i in gz_data.columns]
output_path = "/Users/cai/Desktop/pythonProjects/local_pyProjects/WorkProjects/AutoLabel/config_files/result.xlsx"
lo = LabelOutput(cus_df, gz_data, output_path)
