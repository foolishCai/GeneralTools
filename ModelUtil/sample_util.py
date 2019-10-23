# -*- coding:utf-8 -*-

'''
Created date: 2019-05-16

@author: Cai

note: 各种采样方法
'''


from sklearn import metrics
from collections import Counter

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, TomekLinks, EditedNearestNeighbours

from ToolBars.configs import log

def model_resampling_pipeline(Xtrain, Xtest, y_train, y_test, model):
    '''
    以原始数据的模型评估作为基准，判定重采样是否需要，采样基于imblear
    :param Xtrain: 训练数据
    :param Xtest: 测试数据
    :param y_train:训练数据
    :param y_test: 测试数据
    :param model:  模型
    :return: 给回采样结果以后的各个评估指标
    '''

    results = {
        'ordinary': {},
        'class_weight': {},
        'oversample': {},
        'undersample': {}
    }

    # step1.非平衡样本
    model.fit(Xtrain, y_train)
    predictions = model.predict(Xtest)
    accuracy = metrics.accuracy_score(y_test, predictions)
    predicion, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
    fpr, tpr, thresholds = metrics.roc_auc_score(y_test, predictions)
    auc = metrics.auc(fpr, tpr)
    ks_value = abs(fpr - tpr).max()

    results['ordinary'] = {
        'accuracy': accuracy,
        'predicion': predicion,
        'recall': recall,
        'fscore': fscore,
        'no_occurance': support,
        'predictions_cnt': Counter(predictions),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'auc': auc, 'ks': ks_value
    }

    # step2.给类别设定权重
    if 'class.weight' in model.get_params().keys():
        model.set_params(class_weight = 'balanced')
        model.fit(Xtrain, y_train)
        predictions = model.predict(Xtest)
        accuracy = metrics.accuracy_score(y_test, predictions)
        predicion, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        fpr, tpr, thresholds = metrics.roc_auc_score(y_test, predictions)
        auc = metrics.auc(fpr, tpr)
        ks_value = abs(fpr - tpr).max()

        results['class_weight'] = {
            'accuracy': accuracy,
            'predicion': predicion,
            'recall': recall,
            'fscore': fscore,
            'no_occurance': support,
            'predictions_cnt': Counter(predictions),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'auc': auc, 'ks': ks_value
        }

    # step3.过采样(OVERSAMPLING TECHNIQUES)
    log.info('*'*15 + '过采样' + '*'*15)
    tequniques = [RandomOverSampler(),
                  SMOTE(),
                  ADASYN()]
    log.info('Before resampling: {}'.format(sorted(Counter(y_train).items())))

    for sampler in tequniques:
        teq = sampler.__class__.__name__
        log.info('过采样 - {}'.format(teq))
        Xresampled, y_resampled = sampler.fit_sample(Xtrain, y_train)
        log.info('After resampling: {}'.format(sorted(Counter(y_resampled).items())))

        model.fit(Xresampled, y_resampled)
        predictions = model.predict(Xtest)
        accuracy = metrics.accuracy_score(y_test, predictions)
        predicion, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        fpr, tpr, thresholds = metrics.roc_auc_score(y_test, predictions)
        auc = metrics.auc(fpr, tpr)
        ks_value = abs(fpr - tpr).max()

        results['oversample'][teq] = {
            'accuracy': accuracy,
            'predicion': predicion,
            'recall': recall,
            'fscore': fscore,
            'no_occurance': support,
            'predictions_cnt': Counter(predictions),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'auc': auc, 'ks': ks_value
        }

    # step4.欠采样
    log.info('*' * 15 + '欠采样' + '*' * 15)
    tequniques = [RandomOverSampler(),
                  NearMiss(version=1),
                  NearMiss(version=2),
                  TomekLinks(),
                  EditedNearestNeighbours()
            ]
    log.info('Before resampling: {}'.format(sorted(Counter(y_train).items())))

    for sampler in tequniques:
        teq = sampler.__class__.__name__
        log.info('过采样 - {}'.format(teq))
        Xresampled, y_resampled = sampler.fit_sample(Xtrain, y_train)
        log.info('After resampling: {}'.format(sorted(Counter(y_resampled).items())))

        model.fit(Xresampled, y_resampled)
        predictions = model.predict(Xtest)
        accuracy = metrics.accuracy_score(y_test, predictions)
        predicion, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, predictions)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, predictions).ravel()
        fpr, tpr, thresholds = metrics.roc_auc_score(y_test, predictions)
        auc = metrics.auc(fpr, tpr)
        ks_value = abs(fpr - tpr).max()

        results['undersample'][teq] = {
            'accuracy': accuracy,
            'predicion': predicion,
            'recall': recall,
            'fscore': fscore,
            'no_occurance': support,
            'predictions_cnt': Counter(predictions),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'auc': auc, 'ks': ks_value
        }

    return results