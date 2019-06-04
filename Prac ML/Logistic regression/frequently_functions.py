import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd


def get_accuracy_test(predict, test):
    return ((predict - test) == 0).astype(int).sum() / test.shape[0]


def get_train_and_test(minDf):
    nameTrain = 'news_train.json'
    nameTest = 'news_test.json'

    data_train = pd.read_json(nameTrain)
    all_texts = list()
    y_train = list()
    for i, one_text in enumerate(data_train['text'].values):
        if data_train.loc[i].sentiment == 'positive':
            y_train.append(1)
        elif data_train.loc[i].sentiment == 'negative':
            y_train.append(-1)
        else:
            continue
        one_text = str.lower(one_text)
        one_text = re.sub(r'[^\w]', ' ', one_text)
        all_texts.append(one_text)

    vectorizer = CountVectorizer(min_df=minDf)
    x_train = vectorizer.fit_transform(all_texts)
    y_train = np.array(y_train)

    data_test = pd.read_json(nameTest)
    all_texts = list()
    y_test = list()
    for i, one_text in enumerate(data_test['text'].values):
        if data_test.loc[i].sentiment == 'positive':
            y_test.append(1)
        elif data_test.loc[i].sentiment == 'negative':
            y_test.append(-1)
        else:
            continue
        one_text = str.lower(one_text)
        one_text = re.sub(r'[^\w]', ' ', one_text)
        all_texts.append(one_text)

    x_test = vectorizer.transform(all_texts)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


def get_train_and_test_3_classes(minDf):
    nameTrain = 'news_train.json'
    nameTest = 'news_test.json'

    data_train = pd.read_json(nameTrain)
    all_texts = list()
    y_train = list()
    for i, one_text in enumerate(data_train['text'].values):
        if data_train.loc[i].sentiment == 'positive':
            y_train.append(0)
        elif data_train.loc[i].sentiment == 'neutral':
            y_train.append(1)
        elif data_train.loc[i].sentiment == 'negative':
            y_train.append(2)
        else:
            raise TypeError
        one_text = str.lower(one_text)
        one_text = re.sub(r'[^\w]', ' ', one_text)
        all_texts.append(one_text)

    vectorizer = CountVectorizer(min_df=minDf)
    x_train = vectorizer.fit_transform(all_texts)
    y_train = np.array(y_train)

    data_test = pd.read_json(nameTest)
    all_texts = list()
    y_test = list()
    for i, one_text in enumerate(data_test['text'].values):
        if data_train.loc[i].sentiment == 'positive':
            y_test.append(0)
        elif data_train.loc[i].sentiment == 'neutral':
            y_test.append(1)
        elif data_train.loc[i].sentiment == 'negative':
            y_test.append(2)
        else:
            raise TypeError
        one_text = str.lower(one_text)
        one_text = re.sub(r'[^\w]', ' ', one_text)
        all_texts.append(one_text)

    x_test = vectorizer.transform(all_texts)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test