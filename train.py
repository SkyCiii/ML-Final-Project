import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train_no_one_hot.csv')
train_label = pd.read_csv('train_label.csv')
train_adr = pd.read_csv('train_adr.csv')
train_orig = pd.read_csv('train_orig.csv')

x = train_data.drop(labels=['ID', 'is_canceled', 'reservation_status', 'reservation_status_date'], axis=1)
adr = train_adr['adr'].values

test_data = pd.read_csv('test_no_one_hot.csv')
test_label = pd.read_csv('test_nolabel.csv')

x_test = test_data.drop(labels='ID', axis=1)

def is_canceled():
    train_is_canceled = pd.read_csv('train_is_canceled.csv')
    train_is_canceled = train_is_canceled.drop(labels=['ID', 'reservation_status', 'adr', 'reservation_status_date'], axis=1)
    is_canceled_label = pd.read_csv('is_canceled_label.csv')
    is_canceled_label = is_canceled_label['is_canceled'].values
    test_orig = pd.read_csv('test_orig.csv')

    index = test_orig.index.values
    test_with_is_canceled = pd.read_csv('test_with_is_canceled.csv')
    test_with_is_canceled = test_with_is_canceled.iloc[index]
    index = test_with_is_canceled[test_with_is_canceled['is_canceled'] != 1].index.values
    print(index)
    x_test_is_canceled = x_test.iloc[index]
    x_test_hong = test_with_is_canceled.iloc[index]
    test_orig_is_canceled = test_orig.iloc[index]

    # # is_canceled_x_train, is_canceled_x_valid, is_canceled_y_train, is_canceled_y_valid = train_test_split(train_is_canceled, is_canceled_label, test_size=0.2, shuffle=False)
    # clf = SVC()
    # # clf.fit(is_canceled_x_train, is_canceled_y_train)
    # # is_canceled_y_predict = clf.predict(is_canceled_x_valid)
    # # E_val_is_canceled = sum(abs(is_canceled_y_predict - is_canceled_y_valid))/len(is_canceled_y_valid)
    # clf.fit(train_is_canceled, is_canceled_label)
    # is_canceled_y = clf.predict(x_test)
    # x_test['is_canceled'] = is_canceled_y
    # test_orig['is_canceled'] = is_canceled_y
    # x_test_is_canceled = x_test[x_test['is_canceled'] != 1]
    # test_orig_is_canceled = test_orig[test_orig['is_canceled'] != 1]
    x_test_is_canceled.to_csv('test_is_canceled.csv', index=False)
    x_test_hong.to_csv('x_test_hong.csv', index=False)
    test_orig_is_canceled.to_csv('test_orig_is_canceled.csv', index=False)
    # pd.DataFrame(is_canceled_y).to_csv('predict.csv')
    # print(E_val_is_canceled)

    return None

# is_canceled()

def validation():
    print('------------------------------------ Validation ------------------------------------')
    
    x_train, x_valid, adr_train, adr_valid = train_test_split(x, adr, test_size=13442, shuffle=False)

    clf = SVR(C=1)
    clf.fit(x_train, adr_train)

    adr_predict = clf.predict(x_valid)
    E_val_adr = sum(abs(adr_predict - adr_valid))/len(adr_predict)

    x_valid['adr_predict'] = adr_predict
    valid_days = train_orig['stays_in_week_nights'].iloc[43684:].values + train_orig['stays_in_weekend_nights'].iloc[43684:].values
    x_valid['revenue'] = valid_days * x_valid['adr_predict'].values
    x_valid_groupby = x_valid.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
    print(x_valid_groupby['revenue'], len(x_valid_groupby['revenue']))
    x_valid = x_valid_groupby.reset_index()
    y_predict = np.floor(x_valid['revenue'].values/10000)
    y_valid = train_label['label'].iloc[489:].values
    E_val_label = sum(abs(y_predict - y_valid))/len(y_predict)

    return E_val_adr, E_val_label

# E_val_adr, E_val_label = validation()

def main():
    x_test_is_canceled = pd.read_csv('test_is_canceled.csv')
    x_test_hong = pd.read_csv('x_test_hong.csv')
    # x_test_is_canceled = x_test_is_canceled.drop('is_canceled', axis=1)
    test_orig_is_canceled = pd.read_csv('test_orig_is_canceled.csv')

    clf = SVR(C=200)
    clf.fit(x, adr)

    # adr_in = clf.predict(x)
    # E_in = sum(abs(adr_in - adr))/len(adr)
    # print('E_in: ', E_in, 'score_in: ', clf.score(x, adr))
    
    adr_predict = clf.predict(x_test_hong)
    print(adr_predict)
    days_test = test_orig_is_canceled['stays_in_week_nights'].values + test_orig_is_canceled['stays_in_weekend_nights'].values
    x_test_is_canceled['revenue'] = days_test * adr_predict
    print(x_test_is_canceled['revenue'])
    # x_test_is_canceled = x_test_is_canceled[x_test_is_canceled['revenue'] > 0]
    x_test_is_canceled_groupby = x_test_is_canceled.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
    print(x_test_is_canceled_groupby)
    x_test = x_test_is_canceled_groupby.reset_index()
    y_test = np.floor(x_test['revenue'].values/10000)

    print(y_test)
    test_label['label'] = y_test
    test_label.to_csv('test_label_'+str(200)+'.csv', index=False)

    return y_test

y_test = main()
