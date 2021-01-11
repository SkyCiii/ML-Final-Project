import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from preprocessing import preprocessing

train_data_orig = pd.read_csv('train.csv')
test_data_orig = pd.read_csv('test.csv')
train_label = pd.read_csv('train_label.csv')
test_label = pd.read_csv('test_nolabel.csv')

train_data, test_data = preprocessing(train_data_orig, test_data_orig)
test_data_orig = test_data_orig.iloc[test_data.index.values]

train_data_is_canceled = train_data[train_data['is_canceled'] != 1]
train_data_orig_is_canceled = train_data_orig.iloc[train_data_is_canceled.index.values]
train_data_drop_is_canceled = train_data_is_canceled.drop(labels=['ID', 'is_canceled', 'reservation_status', 'reservation_status_date'], axis=1)
train_adr = train_data_drop_is_canceled.pop('adr')

train_data_drop_adr = train_data.drop(labels=['ID', 'adr', 'reservation_status', 'reservation_status_date'], axis=1)

def is_canceled(train_data, test_data):
    print('------------------------------------ Is Canceled ------------------------------------')
    
    is_canceled_label = train_data.pop('is_canceled')
    test_data = test_data.drop(labels='ID', axis=1)

    clf = GradientBoostingClassifier()
    # is_canceled_x_train, is_canceled_x_valid, is_canceled_y_train, is_canceled_y_valid = train_test_split(train_data, is_canceled_label, test_size=0.2, shuffle=False)
    # clf.fit(is_canceled_x_train, is_canceled_y_train)
    # is_canceled_y_predict = clf.predict(is_canceled_x_valid)
    # E_val_is_canceled = sum(abs(is_canceled_y_predict - is_canceled_y_valid))/len(is_canceled_y_valid)
    clf.fit(train_data, is_canceled_label)
    is_canceled_y = clf.predict(test_data)
    print(test_data, test_data_orig)
    test_data['is_canceled'] = is_canceled_y
    test_data_orig['is_canceled'] = is_canceled_y
    test_data_is_canceled = test_data[test_data['is_canceled'] != 1]
    test_data_orig_is_canceled = test_data_orig[test_data_orig['is_canceled'] != 1]
    test_data_is_canceled = test_data_is_canceled.drop('is_canceled', axis=1)
    test_data_orig_is_canceled = test_data_orig_is_canceled.drop(labels=['ID', 'is_canceled'], axis=1)

    test_data_is_canceled.to_csv('test_is_canceled.csv', index=False)
    test_data_orig_is_canceled.to_csv('test_orig_is_canceled.csv', index=False)

def validation(train_data, train_adr, train_data_orig):
    print('------------------------------------ Validation ------------------------------------')
    
    x_train, x_valid, adr_train, adr_valid = train_test_split(train_data, train_adr, test_size=13442, shuffle=False)
    
    clf = SVR(C=1)
    clf.fit(x_train, adr_train)

    adr_predict = clf.predict(x_valid)
    E_val_adr = sum(abs(adr_predict - adr_valid))/len(adr_predict)

    x_valid['adr_predict'] = adr_predict
    valid_days = train_data_orig['stays_in_week_nights'].iloc[43684:].values + train_data_orig['stays_in_weekend_nights'].iloc[43684:].values
    x_valid['revenue'] = valid_days * x_valid['adr_predict'].values
    x_valid_groupby = x_valid.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()

    x_valid = x_valid_groupby.reset_index()
    y_predict = np.floor(x_valid['revenue'].values/10000)
    y_valid = train_label['label'].iloc[489:].values
    E_val_label = sum(abs(y_predict - y_valid))/len(y_predict)
    print(E_val_adr, E_val_label)

def main(train_data, train_adr, test_label):
    print('------------------------------------ Main ------------------------------------')
    
    test_is_canceled = pd.read_csv('test_is_canceled.csv')
    test_orig_is_canceled = pd.read_csv('test_orig_is_canceled.csv')

    clf = SVR(C=300)
    clf.fit(train_data, train_adr)

    adr_predict = clf.predict(test_is_canceled)
    days_test = test_orig_is_canceled['stays_in_week_nights'].values + test_orig_is_canceled['stays_in_weekend_nights'].values
    test_is_canceled['revenue'] = days_test * adr_predict
    test_is_canceled_groupby = test_is_canceled.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
    x_test = test_is_canceled_groupby.reset_index()
    y_test = np.floor(x_test['revenue'].values/10000)

    print(y_test)
    test_label['label'] = y_test
    test_label.to_csv('test_label.csv', index=False)

    return y_test

# is_canceled(train_data_drop_adr, test_data)
# validation(train_data_drop_is_canceled, train_adr, train_data_orig_is_canceled)
y_test = main(train_data_drop_is_canceled, train_adr, test_label)