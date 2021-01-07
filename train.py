import pandas as pd
import numpy as np
import math
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

# train_data = pd.read_csv('train_one_hot.csv')
# train_label = pd.read_csv('train_label.csv')
# test_data = pd.read_csv('test_one_hot.csv')
# test_label = pd.read_csv('test_nolabel.csv')

train_data = pd.read_csv('train_no_one_hot.csv')
train_label = pd.read_csv('train_label.csv')
test_data = pd.read_csv('test_no_one_hot.csv')
test_label = pd.read_csv('test_nolabel.csv')
train_adr = pd.read_csv('train_adr.csv')
train_orig = pd.read_csv('train_orig.csv')

adr = train_adr['adr'].values
x_all = train_data.drop(labels=['ID', 'is_canceled', 'reservation_status', 'reservation_status_date'], axis=1)
        
def agg_function(x):
    return x.sum()

def select_model(is_score):
    for c in [1, 10, 100, 200, 500, 1000, 1200, 1500]:
        print('------------------------------------ Yiiiii ------------------------------------')
        train_data = pd.read_csv('train_no_one_hot.csv')
        train_label = pd.read_csv('train_label.csv')
        test_data = pd.read_csv('test_no_one_hot.csv')
        test_label = pd.read_csv('test_nolabel.csv')
        train_adr = pd.read_csv('train_adr.csv')
        train_orig = pd.read_csv('train_orig.csv')

        adr = train_adr['adr'].values
        x_all = train_data.drop(labels=['ID', 'is_canceled', 'reservation_status', 'reservation_status_date'], axis=1)
        # x_all_groupby = x_all_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
        # x_all = x_all_groupby.reset_index().values

        # ------------- validating y, x and selecet g- with minima error ------------- #
        # ------------- split ------------- #
        x_train_part, x_valid_part, adr_train_part, adr_valid_part = train_test_split(x_all, adr, test_size=13442, shuffle=False)

        clf = SVR(C=c)
        # ------------- best score ------------- #
        clf.fit(x_train_part, adr_train_part)
        print(clf)

        adr_val = clf.predict(x_valid_part)
        point = sum(abs(adr_val - adr_valid_part))/len(adr_val)

        # print(x_valid_part, pd.DataFrame(adr_val), point)
        x_valid_part['adr_predict'] = adr_val
        valid_days = train_orig['stays_in_week_nights'].iloc[43684:].values + train_orig['stays_in_weekend_nights'].iloc[43684:].values
        x_valid_part['revenue'] = valid_days * x_valid_part['adr_predict'].values
        # print(x_valid_part['revenue'])
        x_valid_groupby = x_valid_part.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
        # print(x_valid_groupby['revenue'], len(x_valid_groupby['revenue']))
        x_valid = x_valid_groupby.reset_index()
        y_valid_part = np.floor(x_valid['revenue'].values/10000)
        y_val = train_label['label'].iloc[489:].values
        print(c, sum(abs(y_val - y_valid_part))/len(y_val))
    return clf

# clf = select_model(0)

# adr_test_list = []
clf = SVR(C=1)
# ------------- fit the model with g and optimal parameters ------------- #
clf.fit(x_all, adr)

# ------------- E_in ------------- #
adr_in = clf.predict(x_all)
E_in = sum(abs(adr_in - adr))/len(adr)
print(
    'E_in: ', E_in,
    'score_in: ', clf.score(x_all, adr),
)

# ------------- test ------------- #
x_test_orig = test_data.drop(labels='ID', axis=1)
# # x_test_groupby = x_test_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
# # x_test = x_test_groupby.reset_index().values

test_orig = pd.read_csv('test_orig.csv')
adr_test = clf.predict(x_test_orig)
print(adr_test)
days_test = test_orig['stays_in_week_nights'].values + test_orig['stays_in_weekend_nights'].values
x_test_orig['revenue'] = days_test * adr_test
print(x_test_orig['revenue'])
x_test_groupby = x_test_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
print(x_test_groupby)
x_test = x_test_groupby.reset_index()
y_test = np.floor(x_test['revenue'].values/10000)

print(y_test)
test_label['label'] = y_test
test_label.to_csv('test_label.csv', index=False)