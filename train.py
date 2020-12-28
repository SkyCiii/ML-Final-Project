import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR

# BEST: C=3, poly=3, x.sum()

train_data = pd.read_csv('train.csv').fillna(-1)
train_label = pd.read_csv('train_label.csv')

# ------------- drop invalid data ------------- #
adr = train_data['adr']
week = train_data['stays_in_week_nights']
weekend = train_data['stays_in_weekend_nights']
adults = train_data['adults']
children = train_data['children']
babies = train_data['babies']

train_data = train_data[
    (adr >= 0) |
    ((week + weekend) != 0) |
    ((adults + children + babies) != 0)
]

# ------------- convert string to [0, 1] with lower bound != 0 ------------- #
def category_convertion(category_list):
    converter = {}
    step = 1/len(category_list)
    for index in range(0, len(category_list)):
        converter[category_list[index]] = step * (index + 1)
    return converter

# ------------- convert string to [0, 1] with lower bound = 0 ------------- #
def normalization(feature_value):
    normalizer = {}
    step = 1/len(feature_value)
    for index in range(0, len(feature_value)):
        normalizer[feature_value[index]] = step * index
    return normalizer

# ------------- customized list for specific convertion ------------- #
month_category = [
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December'
]
room_category = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'
]

# ------------- revenue per day (part1) ------------- #
# train_data_not_canceled = train_data[train_data['is_canceled'] == 0]
# week_not_canceled = train_data_not_canceled['stays_in_week_nights']
# weekend_not_canceled = train_data_not_canceled['stays_in_weekend_nights']
# adr_not_canceled = train_data_not_canceled['adr']
# revenue = (week_not_canceled + weekend_not_canceled) * adr_not_canceled

# ------------- preprocessing ------------- #
def preprocessing(data):
    try:
        features = data.columns.drop(['ID', 'adr', 'reservation_status', 'reservation_status_date'])
    except KeyError:
        features = data.columns.drop('ID')
    finally:
        for feature in features:
            if 0 in data[feature].values:
                value_as_category = data[feature].astype('category').values.categories
                normalizer = normalization(value_as_category)
                data[feature] = data[feature].map(normalizer)
            elif feature == 'arrival_date_month':
                converter = category_convertion(month_category)
                data[feature] = data[feature].map(converter)
            elif feature == 'reserved_room_type' or feature == 'assigned_room_type':
                converter = category_convertion(room_category)
                data[feature] = data[feature].map(converter)
            else:
                category = data[feature].astype('category').values.categories
                converter = category_convertion(category)
                data[feature] = data[feature].map(converter)

        return data

train_data = preprocessing(train_data)

# ------------- revenue per day (part2) ------------- #
# train_data['revenue'] = revenue
# train_data = train_data.fillna(0)
# revenue_feature = train_data[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'revenue']]
# revenue_sum_groupby = revenue_feature.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
# revenue_sum = revenue_sum_groupby.reset_index()
# train_label['revenue'] = revenue_sum['revenue']
# train_label.to_csv('train_lable_trans.csv')

def agg_function(x):
    return x.sum()

# ------------- split data into training and validating set ------------- #
valid_data = train_data.iloc[70000:, :]
train_data_part = train_data.iloc[:70000:, :]

valid_label = train_label.iloc[486:, :]
train_label_part = train_label.iloc[:487, :]

# ------------- original data without splitting ------------- #
y_fit = train_label['label'].values

x_fit = train_data.drop(labels=['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date'], axis=1)
x_fit_groupby = x_fit.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
x_fit = x_fit_groupby.reset_index().values

# ------------- training y, x ------------- #
y_train = train_label_part['label'].values
y_valid = valid_label['label'].values

x_train_orig = train_data_part.drop(labels=['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date'], axis=1)
x_train_groupby = x_train_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
x_train = x_train_groupby.reset_index().values

x_valid_orig = valid_data.drop(labels=['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date'], axis=1)
x_valid_groupby = x_valid_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
x_valid = x_valid_groupby.reset_index().values

# ------------- validating y, x and selecet g- with minima error ------------- #
gamma = ['scale', 'auto']
kernel = ['poly', 'rbf', 'sigmoid']
def select_model(C, kernel, degree, coef0):
    score_list = []
    for c in range(5, C*10+1, 5):
        c = c/10
        for k in kernel:
            for d in range(3, degree+1):
                for coef in range(0, coef0+1):
                    clf = SVC(C=c, kernel=k, degree=d, coef0=coef)

                    # ------------- best score ------------- #
                    clf.fit(x_train, y_train)
                    score = clf.score(x_valid, y_valid)
                    score_list.append([c, k, d, coef, score])

                    y_val = clf.predict(x_valid)
                    error = 0
                    for i in range(0, len(y_valid)):
                        error += abs(y_val[i]-y_valid[i])
                    E_val = error/len(y_valid)
                    
                    if E_val < 0.5:
                        print(c, k ,d ,coef, E_val)
                    
    return score_list

score_list = select_model(5, kernel, 5, 5)
score_frame = pd.DataFrame(score_list, columns=['C', 'kernel', 'degree', 'coef', 'score'])
score_frame.to_csv('validation_error.csv')
score_max = score_frame[score_frame['score'] == score_frame['score'].max()]

# ------------- recalculate g with g- ------------- #
c_best = score_max['C'].values[0]
k_best = score_max['kernel'].values[0]
d_best = score_max['degree'].values[0]
coef_best = score_max['coef'].values[0]
print('C: ', c_best, 'K: ', k_best, 'degree: ', d_best, 'coef0: ', coef_best)
clf = SVC(C=c_best, kernel=k_best, degree=d_best, coef0=coef_best)
clf.fit(x_fit, y_fit)

# ------------- E_in ------------- #
y_in = clf.predict(x_fit)
print('score_in: ', clf.score(x_fit, y_fit))

# ------------- test ------------- #
test_data = pd.read_csv('test.csv').fillna(-1)
test_label = pd.read_csv('test_nolabel.csv')
test_data = preprocessing(test_data)

x_test_orig = test_data.drop(labels='ID', axis=1)
x_test_groupby = x_test_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
x_test = x_test_groupby.reset_index().values

y_test = clf.predict(x_test)
test_label['label'] = y_test
test_label.to_csv('test_label.csv', index=False)
print('test output: ', y_test, sep='\n')

# ------------- without data compression ------------- #
# y = train_data['revenue'].values
# x_orig = train_data.drop(labels=['ID', 'is_canceled', 'adr', 'revenue', 'reservation_status', 'reservation_status_date'], axis=1)
# x = x_orig.values
# regr = SVR(C=1.0, epsilon=0.2)
# print(regr.fit(x, y))

# test_data = pd.read_csv('test.csv').fillna(-1)
# test_label = pd.read_csv('test_nolabel.csv')

# week = test_data['stays_in_week_nights']
# weekend = test_data['stays_in_weekend_nights']

# test_data = preprocessing(test_data)
# test_x_orig = test_data.drop(labels='ID', axis=1).fillna(0)
# test_x = test_x_orig.values
# test_y = regr.predict(test_x)
# test_data['adr'] = test_y
# adr = test_data['adr']
# revenue = (week + weekend) * adr
# test_data['revenue'] = revenue
# test_revenue_feature = test_data[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'revenue']]
# test_revenue_sum_groupby = test_revenue_feature.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
# test_revenue_sum_groupby = test_revenue_sum_groupby.reset_index()
# revenue = test_revenue_sum_groupby['revenue']
# label = revenue.agg(lambda x: int(x/10000))
# print(label.values)
# test_label['label'] = label
# test_label.to_csv('test_label.csv')