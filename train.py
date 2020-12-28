import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ------------- import ------------- #
train_data = pd.read_csv('train.csv').fillna(-1)
train_label = pd.read_csv('train_label.csv')
test_data = pd.read_csv('test.csv').fillna(-1)
test_label = pd.read_csv('test_nolabel.csv')

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
test_data = preprocessing(test_data)

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

y_all = train_label['label'].values

x_all = train_data.drop(labels=['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date'], axis=1)
x_all_groupby = x_all.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
x_all = x_all_groupby.reset_index().values

# ------------- validating y, x and selecet g- with minima error ------------- #
gamma = ['scale', 'auto']
kernel = ['poly', 'rbf', 'sigmoid']
def select_model(C, kernel, degree, coef0, is_score):
    # ------------- split ------------- #
    x_train_part, x_valid_part, y_train_part, y_valid_part = train_test_split(x_all, y_all, test_size=0.3)

    point_list = []
    for c in range(5, C*10+1, 5):
        c = c/10
        for k in kernel:
            for d in range(3, degree+1):
                for coef in range(0, coef0+1):
                    clf = SVC(C=c, kernel=k, degree=d, coef0=coef)

                    # ------------- best score ------------- #
                    clf.fit(x_train_part, y_train_part)
                    if is_score:
                        point = clf.score(x_valid_part, y_valid_part)
                    else:
                        y_val = clf.predict(x_valid_part)
                        point = sum(abs(y_val - y_valid_part))/len(y_val)

                    point_list.append([c, k, d, coef, point])
                    
    return point_list

def main(counts, is_score):
    y_test_list = []
    for count in range(0, counts):
        print('------------------------------------ ' + str(count) + ' ------------------------------------')
        point_list = select_model(5, kernel, 5, 5, is_score)
        point_frame = pd.DataFrame(point_list, columns=['C', 'kernel', 'degree', 'coef', 'point(score->max/E_val->min)'])
        point_frame.to_csv('validation_error.csv')
        if is_score:
            point = point_frame[point_frame['point(score->max/E_val->min)'] == point_frame['point(score->max/E_val->min)'].max()]
        else:
            point = point_frame[point_frame['point(score->max/E_val->min)'] == point_frame['point(score->max/E_val->min)'].min()]
        print(point)

        # ------------- recalculate g with g- and get the optimal parameters ------------- #
        c_best = point['C'].values[0]
        k_best = point['kernel'].values[0]
        d_best = point['degree'].values[0]
        coef_best = point['coef'].values[0]
        print(
            'C: ', c_best,
            'K: ', k_best,
            'degree: ', d_best,
            'coef0: ', coef_best,
        )

        # ------------- fit the model with g and optimal parameters ------------- #
        clf = SVC(C=c_best, kernel=k_best, degree=d_best, coef0=coef_best)
        clf.fit(x_all, y_all)

        # ------------- E_in ------------- #
        y_in = clf.predict(x_all)
        E_in = sum(abs(y_in - y_all))/len(y_all)
        print(
            'E_in: ', E_in,
            'score_in: ', clf.score(x_all, y_all),
        )

        # ------------- test ------------- #
        x_test_orig = test_data.drop(labels='ID', axis=1)
        x_test_groupby = x_test_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
        x_test = x_test_groupby.reset_index().values

        y_test = clf.predict(x_test)
        y_test_list.append(y_test)

    y_final = np.around(sum(y_test_list)/counts)

    return y_final

y_final = main(1000, 1)
print(y_final)
test_label['label'] = y_final
test_label.to_csv('test_label.csv', index=False)

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