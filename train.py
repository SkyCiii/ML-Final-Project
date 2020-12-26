import pandas as pd
from sklearn.svm import SVC

train_data = pd.read_csv('train.csv').fillna(-1)
train_label = pd.read_csv('train_label.csv')

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
train_data_valid = train_data[train_data['is_canceled'] == 0]
week = train_data_valid['stays_in_week_nights']
weekend = train_data_valid['stays_in_weekend_nights']
adr = train_data_valid['adr']
revenue = (week + weekend) * adr

# ------------- main ------------- #
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

# ------------- main ------------- #
train_data = preprocessing(train_data)

# ------------- revenue per day (part2) ------------- #
train_data['revenue'] = revenue
train_data = train_data.fillna(0)
revenue_feature = train_data[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'revenue']]
revenue_sum_groupby = revenue_feature.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
revenue_sum = revenue_sum_groupby.reset_index()
train_label['revenue'] = revenue_sum['revenue']
train_label.to_csv('train_lable_trans.csv')

# ------------- train y, x ------------- #
y = train_label['label'].values
x_orig = train_data.drop(labels=['ID', 'is_canceled', 'adr', 'revenue', 'reservation_status', 'reservation_status_date'], axis=1)
x_groupby = x_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
x = x_groupby.reset_index().values
clf = SVC(C=3, gamma='scale', kernel='poly', degree=3, coef0=0, decision_function_shape='ovr')
clf.fit(x, y)

# ------------- E_in ------------- #
y_in = clf.predict(x)
error = 0
for i in range(0, len(y)):
    error += abs(y[i]-y_in[i])
E_in = error/len(y)
print(E_in)

# ------------- test x ------------- #
test_data = pd.read_csv('test.csv').fillna(-1)
test_label = pd.read_csv('test_nolabel.csv')

test_data = preprocessing(test_data)
test_x_orig = test_data.drop(labels='ID', axis=1)
test_x_groupby = test_x_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).sum()
test_x = test_x_groupby.reset_index().values
test_y = clf.predict(test_x)

# ------------- output ------------- #
print(test_y)
test_label['label'] = test_y
test_label.to_csv('test_label.csv', index=False)