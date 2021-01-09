import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler

train_data = pd.read_csv('train.csv')
train_label = pd.read_csv('train_label.csv')
test_data = pd.read_csv('test.csv')
test_label = pd.read_csv('test_nolabel.csv')

adr = train_data['adr']
week = train_data['stays_in_week_nights']
weekend = train_data['stays_in_weekend_nights']
adults = train_data['adults']
children = train_data['children']
babies = train_data['babies']
week_test = test_data['stays_in_week_nights']
weekend_test = test_data['stays_in_weekend_nights']
adults_test = test_data['adults']
children_test = test_data['children']
babies_test = test_data['babies']

train_data = train_data[adr > 0]
train_data = train_data[(week + weekend) != 0]
train_data = train_data[(adults + children + babies) != 0]

test_data = test_data[(week_test + weekend_test) != 0]
test_data = test_data[(adults_test + children_test + babies_test) != 0]

month_converter = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
one_hot_category = [
    'hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type'
]
one_hot_category_num = [
    'agent', 'company'
]

def preprocessing(train_data, test_data):
    train_features = train_data.columns.drop(['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date'])
    test_features = test_data.columns.drop('ID')
    for feature in train_features:
        scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        le = LabelEncoder()
        if feature in one_hot_category_num:
            # scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            train_data[feature] = train_data[feature].fillna(0)
            test_data[feature] = test_data[feature].fillna(0)
            scaler.fit(train_data[[feature]])
            train_data[feature] = scaler.transform(train_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])
        elif train_data[feature].values.dtype == np.int32 or train_data[feature].values.dtype == np.float64:
            train_data[feature] = train_data[feature].fillna(0)
            test_data[feature] = test_data[feature].fillna(0)
            scaler.fit(train_data[[feature]])
            train_data[feature] = scaler.transform(train_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])
        elif feature in one_hot_category:
            # scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            train_data[feature] = train_data[feature].fillna('N/A')
            test_data[feature] = test_data[feature].fillna('N/A')
            train_category = train_data[feature].astype('category').values.categories
            test_category = test_data[feature].astype('category').values.categories
            categories = np.append(train_category, test_category)
            le.fit(categories)
            train_data[feature] = le.transform(train_data[feature])
            test_data[feature] = le.transform(test_data[feature])
            scaler.fit(train_data[[feature]])
            train_data[feature] = scaler.transform(train_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])
        elif feature == 'arrival_date_month':
            train_data[feature] = train_data[feature].map(month_converter)
            test_data[feature] = test_data[feature].map(month_converter)
            scaler.fit(train_data[[feature]])
            train_data[feature] = scaler.transform(train_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])
        else:
            train_data[feature] = train_data[feature].fillna('N/A')
            test_data[feature] = test_data[feature].fillna('N/A')
            train_category = train_data[feature].astype('category').values.categories
            test_category = test_data[feature].astype('category').values.categories
            categories = np.append(train_category, test_category)
            le.fit(categories)
            train_data[feature] = le.transform(train_data[feature])
            test_data[feature] = le.transform(test_data[feature])
            scaler.fit(train_data[[feature]])
            train_data[feature] = scaler.transform(train_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])

    return train_data, test_data

train_data, test_data = preprocessing(train_data, test_data)

is_canceled_label = train_data.pop('is_canceled')
is_canceled_label.to_csv('is_canceled_label.csv', index=False)
train_data.to_csv('train_is_canceled.csv', index=False)