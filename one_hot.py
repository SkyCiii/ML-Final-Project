import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

# ------------- import ------------- #
train_data = pd.read_csv('train.csv')
train_label = pd.read_csv('train_label.csv')
test_data = pd.read_csv('test.csv')
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

# ------------- customized list for specific convertion ------------- #
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

def one_hot_processing(feature):
    enc = OneHotEncoder()
    if feature in one_hot_category:
        train_data[feature] = train_data[feature].fillna('N/A')
        test_data[feature] = test_data[feature].fillna('N/A')
    elif feature in one_hot_category_num:
        train_data[feature] = train_data[feature].fillna(-1)
        test_data[feature] = test_data[feature].fillna(-1)
    train_category = train_data[feature].astype('category').values.categories
    test_category = test_data[feature].astype('category').values.categories
    categories = np.append(train_category, test_category).reshape(-1, 1)
    enc.fit(categories)
    enc_len = enc.transform(train_data[feature].values.reshape(-1, 1)).toarray().shape[1]
    for idx in range(1, enc_len+1):
        train_data[feature+'_'+str(idx)] = pd.DataFrame(enc.transform(train_data[feature].values.reshape(-1, 1)).toarray())[idx-1]
        test_data[feature+'_'+str(idx)] = pd.DataFrame(enc.transform(test_data[feature].values.reshape(-1, 1)).toarray())[idx-1]
    train_data.pop(feature)
    test_data.pop(feature)
    
    return train_data, test_data

# ------------- preprocessing ------------- #
def preprocessing(train_data, test_data):
    train_features = train_data.columns.drop(['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date'])
    test_features = test_data.columns.drop('ID')
    for feature in train_features:
        scaler = MinMaxScaler(feature_range=(0, 1))
        le = LabelEncoder()
        if feature in one_hot_category or feature in one_hot_category_num:
            train_data, test_data = one_hot_processing(feature)
        elif train_data[feature].values.dtype == np.int32 or train_data[feature].values.dtype == np.float64:
            train_data[feature] = train_data[feature].fillna(0)
            test_data[feature] = test_data[feature].fillna(0)
            scaler.fit(train_data[[feature]])
            train_data[feature] = scaler.transform(train_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])
        elif feature == 'arrival_date_month':
            train_data[feature] = train_data[feature].map(month_converter)
            test_data[feature] = test_data[feature].map(month_converter)
            scaler.fit(train_data[[feature]])
            train_data[feature] = scaler.transform(train_data[[feature]])
            test_data[feature] = scaler.transform(test_data[[feature]])

    return train_data, test_data

train_data, test_data = preprocessing(train_data, test_data)

train_data.to_csv('train_one_hot.csv', index=False)
test_data.to_csv('test_one_hot.csv', index=False)