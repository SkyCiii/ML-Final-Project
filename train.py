import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train_one_hot.csv')
train_label = pd.read_csv('train_label.csv')
test_data = pd.read_csv('test_one_hot.csv')
test_label = pd.read_csv('test_nolabel.csv')

# train_data = pd.read_csv('train_no_one_hot.csv')
# train_label = pd.read_csv('train_label.csv')
# test_data = pd.read_csv('test_no_one_hot.csv')
# test_label = pd.read_csv('test_nolabel.csv')

def agg_function(x):
    return x.sum()

y_all = train_label['label'].values
x_all_orig = train_data.drop(labels=['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date'], axis=1)
x_all_groupby = x_all_orig.groupby(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']).agg(agg_function)
x_all = x_all_groupby.reset_index().values

# ------------- validating y, x and selecet g- with minima error ------------- #
def select_model(is_score):
    # ------------- split ------------- #
    x_train_part, x_valid_part, y_train_part, y_valid_part = train_test_split(x_all, y_all, test_size=0.2, shuffle=False)

    clf = RandomForestClassifier()

    # ------------- best score ------------- #
    clf.fit(x_train_part, y_train_part)
    if is_score:
        point = clf.score(x_valid_part, y_valid_part)
    else:
        y_val = clf.predict(x_valid_part)
        point = sum(abs(y_val - y_valid_part))/len(y_val)
        y_ref = clf.predict(x_train_part)
        point_ref = sum(abs(y_ref - y_train_part))/len(y_ref)
    print(point, point_ref)

    return clf

def main(counts, is_score):
    y_test_list = []
    for count in range(0, counts):
        print('------------------------------------ ' + str(count) + ' ------------------------------------')
        clf = select_model(is_score)

        # ------------- fit the model with g and optimal parameters ------------- #
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

y_final = main(1, 0)
print(y_final)
test_label['label'] = y_final
test_label.to_csv('test_label.csv', index=False)