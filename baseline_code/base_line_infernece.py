import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path

def main():
    group_size = 27

    # Load info and feature paths
    train_info = pd.read_csv('./39_Training_Dataset/train_info.csv')
    test_info = pd.read_csv('./39_Test_Dataset/test_info.csv')
    train_datapath = './39_Training_Dataset/tabular_data_train'
    test_datapath = './39_Test_Dataset/tabular_data_test'
    train_datalist = list(Path(train_datapath).glob('**/*.csv'))
    test_datalist = list(Path(test_datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    # Prepare training features and labels
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    for file in train_datalist:
        unique_id = int(Path(file).stem)
        row = train_info[train_info['unique_id'] == unique_id]
        if row.empty:
            continue
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        x_train = pd.concat([x_train, data], ignore_index=True)
        y_train = pd.concat([y_train, target_repeated], ignore_index=True)

    # Prepare test features and test IDs
    x_test = pd.DataFrame()
    test_ids = []
    for file in test_datalist:
        unique_id = int(Path(file).stem)
        test_ids.append(unique_id)
        data = pd.read_csv(file)
        x_test = pd.concat([x_test, data], ignore_index=True)

    # Normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    # Label encoding
    le_gender = LabelEncoder()
    le_hold = LabelEncoder()
    le_years = LabelEncoder()
    le_level = LabelEncoder()

    y_gender = le_gender.fit_transform(y_train['gender'])
    y_hold = le_hold.fit_transform(y_train['hold racket handed'])
    y_years = le_years.fit_transform(y_train['play years'])
    y_level = le_level.fit_transform(y_train['level'])

    # Binary prediction helper
    def predict_binary(X_train, y_train, X_test):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        pos_probs = [p[0] for p in proba]
        group_preds = [max(pos_probs[i*group_size:(i+1)*group_size]) for i in range(len(pos_probs)//group_size)]
        return group_preds

    # Multi-class prediction helper
    def predict_multi(X_train, y_train, X_test, num_class):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        pred_list = []
        for i in range(len(proba) // group_size):
            group = proba[i*group_size:(i+1)*group_size]
            class_sums = [sum([g[j] for g in group]) for j in range(num_class)]
            chosen_class = np.argmax(class_sums)
            best_idx = np.argmax([g[chosen_class] for g in group])
            pred_list.append(group[best_idx])  # softmax-like
        return pred_list

    # Generate predictions
    pred_gender = predict_binary(X_train_scaled, y_gender, X_test_scaled)
    pred_hold = predict_binary(X_train_scaled, y_hold, X_test_scaled)
    pred_years = predict_multi(X_train_scaled, y_years, X_test_scaled, len(le_years.classes_))
    pred_level = predict_multi(X_train_scaled, y_level, X_test_scaled, len(le_level.classes_))

    # Submission formatting
    year_cols = [f'play years_{i}' for i in range(len(le_years.classes_))]
    level_cols = [f'level_{i+2}' for i in range(len(le_level.classes_))]

    result_df = pd.DataFrame()
    result_df['unique_id'] = test_ids
    result_df['gender'] = pred_gender
    result_df['hold racket handed'] = pred_hold

    # One-hot-ish probabilities for multi-class
    for i, probs in enumerate(pred_years):
        for j, col in enumerate(year_cols):
            if col not in result_df:
                result_df[col] = 0.0
            result_df.loc[i, col] = probs[j]

    for i, probs in enumerate(pred_level):
        for j, col in enumerate(level_cols):
            if col not in result_df:
                result_df[col] = 0.0
            result_df.loc[i, col] = probs[j]

    # Final column ordering
    result_df = result_df[['unique_id', 'gender', 'hold racket handed'] + year_cols + level_cols]
    result_df.to_csv('submission.csv', index=False)
    print("✅ submission.csv generated!")

if __name__ == '__main__':
    main()