import math
import pandas as pd
import numpy as np


def read_data(path):
    data = pd.read_csv(path)
    for name_column in data.columns:
        if type_attribute(data[name_column][0]) != 'str':
            if type(data[name_column][0]) == str:
                data[name_column] = data[name_column].str.replace(',', '.')
            data[name_column] = pd.to_numeric(data[name_column], errors='coerce')

    return data


def mean(df_column):
    return round(np.sum(df_column) / len(df_column), 2)


def mod(df_column):
    df_column = df_column.dropna()
    unique, counts = np.unique(df_column, return_counts=True)
    return unique[np.argmax(counts)]


def median(df_column):
    column = df_column.dropna()
    column = column.sort_values(ascending=True).reset_index(drop=True)
    if len(column) % 2 == 0:
        return (column[len(column) // 2] + column[len(column) // 2 - 1]) / 2
    else:
        return column[len(column) // 2]


def ecartType(df_column):
    return np.sqrt((np.sum(np.power(df_column - mean(df_column), 2))) / len(df_column))


def variance(df_column):
    return np.power(ecartType(df_column), 2)


def type_attribute(val):
    if isinstance(val, str):
        val = val.replace(",", ".")
    if isinstance(val, str) and '.' in val and all(c.isdigit() or c == '.' for c in val):
        return 'float'
    elif isinstance(val, (int, np.int64)):
        return 'int'
    elif isinstance(val, (float, np.float64)):
        return 'float'
    else:
        return 'str'


def min_max_normalisation(df, min=0, max=1):
    if max < min:
        print("min superior to max, please change values")
        return df

    for column in df.columns:
        if type_attribute(df[column][0]) == 'str': continue

        min_old = df[column].min()
        max_old = df[column].max()

        df[column] = (df[column] - min_old) / (max_old - min_old) * (max - min) + min

    return df


def z_score_normalisation(df):
    for column in df.columns:
        if type_attribute(df[column][0]) == 'str': continue

        mean_column = mean(df[column])
        standard_deviation = ecartType(df[column])

        df[column] = (df[column] - mean_column) / standard_deviation

    return df


def replace_missing_values(df_column, method='mean'):
    if type_attribute(df_column[0]) == 'str':
        if method != 'mod':
            return df_column
        df_column.replace(['', ' ', '?'], np.nan, inplace=True)
    else: df_column = pd.to_numeric(df_column, errors='coerce')
    if method == 'mean':
        df_column.fillna(mean(df_column), inplace=True)
    elif method == 'mod':
        df_column.fillna(mod(df_column), inplace=True)
    elif method == 'median':
        df_column.fillna(median(df_column), inplace=True)
    return df_column


def detect_outliers(df_column):
    column = df_column.dropna()
    column = column.sort_values(ascending=True).reset_index(drop=True)

    length = len(column)
    Q1 = column[length // 4] if length % 4 != 0 else (column[length // 4 - 1] + column[length // 4]) / 2
    Q3 = column[(3 * length) // 4] if (3 * length) % 4 != 0 else (column[(3 * length) // 4 - 1] + column[
        (3 * length) // 4]) / 2
    IQR = Q3 - Q1
    IQR_min = Q1 - IQR * 1.5
    IQR_max = Q3 + IQR * 1.5
    IQR_min = round(IQR_min, 2)
    IQR_max = round(IQR_max, 2)

    outliers_index = df_column[(df_column < IQR_min) | (df_column > IQR_max)].index

    return outliers_index, IQR_min, IQR_max


def replace_outliers(df_column, method='mean'):
    if type_attribute(df_column[0]) == 'str': return df_column
    df_column = df_column.copy()
    outliers_index, IQR_min, IQR_max = detect_outliers(df_column)
    while len(outliers_index) != 0:
        if method == 'null':
            df_column.iloc[outliers_index] = np.nan
        elif method == 'mean':
            df_column[outliers_index] = mean(df_column)
        elif method == 'mod':
            df_column[outliers_index] = mod(df_column)
        elif method == 'median':
            df_column[outliers_index] = median(df_column)
        elif method == 'IQR-min':
            df_column[outliers_index] = IQR_min
        elif method == 'IQR-max':
            df_column[outliers_index] = IQR_max
        else:
            print('Methode no recognized')
            return df_column
        outliers_index, IQR_min, IQR_max = detect_outliers(df_column)

    return df_column


def delete_duplicate_rows(df):
    unique_rows = []
    for index, row in df.iterrows():
        if row.tolist() not in unique_rows:
            unique_rows.append(row.tolist())
    return pd.DataFrame(unique_rows, columns=df.columns)


def equalFrequencyDiscretization(df_column, Q, interval=None):
    data_type = type_attribute(df_column[0])
    if data_type == 'str':
        print("You can't discretize a string attribute.")
        return df_column
    if type(df_column[0]) == str:
        df_column = df_column.str.replace(',', '.')
    df_column = pd.to_numeric(df_column, errors='coerce')
    column = df_column.dropna()
    column = column.sort_values(ascending=True).reset_index(drop=True)
    index_Q = [int(len(column) * i / Q) for i in range(1, Q)]
    borne_Q = [column[0]] + [column[i] for i in index_Q] + [float('inf')]
    if interval is None: interval = range(Q)
    for index, value in df_column.items():
        for i in range(len(borne_Q) - 1):
            if borne_Q[i] <= value < borne_Q[i + 1]:
                df_column.at[index] = interval[i]
                break
    return df_column


def equalWidthDiscretization(df_column, Q, interval=None):
    data_type = type_attribute(df_column[0])
    if data_type == 'str':
        print("You can't discretize a string attribute.")
        return df_column
    if type(df_column[0]) == str:
        df_column = df_column.str.replace(',', '.')
    df_column = pd.to_numeric(df_column, errors='coerce')
    column = df_column.dropna()
    min_value = column.min()
    max_value = column.max()
    size = (max_value - min_value) / Q
    size = round(size, 4)
    if interval is None: interval = range(Q)
    for index, value in df_column.items():
        try:
            classe = int((value - min_value) / size)
            classe = min(classe, Q - 1)
            df_column.at[index] = interval[int(classe)]
        except:
            print("error has occurred")

    return df_column


def dataset_describe(df):
    count_missing_values = 0
    df_info = {'Measures/Attributes': ['Type', 'Unique values', 'Number of missing values',
                                       'Min', 'Max', 'Mean', 'Median', 'Mod', 'symetrie']}
    for column in df.columns:
        symetrie = ""
        data_type = type_attribute(df[column][0])
        unique_values = len(df[column].unique())
        if data_type != 'str' and unique_values > 10:
            meanVal = mean(df[column])
            medianVal = median(df[column])
            minVal = min(df[column])
            maxVal = max(df[column])
        else:
            minVal = "/"
            maxVal = "/"
            meanVal = "/"
            medianVal = "/"
            symetrie = '/'
        modVal = mod(df[column])
        if symetrie == "":
            if round(meanVal, 1) == round(medianVal, 1) == round(modVal, 1):
                symetrie = "Symetique"
            elif round(meanVal, 1) > round(medianVal, 1) > round(modVal, 1):
                symetrie = "Positivement"
            elif round(meanVal, 1) < round(medianVal, 1) < round(modVal, 1):
                symetrie = "Négativement"
            else:
                symetrie = "Inindentifiée"
        count_missing = df[column].isnull().sum()
        count_missing_values += count_missing
        df_info[column] = [data_type, unique_values, count_missing, minVal, maxVal,
                           meanVal, medianVal, modVal, symetrie]
    print("Dataset shape: ", df.shape)
    print("Number of missing values: ", count_missing_values)
    return pd.DataFrame(df_info), count_missing_values
