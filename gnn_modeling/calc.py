import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def calc_ddg(ee, T):
    '''
    ee in %
    T in Kelvin
    er = (1 + ee)/(1 - ee)
    er = exp(-ddg/RT)
    ln(er) = -ddg/RT
    ddg = -ln(er)RT
    '''
    # cap ee
    if ee >= 100:
        ee = 99.9
    elif ee <= -100:
        ee = -99.9
    T = float(T)
    R_const = 8.31446261815324/1000 # kJ / (mol K)
    er = (1 + ee/100) / (1 - ee/100)
    ddg = np.log(er) * R_const * T # kJ / mol
    ddg *= 1/4.1839954  # kcal / mol
    
    return ddg

def calc_ee(ddg, T):
    '''
    ddg in kcal/mol
    T in Kelvin
    er = exp(-ddg/RT)
    ee = 100 * (er - 1) / (er + 1)
    '''
    # prevent overflow
    if ddg > 10:
        return 99.9
    
    T = float(T)
    R_const = 8.31446261815324/1000 # kJ / (mol K)
    ddg *= 4.1839954 # convert from kcal/mol to kJ/mol
    er = np.exp(ddg / (R_const * T))
    ee = 100 * (er - 1) / (er + 1)
    if ee <= 99.9:
        return ee
    else:
        return 99.9

def calc_switch(dataframe, T_frame, tc):
    df_switched = pd.DataFrame()
    tc_sw = 'something_wrong_in_switching'
    df_list = []
    dataframe['T'] = T_frame['T']
    for col in dataframe.columns:
        if col != 'T':
            if tc == 'ee':
                df_col = dataframe.apply(lambda row: calc_ddg(row[col], row['T']), axis=1)
                tc_sw = 'ddg'
            else:
                df_col = dataframe.apply(lambda row: calc_ee(row[col], row['T']), axis=1)
                tc_sw = 'ee'
            df_list.append(df_col)
    df_switched = pd.concat(df_list, axis=1)
    dataframe = dataframe.drop(columns=['T'])
    df_switched.columns = dataframe.columns
    
    return df_switched, tc_sw

def calc_metrics(data_orig, data_pred, only_test = True):

    # only from test set
    maes = []
    rmses = []
    r2s = []
    for col in data_pred.columns:
        if not 'Unnamed' in col and not 'T' in col:
            if only_test:
                train_idx, test_idx = train_test_split(range(len(data_pred)), test_size = 0.2, random_state = int(col))
                data_pred_test = data_pred[col].iloc[test_idx]
                data_orig_test = data_orig.iloc[test_idx]
            else:
                data_pred_test = data_pred[col]
                data_orig_test = data_orig
            mae = mean_absolute_error(data_orig_test, data_pred_test)
            rmse = np.sqrt(mean_squared_error(data_orig_test, data_pred_test))
            r2 = r2_score(data_orig_test, data_pred_test)
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)
    mae = np.mean(maes)
    rmse = np.mean(rmses)
    r2 = np.mean(r2s)
    return mae, rmse, r2