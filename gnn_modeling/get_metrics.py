from data_preprocessing import load_both_datasets
from calc import calc_metrics, calc_switch

def main():
    print('-'*40)
    for dataset in ['List_ACIE_THF',
                    'List_ACIE_THP',
                    'Sunoj_PNAS',
                    'HongAckermann_NatSyn',
                    'SigmanToste_Science']:
        
        for target_column in ['ee', 'ddg']:

            for with_T in [True, False]:

                df_orig, df_pred = load_both_datasets(dataset, target_column, with_T)
                # switch domain
                df_pred_switched, tc_sw = calc_switch(df_pred, df_orig, target_column)

                # Calculate metrics
                mae, rmse, r2  = calc_metrics(df_orig[target_column], df_pred)
                mae_sw, rmse_sw, r2_sw = calc_metrics(df_orig[tc_sw], df_pred_switched)

                print(f'{dataset} {target_column} with_T={with_T}:')
                if target_column == 'ee':
                    x = 100
                    x_neg = -100
                    num_gt_x = (df_pred > x).sum().sum()
                    num_gt_x_neg = (df_pred < x_neg).sum().sum()
                    num_tot = num_gt_x + num_gt_x_neg
                    prop_gt_x = num_tot / df_pred.size * 100
                    print(f'Predictions over {x}% : {num_tot}, {prop_gt_x:.2f}%')

                print(f'MAE: {mae:.3f}')
                print(f'RMSE: {rmse:.3f}')
                print(f'R2: {r2:.3f}')
                print('-'*20)
                print('Domain-Switch')
                print('-'*20)
                print(f'MAE: {mae_sw:.3f}')
                print(f'RMSE: {rmse_sw:.3f}')
                print(f'R2: {r2_sw:.3f}')                
                print('-'*40)

if __name__ == '__main__':
    main()