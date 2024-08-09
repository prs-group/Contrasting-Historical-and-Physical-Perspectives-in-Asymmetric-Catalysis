import config
from model import GCN
from graph_preprocessing import my_disc_graph_only_dataloader
from data_preprocessing import preprocess_data
import torch
from sklearn.model_selection import train_test_split
from train_val import run_epochs, get_predictions
from sklearn.metrics import r2_score
import pandas as pd

def select_params(dataset, target_column, with_T):
    if with_T:
        T = 'Yes'
    else:
        T = 'No'
    params = getattr(config, f'BEST_PARAMETERS_{dataset}_{target_column}_T_{T}')
    return params

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_states_l = 0
    n_states_r = 500
    patience = 50
    delta = 0.00001
    r2_threshold = 0.4

    for dataset in [#'List_ACIE_THF',
                    #'List_ACIE_THP',
                    'Sunoj_PNAS',
                    'HongAckermann_NatSyn',
                    'SigmanToste_Science']:
        for target_column in ['ee', 'ddG']:
            df, smiles_list, scaler = preprocess_data(dataset, target_column)
            for with_T in [True, False]:
                # safety measures
                orig_with_T = with_T
                predictions_df = pd.DataFrame()
                params = select_params(dataset, target_column, with_T)
                for rs in range(n_states_l, n_states_r):
                    print(rs)
                    trys_counter = 0
                    current_r2 = -100.0
                    train_df, val_df = train_test_split(df, random_state=1337)
                    model = GCN(num_features=79,
                                num_layers=params['num_layers'],
                                hidden_channels=params['channel_size'],
                                hl_size=params['hl_size'],
                                with_T=with_T).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
                    loss_fn = torch.nn.MSELoss()

                    train_data, _ = my_disc_graph_only_dataloader(train_df,
                                        smiles_list, 
                                        target_column=target_column,
                                        bs=params['bs'])
                    val_data, _ = my_disc_graph_only_dataloader(val_df,
                                                                smiles_list,
                                                                target_column=target_column,
                                                                bs=len(val_df))
                    while current_r2 < r2_threshold and trys_counter < 5:
                        trys_counter += 1
                        run_epochs(model,
                                   train_data,
                                   val_data,
                                   optimizer,
                                   loss_fn,
                                   device,
                                   patience,
                                   delta,
                                   dataset,
                                   rs,
                                   orig_with_T,
                                   target_column,
                                   verbose=False)
                        if orig_with_T:
                            with_T = 'Yes'
                        else:    
                            with_T = 'No'  

                        model.load_state_dict(torch.load(f'n_fold_models/{dataset}_{target_column}_{with_T}_{rs}.pt'))
                        predictions, targets = get_predictions(model, val_data, device)
                        current_r2 = r2_score(targets, predictions)

                    if current_r2 > r2_threshold:
                        # succesful training
                        print('Run scucessful, n_tries: ', trys_counter)
                    else:
                        print('run failed check')
                        continue
                    
                    # new dataloader for predictions

                    all_data, _ = my_disc_graph_only_dataloader(df,
                                                                smiles_list, 
                                                                target_column=target_column,
                                                                bs=len(df))
                    
                    predictions, trues = get_predictions(model, all_data, device)
                    df['predictions'] = predictions
                    df['predictions'] = scaler.inverse_transform(df[['predictions']])

                    # Append the predictions to the predictions_df DataFrame
                    predictions_df[f'{rs}'] = df['predictions']

                if orig_with_T:
                    with_T = 'Yes'
                else:    
                    with_T = 'No'   

                # Save the predictions_df DataFrame
                predictions_df.to_csv(f'nfold/{dataset}_{target_column}_{with_T}.csv', index=False)

if __name__ == '__main__':
    main()
