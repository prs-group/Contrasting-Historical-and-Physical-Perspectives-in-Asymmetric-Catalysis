from model import GCN
import torch
import optuna 
import joblib
import pandas  as pd
from sklearn.model_selection import train_test_split
import json 
import os.path
from data_preprocessing import preprocess_data
from graph_preprocessing import my_disc_graph_only_dataloader
from train_val import train, val

def objective(trial, train_df, val_data, smiles_list, target_column, with_T, device):
    cfg = { 'lr'                        :   trial.suggest_float("lr", 1e-6, 1e-2),
            'hl_size'                   :   trial.suggest_int("hl_size", 32, 512),  
            'channel_size'              :   trial.suggest_int("channel_size", 32, 512),  
            'num_layers'                :   trial.suggest_int('num_layers', 3,5),
            'bs'                        :   trial.suggest_int('bs', 2, 32)}

    model = GCN(num_features=79,
                num_layers=cfg['num_layers'],
                hidden_channels=cfg['channel_size'],
                hl_size=cfg['hl_size'],
                with_T=with_T).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.MSELoss()
    train_data, _ = my_disc_graph_only_dataloader(train_df,
                                                smiles_list, 
                                                target_column=target_column,
                                                bs=cfg['bs'])

    print(cfg)
    best_loss = float('inf')
    patience = 50
    delta = 0.00001
    counter = 0
    epoch = 0
    for epoch in range(5000):
        loss = train(model, train_data, optimizer, loss_fn, device)
        val_loss = val(model, val_data, loss_fn, device)
        
        if epoch % 5 == 0:
            print(f"< Epoch : {epoch} ||  Loss : {loss} || <<>> Val-Loss : {val_loss}")
        # Check for early stopping and save the best model
        if val_loss + delta < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'hpo_models/hpo_model.pt')  # Save the best model
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Training stopped.')
                break

    model.load_state_dict(torch.load(f"hpo_models/hpo_model.pt"))
    val_loss = val(model, val_data, loss_fn, device)
    trial.report(val_loss, epoch)

    return val_loss


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_trials = 200

    for dataset in ['List_ACIE_THF',
                    'List_ACIE_THP',
                    'Sunoj_PNAS',
                    'HongAckermann_NatSyn',
                    'SigmanToste_Science']:
        
        for target_column in ['ee', 'ddG']:

            for with_T in [True, False]:

                df, smiles_list, _ = preprocess_data(dataset)
                
                train_df, val_df = train_test_split(df, random_state=1337)
                bs_val = len(val_df)
                
                # val data has to be generated only once
                val_data, _ = my_disc_graph_only_dataloader(val_df,
                                                            smiles_list,
                                                            target_column=target_column,
                                                            bs=bs_val)

                study = optuna.create_study(direction='minimize',
                                            study_name='hpo-enantioselectivity',
                                            pruner=optuna.pruners.MedianPruner())

                study.optimize(lambda trial: objective(trial,
                                                    train_df,
                                                    val_data,
                                                    smiles_list,
                                                    target_column,
                                                    with_T,
                                                    device),
                                                    n_trials=n_trials)
                
                joblib.dump(study, f'hpo_{dataset}.pkl')

                print('Best trial:')
                trial = study.best_trial

                print('Value: ', trial.value)

                print('Params: ')
                for key, value in trial.params.items():
                    print(f"    {key}: {value}")

                best_parameters = study.best_params
                print(best_parameters)
                
                if with_T:
                    with_T = 'Yes'
                else:
                    with_T = 'No'

                # Append best parameters to config file
                config_file = 'config.py'
                if os.path.isfile(config_file):
                    with open(config_file, 'r') as f:
                        config_content = f.read()
                else:
                    config_content = ''
                with open(config_file, 'w') as f:
                    f.write(config_content + f'\nBEST_PARAMETERS_{dataset}_{target_column}_T_{with_T} = {json.dumps(best_parameters)}')



if __name__ == "__main__":
    main()