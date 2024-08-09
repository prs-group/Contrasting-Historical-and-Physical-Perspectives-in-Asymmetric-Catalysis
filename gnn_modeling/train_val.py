import torch

def train(model, graph_loader, optimizer, loss_fn, device):
    model.train()
    loss_all = 0
    # Enumerate over the data
    for data_graph in graph_loader:
        # Use GPU
        data_graph.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        mu = model(data_graph.x, data_graph.edge_index, data_graph.batch, data_graph.temp)

        # Calculating the loss and gradients
        loss = loss_fn(mu, data_graph.y)  
        loss_all += loss.item() 
        loss.backward()
        # Update using the gradients
        optimizer.step()

    return loss_all / len(graph_loader)

def val(model, graph_loader, loss_fn, device):
    loss_all = 0
    model.eval()
    for data_graph_val in graph_loader:
        with torch.no_grad():
            data_graph_val.to(device)
            mu = model(data_graph_val.x, data_graph_val.edge_index, data_graph_val.batch, data_graph_val.temp)

            # Calculating the loss and gradients
            loss = loss_fn(mu, data_graph_val.y)  
            loss_all += loss.item() 

    return loss_all / len(graph_loader)

def get_predictions(model, graph_loader, device):
    model.eval()
    prediction = []
    trues = []
    for data_graph_val in graph_loader:
        with torch.no_grad():
            data_graph_val.to(device)
            pred = model(data_graph_val.x, data_graph_val.edge_index, data_graph_val.batch, data_graph_val.temp)
            prediction.extend(pred.cpu().detach().numpy().flatten())
            trues.extend(data_graph_val.y.cpu().detach().numpy().flatten())

    return prediction, trues

def run_epochs(model,
               train_loader,
               val_loader,
               optimizer,
               loss_fn,
               device,
               patience,
               delta,
               dataset,
               rs,
               with_T,
               tc,
               verbose=False
               ):
    if with_T:
        T = 'Yes'
    else:
        T = 'No'
        
    best_loss = float('inf')
    counter = 0
    torch.save(model.state_dict(), f'n_fold_models/{dataset}_{tc}_{T}_{rs}.pt')  # Save the best model
    for epoch in range(5000):
        loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = val(model, val_loader, loss_fn, device)
        
        if epoch % 5 == 0:
            if verbose:
                print(f"< Epoch : {epoch} ||  Loss : {loss} || <<>> Val-Loss : {val_loss}")
        # Check for early stopping and save the best model
        if val_loss + delta < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), f'n_fold_models/{dataset}_{tc}_{T}_{rs}.pt')  # Save the best model
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Training stopped.')
                break

    val_loss = val(model, val_loader, loss_fn, device)
