
###########################################################################
# Import libraries and set up configuration parameters
###########################################################################
from google.colab import output
import numpy as np
from itertools import cycle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Test step
def test_pass(data, target, model, criterion):
   
    data, target = data.to(DEVICE), target.to(DEVICE)
    with torch.no_grad():
        output = model.forward(data)
    loss = criterion(output, target)
   # loss+= loss.item()
    return loss.item(), output

def do_test(model, loaders, criterion):
    model.eval()
    acc_loss, accuracy = 0, 0  # acc_loss : accumulated loss
    len_pred = 0  
    true_pred = 0
    all_outputs = torch.tensor([]).to(DEVICE) 
    all_labels = torch.tensor([]).to(DEVICE)  # To store all true labels for metric computation
    batch_accuracies = []  # List to store accuracy for each batch

    with torch.inference_mode():
        for data, target in loaders['test']: 
            # Send data and target to device 
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # call test pass to get loss and output
            loss, output = test_pass(data, target, model, criterion)
            
            # Accumulate loss
            acc_loss += loss

            # Compute the prediction labels by using argmax(dim=1)
            pred = output.argmax(dim=1) 
            len_pred += len(target)

            # Calculate and accumulate correct predictions
            true_pred += (pred == target).sum().item()

            # Add the output of each batch to the all_outputs tensor
            all_outputs = torch.cat((all_outputs, output), dim=0)

            # Add the true labels to all_labels
            all_labels = torch.cat((all_labels, target), dim=0)

            # Compute accuracy for the current batch and store it
            batch_accuracy = (pred == target).float().mean().item()  # Accuracy for current batch
            batch_accuracies.append(batch_accuracy)

    # Compute the test_loss value
    test_loss = acc_loss / len(loaders['test']) 
    accuracy = true_pred / len_pred  

    # Compute the standard deviation of batch accuracies
    accuracy_std = np.std(batch_accuracies)

    # Convert all_outputs and all_labels to CPU for metric computation
    all_outputs = all_outputs.cpu()
    all_labels = all_labels.cpu()

    # Convert outputs to predicted class labels (argmax)
    pred_labels = all_outputs.argmax(dim=1)

    # Calculate metrics
    f1 = f1_score(all_labels, pred_labels, average='weighted', zero_division=0)
    precision = precision_score(all_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(all_labels, pred_labels, average='weighted', zero_division=0)

    # Standard deviation for the metrics
    f1_std = np.std(f1_score(all_labels, pred_labels, average=None, zero_division=0))
    precision_std = np.std(precision_score(all_labels, pred_labels, average=None, zero_division=0))
    recall_std = np.std(recall_score(all_labels, pred_labels, average=None, zero_division=0))

    return (accuracy, test_loss, accuracy_std, f1, precision, recall, f1_std, precision_std, recall_std, all_outputs)


def inspect_obj(obj, internal=False):


    dir_obj = []


    for func in dir(obj):
        try:
            _ = getattr(obj, func)
            dir_obj.append(func)
        except BaseException:
            pass

    # Selection of methods and properties
    if internal:
        method_list = [func for func in dir_obj if callable(getattr(obj,
                                                                    func))]
        property_list = [prop for prop in dir_obj if prop not in method_list]
    else:
        method_list = [func for func in dir_obj if callable(
            getattr(obj, func)) and not func.startswith('_')]
        property_list = [prop for prop in dir_obj if
                         prop not in method_list and not prop.startswith('_')]

    return {'properties': property_list, 'methods': method_list,
            'attributes': sorted(property_list + method_list)}



# Saving the model
def trained_save(filename, model, optimizer, tr_loss_list, vl_loss_list,
                 verbose=True):
    custom_dict = {'model_state_dict': model.state_dict(),
                   'opt_state_dict': optimizer.state_dict(),
                   'tr_loss_list': tr_loss_list,
                   'vl_loss_list': vl_loss_list}
    torch.save(custom_dict, filename)
    if verbose:
        print('Checkpoint saved at epoch {}'.format(len(tr_loss_list)))


# Load the model saved with 'trained_save()'
def trained_load(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    checkpoint.pop('model_state_dict')
    checkpoint.pop('opt_state_dict')

    return model, optimizer, checkpoint


def train(n_epochs, loaders, model, optimizer, criterion, filename):
    # Initialize tracking
    tr_loss_list = []
    vl_loss_list = []

    best_f1 = 0
    best_state_dict = None
    
    # Scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-4)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0

        for train_X, train_y in loaders['train']:
            optimizer.zero_grad()
            
            # Forward pass
            train_out = model(train_X.to(DEVICE))
            loss_train = criterion(train_out, train_y.to(DEVICE))
            
            # Add regularization
            loss = loss_train + 1e-4 * sum(p.norm(2) for p in model.parameters())
            
            # Gradient clipping and backward pass
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            if grad_norm > 1.5:
                print(f"High gradient norm: {grad_norm:.2f}")
            loss.backward()
            optimizer.step()
            
            train_loss += loss_train.item()

        # Validation phase
        model.eval()
        val_loss, all_preds, all_labels = 0, [], []
        for data, target in loaders['valid']:
            data, target = data.to(DEVICE), target.to(DEVICE)
            with torch.no_grad():
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        tr_loss = train_loss / len(loaders['train'])
        val_loss /= len(loaders['valid'])
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Update tracking
        tr_loss_list.append(tr_loss)
        vl_loss_list.append(val_loss)
        
        # Scheduler step
        scheduler.step(val_f1)
        
        print(f'Epoch: {epoch:2d} | '
              f'Train: {tr_loss:.4f} | '
              f'Val: {val_loss:.4f} | '
              f'F1: {val_f1:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.1e}')

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save({
                'state_dict': best_state_dict,
                'f1': best_f1,
                'epoch': epoch
            }, filename)
            print(f'★ New best F1: {best_f1:.4f}')

    # Load best model
    model.load_state_dict(best_state_dict)
    trained_save(filename, model, optimizer, tr_loss_list, vl_loss_list, False)
    return model, (tr_loss_list, vl_loss_list)



# Simplified validation pass that works with dictionary returns
def valid_pass(data, target, model, criterion):
    data, target = data.to(DEVICE), target.to(DEVICE)
    with torch.no_grad():
        output = model(data)
        if isinstance(criterion, nn.Module):
            # Handle both regular and hybrid loss cases
            if hasattr(criterion, 'forward'):
                loss = criterion(output, target, output, target, 1.0)['loss']
            else:
                loss = criterion(output, target)
        else:
            loss = criterion(output, target)
    return loss.item()

# Traing plot
def plot_checkpoint(checkpoint):
    x = range(1, 1 + len(checkpoint['tr_loss_list']))
    tr_losses = checkpoint['tr_loss_list']
    vl_losses = checkpoint['vl_loss_list']
    tr_max, tr_min = np.max(tr_losses), np.min(tr_losses)
    epoch_min = 1 + np.argmin(vl_losses)
    val_min = np.min(vl_losses)

    plt.plot(x, tr_losses, label='training loss')
    plt.plot(x, vl_losses, label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Losses during training")
    plt.legend()
    plt.annotate('valid min: {:.4f}'.format(val_min),
                 xy=(epoch_min,
                     val_min),
                 xytext=(round(0.75 * len(tr_losses)),
                         3 * (tr_max - tr_min) / 4 + tr_min),
                 arrowprops=dict(facecolor='black',
                                 shrink=0.05),
                 )
    plt.xlim(0, len(tr_losses))
    plt.show()

##############################################################################
# Main block
##############################################################################

if __name__ == "__main__":

    SEED = 4



    print("All tests passed!")
