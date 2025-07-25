
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

def train_step(model: torch.nn.Module, dataloader: DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Performs a single training step over an entire DataLoader.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Computation device (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: Average training loss and accuracy over the epoch.
    """
    train_loss = 0
    train_acc = 0
    
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        train_preds = model(X)
        loss = loss_fn(train_preds, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        pred_labels = torch.argmax(train_preds, dim=1)
        acc = (pred_labels == y).sum().item() / len(y)
        train_acc += acc
    
    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model: torch.nn.Module, dataloader: DataLoader, 
              loss_fn: torch.nn.Module, device: torch.device):
    """
    Performs a single evaluation step over an entire DataLoader.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader for test/validation data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device): Computation device (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: Average test loss and accuracy.
    """
    test_loss = 0
    test_acc = 0
    
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_preds = model(X)
            loss = loss_fn(test_preds, y)
            test_loss += loss.item()

            # Accuracy
            pred_labels = torch.argmax(test_preds, dim=1)
            acc = (pred_labels == y).sum().item() / len(y)
            test_acc += acc

    return test_loss / len(dataloader), test_acc / len(dataloader)


def train(model: torch.nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, 
          epochs: int, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, 
          scheduler: torch.optim.lr_scheduler._LRScheduler = None):
    """
    Trains and evaluates a PyTorch model for a given number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
        test_dataloader (torch.utils.data.DataLoader): Dataloader for test/validation data.
        epochs (int): Number of training epochs.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Computation device (e.g., 'cuda' or 'cpu').

    Returns:
        dict: Dictionary containing training and testing loss and accuracy history.
    """
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

    return results
