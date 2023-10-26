import torch
from tqdm.auto import tqdm
from utils import accuracy_fn

device = "cuda" if torch.cuda.is_available() else "cpu"
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module):

    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()
    for _, (X, y) in enumerate(data_loader):
        
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=torch.argmax(torch.softmax(y_pred, dim=1), dim=1)) # Go from logits-> prob -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    # print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):

    test_loss, test_acc = 0, 0
    model.to(device)

    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for _, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(y_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=torch.argmax(torch.softmax(y_pred, dim=1), dim=1))


        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        # print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

        return test_loss, test_acc


def bert_train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               accuracy_fn = accuracy_fn,
               device: torch.device = device):

    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()
    for _, data in enumerate(data_loader):
        # Send data to GPU
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['label'].to(device)

        # 1. Forward pass
        y_pred = model(input_ids, attention_mask=attention_mask)

        # 2. Calculate loss
        loss = loss_fn(y_pred, labels)
        train_loss += loss
        train_acc += accuracy_fn(y_true=labels,
                                 y_pred=torch.argmax(torch.softmax(y_pred, dim=1), dim=1)) # Go from logits-> prob -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    # print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    return train_loss, train_acc


def bert_test_step(model: torch.nn.Module,
                    data_loader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    accuracy_fn = accuracy_fn,
                    device: torch.device = device):

    test_loss, test_acc = 0, 0
    model.to(device)

    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for _, data in enumerate(data_loader):
          # Send data to GPU
          input_ids = data['input_ids'].to(device)
          attention_mask = data['attention_mask'].to(device)
          labels = data['label'].to(device)

          # 1. Forward pass
          y_pred = model(input_ids, attention_mask=attention_mask)

          # 2. Calculate loss
          loss = loss_fn(y_pred, labels)
          test_loss += loss
          test_acc += accuracy_fn(y_true=labels,
                                  y_pred=torch.argmax(torch.softmax(y_pred, dim=1), dim=1)) # Go from logits-> prob -> pred labels


        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        # print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

        return test_loss, test_acc



def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int = 5,
          bert: bool = False):

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        if not bert:
            train_loss, train_acc = train_step(model=model, 
                                               data_loader=train_dataloader, 
                                               optimizer=optimizer, 
                                               loss_fn = loss_fn)
            
            test_loss, test_acc = test_step(model=model, 
                                            data_loader=test_dataloader, 
                                            loss_fn = loss_fn)
        else:
            train_loss, train_acc = bert_train_step(model=model, 
                                                    data_loader=train_dataloader, 
                                                    optimizer=optimizer, 
                                                    loss_fn = loss_fn)
            
            test_loss, test_acc = bert_test_step(model=model, 
                                                 data_loader=test_dataloader, 
                                                 loss_fn = loss_fn)


        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        # Update results dictionary
        results["train_loss"].append(train_loss.cpu().detach())
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss.cpu().detach())
        results["test_acc"].append(test_acc)


    return results