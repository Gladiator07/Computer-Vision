import torch
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu0'

# PARAMETERS
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10


def prepare_data():
    # define transforms
    transforms = transforms.Compose([transforms.Resize(32, 32),
                                     transforms.ToTensor()])
    train_data = MNIST(root='/', train=True,
                       transform=transforms, download=True)
    valid_data = MNIST(root='/', train=False,
                       transform=transforms, download=True)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)

    valid_loader = DataLoader(
        dataset=valid_data, batch_size=BATCH_SIZE, shuffle=False)


def train(train_loader, model, loss_fn, optimizer, device):
    """
    Function for the training step of the training loop
    """

    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()
        X = X.to(device=device)
        y_true = y.to(device=device)

        # Forward pass
        y_pred, _ = model(X)
        loss = loss_fn(y_pred, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, optimizer, epoch_loss


def validate(valid_loader, model, loss_fn, device):
    """
    Function for the validation step of the training loop
    """

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:

        X = X.to(device=device)
        y_true = y_true.to(device=device)

        # Forward pass and record loss
        y_pred, _ = model(X)
        loss = loss_fn(y_pred, y_true)

        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)


def training_loop(model, loss_fn, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    """
    Function defining the entire training loop
    """

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(
            train_loader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):

            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)

    return model, optimizer, (train_losses, valid_losses)
