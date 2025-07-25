import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def plot_loss_curves(results: dict):
    """
    Plots training and testing loss and accuracy curves.

    Args:
        results (dict): A dictionary containing the keys 'train_loss', 'train_acc',
                        'test_loss', and 'test_acc' with values as lists of floats
                        for each epoch.

    Returns:
        None. Displays the loss and accuracy plots.
    """

    # Just in case the results are not appended properly
    train_loss = results.get('train_loss', [])
    test_loss = results.get('test_loss', [])
    train_acc = results.get('train_acc', [])
    test_acc = results.get('test_acc', [])

    epochs = range(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def write_tensorboard(results: dict, writer: SummaryWriter, epoch_offset: int = 0):
    """
    Writes training and testing metrics to TensorBoard.

    Args:
        results (dict): A dictionary with keys 'train_loss', 'train_acc',
                        'test_loss', and 'test_acc' — each a list of floats.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer instance.
        epoch_offset (int, optional): Epoch number to start logging from (useful for resuming training).

    Returns:
        None. Writes logs to TensorBoard.
    """
    for epoch in range(len(results['train_loss'])):
        writer.add_scalar('Loss/Train', results['train_loss'][epoch], epoch + epoch_offset)
        writer.add_scalar('Loss/Test', results['test_loss'][epoch], epoch + epoch_offset)
        writer.add_scalar('Accuracy/Train', results['train_acc'][epoch], epoch + epoch_offset)
        writer.add_scalar('Accuracy/Test', results['test_acc'][epoch], epoch + epoch_offset)
