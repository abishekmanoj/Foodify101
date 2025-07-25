
import os
import torch
from torchinfo import summary


def model_summary(model, batch_size: int, image_size: tuple):
    """
    Prints a detailed model summary including input/output sizes, 
    parameter counts, and trainable status using torchinfo.

    Args:
        model (torch.nn.Module): The PyTorch model to inspect.
        batch_size (int): Batch size used for the input.
        image_size (tuple): A tuple of (channels, height, width), e.g., (3, 224, 224).

    Returns:
        None. Prints the model summary to stdout.
    """
    input_shape = [batch_size] + list(image_size)
    summary(
        model=model,
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    

def save_model(model: torch.nn.Module, target_path: str):
    """
    Saves a PyTorch model's state dictionary to the specified file path.

    Constraints:
        - The parent directory of the save path must start with 'models/'.

    Args:
        model (torch.nn.Module): The model to save.
        target_path (str): Full path to save the model file (e.g., "models/foodify_v1.pth").

    Returns:
        None. Saves the model to disk.

    Raises:
        ValueError: If the parent directory does not start with 'models/'.
    """
    parent_dir = os.path.normpath(os.path.dirname(target_path))

    # Ensure it starts with 'models/'
    if not parent_dir.startswith("models"):
        raise ValueError(f"[ERROR] The target directory must start with 'models/': got '{parent_dir}'")

    os.makedirs(parent_dir, exist_ok=True)
    torch.save(model.state_dict(), target_path)
    print(f"[INFO] Model saved to: {target_path}")
    

def load_model(model: torch.nn.Module, path: str, device: torch.device = torch.device("cpu")):
    """
    Loads a saved state dictionary into a PyTorch model.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        path (str): Path to the saved .pth file (e.g., "models/foodify_v1.pth").
        device (torch.device): Device to map the model to (e.g., torch.device("cuda") or torch.device("cpu")).

    Returns:
        torch.nn.Module: The model with loaded weights.

    Raises:
        FileNotFoundError: If the given path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    model.load_state_dict(torch.load(path, map_location=device))
    print(f"[INFO] Model loaded from: {path}")
    return model
