import torch

def load_model(model, model_path, device):
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_model_old(model, model_path, device):
    """
    Load the I-VAE model that used deprecated field names.
    """
    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = {}
    rename_mapping = {
        "encoder": "state_encoder",
        "prior": "obs_encoder",
    }
    for old_key in checkpoint.keys():
        new_key = old_key
        for old_name, new_name in rename_mapping.items():
            if old_key.startswith(old_name):
                new_key = old_key.replace(old_name, new_name, 1)
        new_state_dict[new_key] = checkpoint[old_key]

    # Set strict=False in case some keys don't match
    model.load_state_dict(new_state_dict, strict=False)
    return model

