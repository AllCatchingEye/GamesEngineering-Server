import torch


def load_and_apply_parameters_to_model(
    model: torch.nn.Module, model_path: str, device: torch.device
):
    params = torch.load(model_path, map_location=device)
    assert isinstance(params, dict), (
        "The path "
        + model_path
        + " does not contain the required parameters for that model."
    )
    model.load_state_dict(params)


def persist_model_parameters(model: torch.nn.Module, model_path: str):
    torch.save(
        model.state_dict(),
        model_path,
    )


def as_params_file(name: str) -> str:
    return "%s.pth" % name
