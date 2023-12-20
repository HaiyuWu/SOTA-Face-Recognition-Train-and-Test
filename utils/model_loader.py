from os import path
import torch
from .utils import get_time


def save_state(full_model, optimizer, config, accuracy, step=0, model_only=False):
    save_path = config.model_path

    torch.save(
        full_model.module.model.state_dict(),
        path.join(
            save_path,
            "model_{}_accuracy;{:.4f}_step;{}.pth".format(get_time(), accuracy, step),
        ),
    )

    if not model_only:
        torch.save(
            full_model.module.head.state_dict(),
            path.join(
                save_path,
                "head_{}_accuracy;{:.4f}_step;{}.pth".format(
                    get_time(), accuracy, step
                ),
            ),
        )
        torch.save(
            optimizer.state_dict(),
            path.join(
                save_path,
                "optimizer_{}_accuracy;{:.4f}_step;{}.pth".format(
                    get_time(), accuracy, step
                ),
            ),
        )
