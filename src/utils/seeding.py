import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The base seed to use across libraries.
        deterministic_torch: If True, configures cuDNN for deterministic behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            # Not all builds expose cuDNN backends; ignore if unavailable
            pass


essential_env_seeded_flag = "_seeded_once"


def seed_env(env, seed: int, seed_spaces: bool = True, only_once: bool = False) -> None:
    """Seed a Gymnasium environment safely.

    Args:
        env: The environment instance.
        seed: Seed value.
        seed_spaces: Attempt to seed action and observation spaces as well.
        only_once: If True, only seed the env once; subsequent calls are no-ops.
    """
    if only_once and getattr(env, essential_env_seeded_flag, False):
        return

    # Seed the environment RNG
    try:
        env.reset(seed=seed)
    except TypeError:
        # Older gym API fallback
        try:
            env.seed(seed)
        except Exception:
            pass

    if seed_spaces:
        for space_name in ("action_space", "observation_space"):
            try:
                space = getattr(env, space_name, None)
                if space is not None:
                    space.seed(seed)
            except Exception:
                # Some spaces may not support seeding
                pass

    if only_once:
        try:
            setattr(env, essential_env_seeded_flag, True)
        except Exception:
            pass
