import numpy as np
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
