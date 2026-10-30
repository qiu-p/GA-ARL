
def set_deterministic(seed=42, set_base=True, set_numpy=True, set_cuda=True) -> None:
    #设置随机种子

    if set_base:
        import random
        random.seed(a=seed)

    if set_numpy:
        import numpy as np
        np.random.seed(seed=seed)

    if set_cuda:
        import torch
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.manual_seed(seed=seed)
        
        #设置CUDA确定性计算
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        #设置CUDA随机种子
        torch.cuda.manual_seed(seed=seed)
        torch.cuda.manual_seed_all(seed=seed)
        torch.use_deterministic_algorithms(mode=True)