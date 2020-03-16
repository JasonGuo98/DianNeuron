USING_CUPY = False
USING_NUMPY = False
# try:
#     import cupy as np
#     print("import cupy as np")
#     USING_CUPY = True
# except:
#     import numpy as np
#     print("import numpy as np")
#     USING_NUMPY = True
import numpy as np
print("import numpy as np")
USING_NUMPY = True

__all__ = ["np","USING_NUMPY","USING_CUPY"]

def set_random_seed(seed):
    if USING_CUPY:
        np.random.seed(seed)
        pass
    if USING_NUMPY:
        np.random.seed(seed)
