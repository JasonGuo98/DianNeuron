try:
    import cupy as np
    print("use cupy")
except:
    import numpy as np
    print("use numpy")