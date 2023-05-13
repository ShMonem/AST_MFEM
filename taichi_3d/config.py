USE_TAICHI = False
USE_CUPY = True
if USE_TAICHI and USE_CUPY:
    raise(Exception("Don't be greedy!\nCannot use CUPY and TAICHI at the same time!"))
USE_SVD = True