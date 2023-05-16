USE_TAICHI = False
USE_CUPY = False
if USE_TAICHI and USE_CUPY:
    raise(Exception("Don't be greedy!\nCannot use CUPY and TAICHI at the same time!"))
USE_SVD = True
USE_MG = False
DEBUG = True
SHRINKAGE_FACTOR = 2
NO_PROGRESS_STREAK_THRESHOLD = 10