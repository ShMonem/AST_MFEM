USE_TAICHI = False
USE_CUPY = False
if USE_TAICHI and USE_CUPY:
    raise(Exception("Don't be greedy!\nCannot use CUPY and TAICHI at the same time!"))
USE_SVD = True
SHRINKAGE_FACTOR = 100
NO_PROGRESS_STREAK_THRESHOLD = 10
MODEL_PATH="/Users/hcsong/Desktop/AST_MFEM/data/beam.mesh"
SKELETON_PATH="/Users/hcsong/Desktop/AST_MFEM/data/beam_skeleton.npy"
HIERARCHY_PATH="/Users/hcsong/Desktop/AST_MFEM/data/beam_hierarchy.npy"
MU=50