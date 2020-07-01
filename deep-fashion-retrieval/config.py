# -*- coding: utf-8 -*-

GPU_ID = 0
# TRAIN_BATCH_SIZE = 32
# TEST_BATCH_SIZE = 32
# TRIPLET_BATCH_SIZE = 32

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
TRIPLET_BATCH_SIZE = 64

EXTRACT_BATCH_SIZE = 128
TEST_BATCH_COUNT = 200
NUM_WORKERS = 4
LR = 0.002
MOMENTUM = 0.5
# EPOCH = 10
EPOCH = 40
# DUMPED_MODEL = ""
# DUMPED_MODEL = "model_10_final.pth.tar"
DUMPED_MODEL = "freeze=False/lr=0.001/29_epochs"

LOG_INTERVAL = 10
# LOG_INTERVAL = 500
# DUMP_INTERVAL = 1500  # Currently set to dump after every epoch, so not needed
# TEST_INTERVAL = 600  # Currently set to test after every epoch, so not needed

# DATASET_BASE = r'//content/drive/My Drive/Deep Fashion Retrieval/base'
DATASET_BASE = r'/home/ma02526/ResNet/base'
ENABLE_INSHOP_DATASET = True
# ENABLE_INSHOP_DATASET = False
INSHOP_DATASET_PRECENT = 0.8
IMG_SIZE = 256
CROP_SIZE = 224
INTER_DIM = 512
# CATEGORIES = 20
CATEGORIES = 50
N_CLUSTERS = 50
COLOR_TOP_N = 10
TRIPLET_WEIGHT = 2.0
# TRIPLET_WEIGHT = 0.0
ENABLE_TRIPLET_WITH_COSINE = False  # Buggy when backward...
COLOR_WEIGHT = 0.1
DISTANCE_METRIC = ('euclidean', 'euclidean')
FREEZE_PARAM = True
# FREEZE_PARAM = False
