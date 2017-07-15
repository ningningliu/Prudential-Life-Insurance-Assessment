"""
paths for project and relevant files
"""

import os

PROJECT_PATH = "zhanghan/data"

TRAIN_PATH = os.path.join(PROJECT_PATH, "train.csv")
TEST_PATH = os.path.join(PROJECT_PATH, "test.csv")
SUBMISSION_PATH =os.path.join(PROJECT_PATH, "submission.csv")

TRAIN_TREE = os.path.join(PROJECT_PATH, "train_tree.csv")
TEST_TREE = os.path.join(PROJECT_PATH, "test_tree.csv")

TRAIN_REGRESS = os.path.join(PROJECT_PATH, "train_regress.csv")
TEST_REGRESS = os.path.join(PROJECT_PATH, "test_regress.csv")
