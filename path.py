"""
paths for project and relevant files
"""

import os

path = os.getcwd()
PROJECT_PATH = path + '/data'

TRAIN_PATH = os.path.join(PROJECT_PATH, "train.csv")
TEST_PATH = os.path.join(PROJECT_PATH, "test.csv")
SUBMISSION_PATH = os.path.join(PROJECT_PATH, "submission.csv")

PARSED_TRAIN = os.path.join(PROJECT_PATH, "parsed_train.csv")
PARSED_TEST = os.path.join(PROJECT_PATH, "parsed_test.csv")

TRAIN_NORMAL = os.path.join(PROJECT_PATH, "parsed_train_normalized.csv")
TEST_NORMAL = os.path.join(PROJECT_PATH, "parsed_test_normalized.csv")
