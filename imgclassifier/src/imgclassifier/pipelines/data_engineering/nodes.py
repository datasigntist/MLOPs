import numpy as np
import h5py
from typing import Dict, List
import logging

def load_Data():
    train_dataset = h5py.File('data/01_raw/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/01_raw/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    loaded_Data = {'train_set_x_orig':train_set_x_orig, 
            'train_set_y_orig':train_set_y_orig, 
            'test_set_x_orig':test_set_x_orig, 
            'test_set_y_orig':test_set_y_orig, 
            'classes':classes}
    
    return loaded_Data

def flatten_and_Scale_Data(loaded_Data: Dict):
    train_x_orig = loaded_Data["train_set_x_orig"]
    train_y_orig = loaded_Data["train_set_y_orig"]    
    test_x_orig = loaded_Data["test_set_x_orig"]
    test_y_orig = loaded_Data["test_set_y_orig"]  
    classes = loaded_Data["classes"]

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    logger = logging.getLogger(__name__)
    logger.info("train_x shape %s", str(train_x.shape))
    logger.info("train_y shape %s", str(train_y_orig.shape))
    logger.info("test_x shape %s", str(test_x.shape))
    logger.info("test_y shape %s", str(test_y_orig.shape))


    flattened_and_Scaled_Data = {'train_x':train_x, 
        'train_y':train_y_orig, 
        'test_x':test_x, 
        'test_y':test_y_orig, 
        'classes':classes}


    return flattened_and_Scaled_Data
