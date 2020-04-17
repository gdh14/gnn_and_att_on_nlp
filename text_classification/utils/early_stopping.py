import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datetime
import pprint
import dill
import sys


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    
    adopted from: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, model_dir, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_dir = model_dir
        self.ensure_path_exist(self.model_dir)

        
    def ensure_path_exist(self, path):
        """
        Make path if it has not been made yet.
        """
        os.makedirs(path, exist_ok=True)
        
        
    def __call__(self, val_loss, model, model_object=None):

        score = -val_loss
        
        if np.isnan(val_loss):
            print('loss turns to nan, stop')
            self.early_stop = True
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_object=model_object)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_object=model_object)
            self.counter = 0

            
    def save_checkpoint(self, val_loss, model, model_object):
        
        """ Saves model when validation loss decrease.
        Parameters
        ----------
        val_loss: float
            validation loss
            
            
        model: model.TCNBinClfBase
            the low-level TCNBinClf model object
            
            
        model_object: tuple - (name, object)
            any object needed to save
            
            
        Returns
        ----------
        None
        """
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        checkpoint = {}
        checkpoint['model'] = model.state_dict()   
#         checkpoint['model'] = model

        torch.save(checkpoint, '{}/checkpoint.pt'.format(self.model_dir))
        if model_object:
            name = model_object[0]
            name = "{}.dill".format(name) if ".dill" not in name else name
            dill.dump(model_object[1], open(os.path.join(self.model_dir, name), "wb"))
        self.val_loss_min = val_loss