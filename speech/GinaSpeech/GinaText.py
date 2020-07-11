## GinaText.py
## Gina Module - Speech Intelligent Agent
## www.rcalix.com
## Ricardo A. Calix, Ph.D.
#####################################################################################

import tempfile
import queue
import sys
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
import shutil
import scipy
import matplotlib
import matplotlib.pyplot as plt
import random
import string

##################################################################################

class GinaText():

    def __init__(self):
        self.hello = "hello"

    #################################################################################

    def resolve_predicted_text(self, list_of_letters):
        new_list = []
        for letter in list_of_letters:
            if "0" not in letter:
                new_list.append(letter)
        return new_list

    ######################################################################

