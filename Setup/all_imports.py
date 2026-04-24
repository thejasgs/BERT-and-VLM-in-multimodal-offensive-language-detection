import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
import os
from PIL import Image
import re
from sklearn import *
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder
from scipy.special import expit 
from datasets import *
from transformers.utils import logging
from sklearn.metrics import *
from colorama import Fore, Style
from datetime import date
import json
from torch.utils.data import *
import gc
import time
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["ONEDNN_VERBOSE"] = "0" # Disable oneDNN verbose logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # Reduce verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable tokenizer parallelism warnings
os.environ["TORCH_COMPILE_DISABLE"] = "1" # Disable because causing erorr and it did not support this feature