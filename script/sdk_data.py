import os
import json
from pprint import pprint
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from lyft_dataset_sdk.lyftdataset import LyftDataset
from config import cfg


lyft_data = LyftDataset(
    data_path=cfg.data.base,
    json_path=cfg.data.train_path, 
    verbose=True
)

lyft_data.list_scenes()
# lyft_data.list_sample("first_sample_token")

my_scene = lyft_data.scene[0]
print(my_scene)
my_sample_token = my_scene["first_sample_token"]
lyft_data.render_sample(my_sample_token, out_path=cfg.data.base)

