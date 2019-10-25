import json
import csv
import numpy as np
from config import cfg

with open(cfg.data.train_sample) as json_file:
    data = json.load(json_file)
    pass

with open("data/host-a004_lidar1_1232815252301696606.bin", "rb") as f:
    a = np.fromfile(f, dtype=np.int)
    pass

with open(cfg.data.train_csv, 'rt') as f:
    reader = csv.reader(f, delimiter=" ")
    for row in reader:
        print(row)