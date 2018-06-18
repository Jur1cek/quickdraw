from os import listdir
import json
import numpy as np

DATA_DIR="quickdraw_data_reduced"

def parse_line(ndjson_line):
  """Parse an ndjson line and return ink (as np array) and classname."""
  sample = json.loads(ndjson_line)
  class_name = sample["word"]
  if not class_name:
    print ("Empty classname")
    return None, None
  inkarray = sample["drawing"]
  stroke_lengths = [len(stroke[0]) for stroke in inkarray]
  total_points = sum(stroke_lengths)
  np_ink = np.zeros((total_points, 3), dtype=np.float32)
  current_t = 0
  if not sample["recognized"]:
    return None, None
  if not inkarray:
    return None, None
  for stroke in inkarray:
    if len(stroke[0]) != len(stroke[1]):
      print("Inconsistent number of x and y coordinates.")
      return None, None
    for i in [0, 1]:
      np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
    current_t += len(stroke[0])
    np_ink[current_t - 1, 2] = 1  # stroke_end
  # Preprocessing.
  # 1. Size normalization.
  lower = np.min(np_ink[:, 0:2], axis=0)
  upper = np.max(np_ink[:, 0:2], axis=0)
  scale = upper - lower
  scale[scale == 0] = 1
  np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
  # 2. Compute deltas.
  np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
  np_ink = np_ink[1:, :]
  return np_ink, class_name

if __name__ == "__main__":
  X = []
  Y = []
  i=0

  files = listdir(DATA_DIR)
  for file in files:
    with open(DATA_DIR + "/" + file) as f:
      content = f.readlines()
    
    for line in content:
      x, y = parse_line(line)
      if x is not None or y is not None:
        X.append(x)
        Y.append(y)
      i = i + 1
  
np.save("X.npy", X)
np.save("Y.npy", Y)
