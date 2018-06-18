import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import groupby
from rdp import rdp

def preprocess_strokes(a):
#a = [[[516,513,510,503,495,485,475,463,451,438,426,415,404,395,386,380,374,370,368,366,363],[69,76,83,95,110,131,154,180,208,238,268,296,323,348,372,389,406,418,426,432,436],[0,31,47,63,80,96,112,129,145,162,182,195,212,229,246,261,281,295,312,329,380]],[[519,521,524,528,531,536,542,548,553,558,564,571,579,588,597,606,614,620,625,628,630,631,631],[75,80,87,102,119,138,163,188,214,241,266,292,318,343,369,392,412,428,439,447,452,457,457],[850,930,946,961,979,996,1012,1029,1045,1062,1080,1096,1112,1129,1146,1162,1179,1195,1212,1229,1245,1261,1280]],[[364,371,378,389,402,419,437,458,480,502,525,546,566,581,595,609,609],[447,451,451,452,454,456,459,461,463,464,465,466,467,467,467,468,468],[2299,2346,2363,2380,2395,2412,2429,2446,2462,2479,2495,2512,2529,2545,2562,2578,2596]]]
  b = []
  c = []
  d = []
  e = []
  f = []
  g = []

  min_x = 9999
  max_x = 0
  min_y = 9999
  max_y = 0

  print("*a", a)

for nieco in a:
    if min_x > min(nieco[0]):
        min_x = min(nieco[0])
    if min_y > min(nieco[1]):
        min_y = min(nieco[1])
    
for nieco in a:
    x = [x - min_x for x in nieco[0]]
    y = [y - min_y for y in nieco[1]]
    
    if max_x < max(x):
        max_x = max(x)
    if max_y < max(y):
        max_y = max(y)

    b.append([x, y])

print("*b", b)

max_max = max([max_x, max_y])

for nieco in b:
    x = [int(round(x / max_max * 255)) for x in nieco[0]]
    y = [int(round(x / max_max * 255)) for x in nieco[1]]
    c.append([x, y])

print("*c", c)

for nieco in c:
    tmp = []
    for i in range(0, len(nieco[0])):
        tmp.append([nieco[0][i], nieco[1][i]])
    d.append(tmp)

print("*d", d)

for nieco in d:
    e.append([x[0] for x in groupby(nieco)])

print("*e", e)

for nieco in e:
    f.append(rdp(nieco, epsilon=2))

print("*f", f)


for nieco in f:
  x = []
  y = []
  for point in nieco:
    x.append(point[0])
    y.append(point[1])
  g.append([x, y])

print("*g", g)


inkarray = g
print(len(inkarray))
stroke_lengths = [len(stroke[0]) for stroke in inkarray]
total_points = sum(stroke_lengths)
np_ink = np.zeros((total_points, 3), dtype=np.float32)
current_t = 0
for stroke in inkarray:
  for i in [0, 1]:
    np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
  
  current_t += len(stroke[0])
  np_ink[current_t - 1, 2] = 1

lower = np.min(np_ink[:, 0:2], axis=0)
upper = np.max(np_ink[:, 0:2], axis=0)
scale = upper - lower
scale[scale == 0] = 1
np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale

np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
np_ink = np_ink[1:, :]

print(np_ink)
