# USAGE
# Start the server:
# 	python run_keras_server.py
import numpy as np
import flask
from keras.models import load_model
import keras.preprocessing.sequence

from sklearn.preprocessing import MinMaxScaler
from itertools import groupby
from rdp import rdp

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def switch_xy(strokes):
    strokes_output = []
    for stroke in strokes_input:
        x = []
        y = []
        for point in stroke:
            x.append(point[0])
            y.append(point[1])

        strokes_output.append([x, y])
    
    return strokes_output

def normalize_strokes(strokes):
    min_x = 9999
    max_x = 0
    min_y = 9999
    max_y = 0

    # find x and y min
    # TODO make in like not retard
    for points in strokes:
        if min_x > min(points[0]):
            min_x = min(points[0])
        if min_y > min(points[1]):
            min_y = min(points[1])
    
    # TODO make in like not retard
    tmp = []
    for points in strokes:
        x = [x - min_x for x in points[0]]
        y = [y - min_y for y in points[1]]
        
        if max_x < max(x):
            max_x = max(x)
        if max_y < max(y):
            max_y = max(y)

        strokes.append([x, y])

    max_max = max([max_x, max_y])

    for points in strokes:
        x = [int(round(x / max_max * 255)) for x in points[0]]
        y = [int(round(x / max_max * 255)) for x in points[1]]
        strokes_output.append([x, y])

    return strokes_output


    def rdp_wrapper(strokes):
        points_switched = []
        for points in strokes:
            tmp = []
            for i in range(0, len(points[0])):
                tmp.append([points[0][i], points[1][i]])
            points_switched.append(tmp)

        tmp = []
        for points in points_switched:
            tmp.append([point[0] for point in groupby(points)])

        strokes_rdp = []
        for stroke in strokes_grouped:
            strokes_rdp.append(rdp(stroke, epsilon=2))

    return switch_xy(strokes_rdp)


def preprocess_strokes(strokes_input):
    strokes_coords_switched = switch_xy(strokes_input)
    strokes_coords_normalized = normalize_strokes(strokes_coords_switched)
    strokes_rdp = rdp_wrapper(strokes_coords_normalized)

    inkarray = strokes_rdp
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
    
    np_ink = np_ink.reshape(1, np_ink.shape[0], np_ink.shape[1])
    np_ink = keras.preprocessing.sequence.pad_sequences(np_ink, maxlen=100, dtype='float32', padding='post', truncating='post')
    np_ink = np_ink.reshape(1, np_ink.shape[1], np_ink.shape[2], 1)

    return np_ink
    
def load_model_keras():
    global model
    model = load_model("quickdraw.h5")
    model.predict(np.zeros((1, 100, 3, 1)))

@app.route("/predict", methods=["POST"])
def predict():

    data = {"success": False}
    strokes = flask.request.get_json()["strokes"]
    inks = preprocess_strokes(strokes)
    predictions = model.predict(inks)

    # TODO load from file
    classes={0:'circle', 1:'cloud', 2:'hexagon', 3:'line', 4:'octagon', 5:'square', 6:'triangle', 7:'zigzag'}

    pred_human = {}
    i = 0
    for x in np.nditer(predictions):
        pred_human[classes[i]] = float(x)
        i = i + 1

    print(pred_human)
        
    data["predictions"] = pred_human
    data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model_keras()
	app.run(host='0.0.0.0')
