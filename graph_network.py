from keras.utils.visualize_util import plot

from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import argparse
import pydot_ng as pydot

pydot.find_graphiz = lambda: True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if not pydot.find_graphiz():
        raise ImportError('Install pydot and graphviz.')
    else:
        plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)
