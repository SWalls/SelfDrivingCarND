from keras.models import load_model
import keras.utils as ut

model = load_model("model-a.h5")
ut.plot_model(model, to_file='model.png')