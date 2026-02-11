import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

OLD_MODEL = "ultimate_holistic_model.keras"
NEW_MODEL = "models/ultimate_holistic_model_fixed.keras"

old = load_model(OLD_MODEL, compile=False)

inp = Input(shape=(48,48,1))
out = old(inp)

new_model = Model(inputs=inp, outputs=out)
new_model.save(NEW_MODEL)

print("✅ Model converted and saved as:", NEW_MODEL)
