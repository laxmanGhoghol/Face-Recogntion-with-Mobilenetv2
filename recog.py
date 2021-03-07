from keras.models import load_model
from keras.preprocessing import image
import numpy as np

im = image.load_img('img/priya/1.jpg')
im = image.img_to_array(im)
print(im.shape)
model=load_model('mobilenet_v2.model')

im = image.smart_resize(im, (224,224))
im = np.reshape(im, (1, 224, 224, 3))
print(im.shape)
print(model.predict(im).argmax())
