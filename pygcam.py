import pyglet
from pyglet.gl import *
from pyglet.window import key
import cv2
import numpy as np
import sys

window = pyglet.window.Window()

camera=cv2.VideoCapture(0)

retval,img = camera.read()
sy,sx,number_of_channels = img.shape
number_of_bytes = sy*sx*number_of_channels

# rotate the image by 180 degrees
print(img.shape)
img  = np.flipud(img)
img = img.ravel()

image_texture = (GLubyte * number_of_bytes)( *img.astype('uint8') )
# my webcam happens to produce BGR; you may need 'RGB', 'RGBA', etc. instead
pImg = pyglet.image.ImageData(sx,sy,'BGR',
       image_texture,pitch=sx*number_of_channels)

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.ESCAPE:
        print 'Application Exited with Key Press'
        window.close()

@window.event
def on_draw():
    window.clear()
    pImg.blit(0,0)

pyglet.app.run()