import math
import pyglet
from pyglet import clock
from pyglet.window import key
import cv2
import numpy as np
from pyglet.gl import *
import scipy.misc
from faceswap import doalign
import argparse
import datetime
import sys
import os
from discgen.interface import DiscGenModel

# Setup
window_height = 800
window_width = 1280

window = pyglet.window.Window(window_width, window_height, resizable=False)
framecount = 0
timecount  = 0
num_sets   = 3
cur_set    = 0
last_aligned_face = None

camera = None
dmodel = None

def setup_camera():
    cam = cv2.VideoCapture(0)
    result1 = cam.set(cv2.CAP_PROP_FRAME_WIDTH,720)
    result2 = cam.set(cv2.CAP_PROP_FRAME_HEIGHT,512)
    result3 = cam.set(cv2.CAP_PROP_FPS,1)
    return cam

def get_camera_image(camera):
    retval,img = camera.read()
    # destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def image_to_texture(img):
    sy,sx,number_of_channels = img.shape
    number_of_bytes = sy*sx*number_of_channels
    img  = np.flipud(img)
    img = img.ravel()
    image_texture = (GLubyte * number_of_bytes)( *img.astype('uint8') )
    # my webcam happens to produce BGR; you may need 'RGB', 'RGBA', etc. instead
    pImg = pyglet.image.ImageData(sx,sy,'BGR',
           image_texture,pitch=sx*number_of_channels)
    return pImg

# layout_happy     = layout('images/happy_0.png', 8, 3/4., window)
# layout_angry     = layout('images/angry_0.png', 8, 2/4., window)
# layout_surprised = layout('images/surprised_0.png', 8, 1/4., window)

def get_aligned(img):
    success, im_resize, rect = doalign.align_face_buffer(img, 256, max_extension_amount=0)
    return im_resize

def write_last_aligned():
    if last_aligned_face is None:
        return
    datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "pipeline/aligned/{}.png".format(datestr)    
    if os.path.exists(filename):
        return
    cv2.imwrite(filename, last_aligned_face)

def get_recon(rawim):
    global dmodel

    mixedim = np.asarray([[rawim[:,:,0], rawim[:,:,1], rawim[:,:,2]]])
    entry = (mixedim / 255.0).astype('float32')
    # print(entry.shape)

    if dmodel is None:
        return None

    encoded = dmodel.encode_images(entry)
    decoded = dmodel.sample_at(encoded)[0]
    decoded_array = (255 * np.dstack(decoded)).astype(np.uint8)
    return decoded_array

# Draw Loop
def draw(dt):
    global window, framecount, timecount
    global camera, last_aligned_face
    window.clear()

    # layout_happy.draw()
    # layout_angry.draw()
    # layout_surprised.draw()
    img = get_camera_image(camera)
    small_im = scipy.misc.imresize(img, 0.35)
    # tex = image_to_texture(small_im)
    # window.clear()
    # tex.blit(0,window_height - small_im.shape[0])

    # fake_im = cv2.imread("inputs/allison.png", cv2.IMREAD_COLOR)
    # print(fake_im.shape)

    align_im = get_aligned(img)
    if align_im is not None:
        last_aligned_face = align_im

    if last_aligned_face is not None:
        align_tex = image_to_texture(last_aligned_face)
        align_tex.blit(0,window_height - small_im.shape[0] - last_aligned_face.shape[0] - 20)

        recon = get_recon(last_aligned_face)
        if recon is not None:
            recon_tex = image_to_texture(recon)
            recon_tex.blit(0, 0)

    framecount += pyglet.clock.get_fps()
    timecount  += dt
    return

@window.event
def on_key_press(symbol, modifiers):
    print("SO: {}".format(symbol))
    if(symbol == key.LEFT):
        print("LEFT")
    elif(symbol == key.SPACE):
        print("SPACEBAR")
        write_last_aligned();
    elif(symbol == key.ESCAPE):
        print("ESCAPE")
        cv2.destroyAllWindows()
        cv2.VideoCapture(0).release()
        sys.exit(0)

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Let get NIPSy')
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    args = parser.parse_args()

    global camera
    camera = setup_camera()

    if args.model is not None:
        dmodel = DiscGenModel(filename=args.model)        

    clock.schedule(draw)
    pyglet.app.run()
