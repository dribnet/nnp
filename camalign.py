import math
import pyglet
from pyglet import clock
from pyglet.window import key
import cv2
import numpy as np
from pyglet.gl import *
import scipy.misc
from faceswap import doalign

# Setup
window_height = 800
window_width = 1200


window = pyglet.window.Window(window_width, window_height, resizable=False)
framecount = 0
timecount  = 0
num_sets   = 3
cur_set    = 0
last_aligned_face = None

camera=cv2.VideoCapture(0)
result1 = camera.set(cv2.CAP_PROP_FRAME_WIDTH,720)
result2 = camera.set(cv2.CAP_PROP_FRAME_HEIGHT,512)
result3 = camera.set(cv2.CAP_PROP_FPS,1)

def get_camera_image(camera):
    retval,img = camera.read()
    destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return destRGB

def image_to_texture(img):
    sy,sx,number_of_channels = img.shape
    number_of_bytes = sy*sx*number_of_channels
    img  = np.flipud(img)
    img = img.ravel()
    image_texture = (GLubyte * number_of_bytes)( *img.astype('uint8') )
    # my webcam happens to produce BGR; you may need 'RGB', 'RGBA', etc. instead
    pImg = pyglet.image.ImageData(sx,sy,'RGB',
           image_texture,pitch=sx*number_of_channels)
    return pImg

# layout_happy     = layout('images/happy_0.png', 8, 3/4., window)
# layout_angry     = layout('images/angry_0.png', 8, 2/4., window)
# layout_surprised = layout('images/surprised_0.png', 8, 1/4., window)

def get_aligned(img):
    success, im_resize, rect = doalign.align_face_buffer(img, 256, max_extension_amount=0)
    return im_resize

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
    tex = image_to_texture(small_im)
    # window.clear()
    tex.blit(0,window_height - small_im.shape[0])

    # fake_im = cv2.imread("inputs/allison.png", cv2.IMREAD_COLOR)
    # print(fake_im.shape)

    align_im = get_aligned(img)
    if align_im is not None:
        last_aligned_face = align_im

    if last_aligned_face is not None:
        align_tex = image_to_texture(last_aligned_face)
        align_tex.blit(0,0)

    framecount += pyglet.clock.get_fps()
    timecount  += dt
    return

def change_set(amount):
    global cur_set, num_sets
    cur_set += amount

    if(cur_set > num_sets):
        cur_set = (cur_set-1) % num_sets
    elif(cur_set < 0):
        cur_set = cur_set%num_sets + 1
    
    # print "Switching to set " + str(cur_set)

    # layout_happy.change_image('images/happy_' + str(cur_set) + '.png', 8)
    # layout_angry.change_image('images/angry_' + str(cur_set) + '.png', 8)
    # layout_surprised.change_image('images/surprised_' + str(cur_set) + '.png', 8)

@window.event
def on_key_press(symbol, modifiers):
    if(symbol == key.LEFT):
        change_set(-1)
    elif(symbol == key.RIGHT):
        change_set(1)

clock.schedule(draw)
pyglet.app.run()
