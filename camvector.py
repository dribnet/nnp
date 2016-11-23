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
from plat.utils import get_json_vectors
from PIL import Image

# Setup
window_height = 800
window_width = 1280

window = pyglet.window.Window(window_width, window_height, resizable=False)
framecount = 0
timecount  = 0
num_sets   = 3
cur_set    = 0
last_aligned_face = None
last_recon_face = None

camera = None
dmodel = None
vector_offsets = None
debug_input = None
debug_outputs = False

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

def write_last_aligned(debugfile=False):
    if last_aligned_face is None:
        return
    if debugfile:
        datestr = "debug"
    else:
        datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "pipeline/aligned/{}.png".format(datestr)
    if not debugfile and os.path.exists(filename):
        return
    cv2.imwrite(filename, last_aligned_face)

    if last_recon_face is None:
        return
    filename = "pipeline/recon/{}.png".format(datestr)    
    if not debugfile and os.path.exists(filename):
        return
    cv2.imwrite(filename, last_recon_face)

def get_recon(rawim):
    global dmodel

    mixedim = np.asarray([[rawim[:,:,2], rawim[:,:,1], rawim[:,:,0]]])
    # mixedim = np.asarray([[rawim[:,:,0], rawim[:,:,0], rawim[:,:,0]]])
    # entry = mixedim[0:3, 0:0+256, 0:0+256]
    entry = (mixedim / 255.0).astype('float32')

    # out = np.dstack(entry[0])
    # out = (255 * out).astype(np.uint8)
    # print(entry.shape, out.shape)
    # outim = Image.fromarray(out)
    # outim.save("debug_passthrough.png")
    # print(entry)
    # print(entry.shape)

    if dmodel is None:
        return None

    encoded = dmodel.encode_images(entry)

    if vector_offsets is not None:
        deblur_vector = smile_offsets[0]
        anchor_index = 0
        smile_vector = smile_offsets[anchor_index+1]
        smile_score = np.dot(smile_vector, encoded)
        smile_detected = (smile_score > 0)
        print("Attribute vector detector for {}: {} {}".format(anchor_index, smile_score, smile_detected))

        if do_smile is not None:
            apply_smile = str2bool(do_smile)
        else:
            apply_smile = not smile_detected

        if apply_smile:
            print("Adding attribute {}".format(anchor_index))
            chosen_anchor = [encoded, encoded + smile_vector + deblur_vector]
        else:
            print("Removing attribute {}".format(anchor_index))
            chosen_anchor = [encoded, encoded - smile_vector + deblur_vector]
    else:
        chosen_anchor = encoded

    decoded = dmodel.sample_at(encoded)[0]
    print(decoded.shape)
    # RGB -> BGR?
    decoded = np.array([decoded[2], decoded[1], decoded[0]])
    decoded_array = (255 * np.dstack(decoded)).astype(np.uint8)
    return decoded_array

# Draw Loop
def draw(dt):
    global window, framecount, timecount
    global camera, last_aligned_face, last_recon_face
    global debug_outputs
    window.clear()

    if debug_input is not None:
        img = debug_input
    else:
        img = get_camera_image(camera)

    align_im = get_aligned(img)
    if align_im is not None:
        last_aligned_face = align_im

    if last_aligned_face is not None:
        align_tex = image_to_texture(last_aligned_face)
        align_tex.blit(0,window_height - last_aligned_face.shape[0] - 20)

        recon = get_recon(last_aligned_face)
        if recon is not None:
            last_recon_face = recon
            recon_tex = image_to_texture(recon)
            recon_tex.blit(0, 0)

    if debug_outputs:
        write_last_aligned(debugfile=True)

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
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--debug-input', dest='debug_input', default=None,
                        help="use this input image instead of camera")
    parser.add_argument('--debug-outputs', dest='debug_outputs', default=False, action='store_true',
                        help="write diagnostic output files each frame")
    args = parser.parse_args()

    debug_outputs = args.debug_outputs
    if args.debug_input is not None:
        debug_input = cv2.imread(args.debug_input, cv2.IMREAD_COLOR)
    else:
        camera = setup_camera()

    if args.model is not None:
        dmodel = DiscGenModel(filename=args.model)        

    if args.anchor_offset is not None:
        anchor_indexes = "0,1,2,3"
        offsets = get_json_vectors(args.anchor_offset)
        dim = len(offsets[0])
        offset_indexes = anchor_indexes.split(",")
        vector_offsets = [ -1 * offset_from_string(offset_indexes[0], offsets, dim) ]
        for i in range(len(offset_indexes) - 1):
            vector_offsets.append(offset_from_string(offset_indexes[i+1], offsets, dim))

    clock.schedule(draw)
    pyglet.app.run()
