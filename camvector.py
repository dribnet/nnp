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
from plat.utils import get_json_vectors, offset_from_string
from plat.grid_layout import grid2img
from PIL import Image
from scipy.misc import imread, imsave

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
    retval, img = camera.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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
    imsave(filename, last_aligned_face)

    if last_recon_face is None:
        return
    filename = "pipeline/recon/{}.png".format(datestr)    
    if not debugfile and os.path.exists(filename):
        return
    imsave(filename, last_recon_face)

def encode_from_image(rawim):
    global dmodel
    mixedim = np.asarray([[rawim[:,:,0], rawim[:,:,1], rawim[:,:,2]]])
    entry = (mixedim / 255.0).astype('float32')
    encoded = dmodel.encode_images(entry)[0]
    return encoded

def pr_map(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / float(istop - istart));

def write_atstrip(debugfile=False):
    global dmodel, vector_offsets

    if last_aligned_face is None or dmodel is None or vector_offsets is None:
        return

    if debugfile:
        datestr = "debug"
    else:
        datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "pipeline/atstrip/{}.png".format(datestr)

    encoded = encode_from_image(last_aligned_face)
    decode_list = []
    deblur_vector = vector_offsets[0]
    anchor_index = 0
    attribute_vector = vector_offsets[anchor_index+1]
    for i in range(7):
        scale_factor = pr_map(i, 0, 5, -0.5, 1.5)
        cur_anchor = encoded + scale_factor * attribute_vector + deblur_vector
        decode_list.append(cur_anchor)
    decoded = dmodel.sample_at(np.array(decode_list))
    img = grid2img(decoded, 1, 5, False)
    img.save(filename)

def get_recon(rawim):
    global dmodel, vector_offsets

    if dmodel is None:
        return None

    encoded = encode_from_image(rawim)

    if vector_offsets is not None:
        deblur_vector = vector_offsets[0]
        anchor_index = 0
        attribute_vector = vector_offsets[anchor_index+1]
        chosen_anchor = encoded + attribute_vector + deblur_vector
        # smile_score = np.dot(smile_vector, encoded)
        # smile_detected = (smile_score > 0)
        # print("Attribute vector detector for {}: {} {}".format(anchor_index, smile_score, smile_detected))

    else:
        chosen_anchor = encoded

    decode_list = np.array([chosen_anchor])
    decoded = dmodel.sample_at(decode_list)[0]
    # RGB -> BGR?
    decoded = np.array([decoded[2], decoded[1], decoded[0]])
    decoded_array = (255 * np.dstack(decoded)).astype(np.uint8)
    return decoded_array

def get_recon_strip(rawim):
    global dmodel, vector_offsets

    if dmodel is None or vector_offsets is None:
        return

    encoded = encode_from_image(rawim)
    decode_list = []
    deblur_vector = vector_offsets[0]
    anchor_index = 0
    attribute_vector = vector_offsets[anchor_index+1]
    for i in range(5):
        scale_factor = pr_map(i, 0, 5, -0.5, 1.5)
        cur_anchor = encoded + scale_factor * attribute_vector + deblur_vector
        decode_list.append(cur_anchor)
    decoded = dmodel.sample_at(np.array(decode_list))
    n, c, y, x = decoded.shape
    decoded_strip = np.concatenate(decoded, axis=2)
    decoded_array = (255 * np.dstack(decoded_strip)).astype(np.uint8)
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
        align_tex.blit(window_width / 2 - 128, window_height - last_aligned_face.shape[0])

        recon = get_recon_strip(last_aligned_face)
        if recon is not None:
            last_recon_face = recon
            recon_tex = image_to_texture(recon)
            recon_tex.blit(0, window_height/2 - 128)

    if debug_outputs:
        write_last_aligned(debugfile=True)
        # write_atstrip(debugfile=True)

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
        # write_atstrip();
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
        debug_input = cv2.cvtColor(debug_input, cv2.COLOR_BGR2RGB)
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
