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

num_vectors   = 4
cur_vector    = 1
do_clear = True

theApp = None
vector_offsets = None
window_height = 800
window_width = 1280
window = pyglet.window.Window(window_width, window_height, resizable=False)
cam_width = 720
cam_height = 512

vector_files = [
    "images/vector_oneshot.png",
    "images/vector_blur.png",
    "images/vector_smile.png",
    "images/vector_surprised.png",
    "images/vector_angry.png",
]

canned_faces = [
    "images/bengio.jpg",
    "images/demis.jpg",
    "images/fei_fei.jpg",
    "images/geoffrey.jpg",
    "images/yann.jpg",
]

def setup_camera():
    cam = cv2.VideoCapture(0)
    result1 = cam.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width)
    result2 = cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_height)
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

def get_aligned(img):
    success, im_resize, rect = doalign.align_face_buffer(img, 256, max_extension_amount=0)
    return im_resize

def encode_from_image(rawim, dmodel):
    mixedim = np.asarray([[rawim[:,:,0], rawim[:,:,1], rawim[:,:,2]]])
    entry = (mixedim / 255.0).astype('float32')
    encoded = dmodel.encode_images(entry)[0]
    return encoded

def pr_map(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / float(istop - istart));

class MainApp():
    last_aligned_face = None
    last_recon_face = None
    last_encoded_vector = None
    one_shot_face = None
    one_shot_source = None
    one_shot_source_vector = None
    input_image = None
    debug_outputs = False
    framecount = 0
    timecount  = 0
    use_camera = True
    camera = None
    model_name = None
    dmodel = None
    one_shot_mode = False
    cur_canned_face = -1;

    """Just a container for unfortunate global state"""
    def __init__(self):
        self.cur_frame = 0
        self.vector_textures = []
        for i in range(len(vector_files)):
            png = Image.open(vector_files[i])
            if png.mode == "RGBA":
                png.load()
                vector_im = Image.new("RGB", png.size, (0, 0, 0))
                vector_im.paste(png, mask=png.split()[3]) # 3 is the alpha channel
            else:
                vector_im = png
            vector_im = np.asarray(vector_im)
            # vector_im = imread(vector_files[i], mode='RGB')
            if i == 0:
                h, w, c = vector_im.shape
                self.vector_x = int((window_width - w) / 2)
                self.vector_y = int((window_height - h) / 2)
            self.vector_textures.append(image_to_texture(vector_im))
        self.cur_canned_face = -1
        self.canned_aligned_faces = []
        for i in range(len(canned_faces)):
            canned_face = imread(canned_faces[i])
            self.canned_aligned_faces.append(get_aligned(canned_face))

    def setDebugOutputs(self, mode):
        self.debug_outputs = mode

    def write_last_aligned(self, debugfile=False):
        if self.last_aligned_face is None:
            return
        if debugfile:
            datestr = "debug"
        else:
            datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "pipeline/aligned/{}.png".format(datestr)
        if not debugfile and os.path.exists(filename):
            return
        imsave(filename, self.last_aligned_face)

        if self.last_recon_face is None:
            return
        filename = "pipeline/recon/{}.png".format(datestr)    
        if not debugfile and os.path.exists(filename):
            return
        imsave(filename, self.last_recon_face)

    def get_recon_strip(self, rawim, dmodel):
        global vector_offsets, cur_vector

        if dmodel is None or vector_offsets is None:
            decode_list = []
            for i in range(5):
                decode_list.append(rawim)
            decoded = np.array(decode_list)
            decoded_array = np.concatenate(decoded, axis=1)
            print(decoded_array.shape)
        else:
            encoded = encode_from_image(rawim, dmodel)
            self.last_encoded_vector = encoded
            decode_list = []
            deblur_vector = vector_offsets[0]
            if self.one_shot_mode:
                if self.one_shot_source_vector is not None:
                    # compute attribute vector
                    attribute_vector = encoded - self.one_shot_source_vector
                else:
                    # smile is debug ?
                    attribute_vector = vector_offsets[1]
                # override encoded to be one_shot_face
                encoded = encode_from_image(self.one_shot_face, dmodel)
            else:
                anchor_index = 0
                attribute_vector = vector_offsets[anchor_index+cur_vector]
            for i in range(5):
                scale_factor = pr_map(i, 0, 5, -1.5, 1.5)
                cur_anchor = encoded + scale_factor * attribute_vector + deblur_vector
                decode_list.append(cur_anchor)
            decoded = dmodel.sample_at(np.array(decode_list))
            n, c, y, x = decoded.shape
            decoded_strip = np.concatenate(decoded, axis=2)
            decoded_array = (255 * np.dstack(decoded_strip)).astype(np.uint8)
        return decoded_array

    def draw(self, dt):
        global window, cur_vector, do_clear

        # clear window only sometimes
        if do_clear:
            window.clear()
            do_clear = False

        # initialize camera and dmodel after warming up
        if self.camera is None and self.use_camera and self.cur_frame > 10:
            print("Initializing camera")
            self.camera = setup_camera()

        if self.dmodel is None and self.model_name and self.cur_frame > 20:
            print("Initializing model {}".format(self.model_name))
            self.dmodel = DiscGenModel(filename=self.model_name)        

        # get source image
        if self.camera:
            img = get_camera_image(self.camera)
        else:
            img = self.input_image

        align_im = get_aligned(img)
        if align_im is not None:
            self.last_aligned_face = align_im

        if self.one_shot_mode:
            vector_index = 0
        else:
            vector_index = cur_vector + 1
        self.vector_textures[vector_index].blit(self.vector_x, self.vector_y)

        if self.last_aligned_face is not None:
            align_tex = image_to_texture(self.last_aligned_face)
            if self.one_shot_mode:
                align_tex.blit(3 * window_width / 4 - 128, int((window_height - 256) / 2))
                one_shot_source_tex = image_to_texture(self.one_shot_source)
                one_shot_source_tex.blit(window_width / 4 - 128, int((window_height - 256) / 2))
                one_shot_face_tex = image_to_texture(self.one_shot_face)
                one_shot_face_tex.blit(window_width / 2 - 128, window_height - 256)
            else:
                align_tex.blit(window_width / 2 - 128, window_height - self.last_aligned_face.shape[0])

            recon = self.get_recon_strip(self.last_aligned_face, self.dmodel)
            if recon is not None:
                self.last_recon_face = recon
                recon_tex = image_to_texture(recon)
                recon_tex.blit(0, 0)

        if self.debug_outputs:
            self.write_last_aligned(debugfile=True)

        self.framecount += pyglet.clock.get_fps()
        self.timecount  += dt
        self.cur_frame += 1
        return

# Draw Loop
def draw(dt):
    theApp.draw(dt)

theApp = MainApp()

@window.event
def on_key_press(symbol, modifiers):
    global cur_vector
    print("SO: {}".format(symbol))
    if(symbol == key.S):
        if theApp.one_shot_mode:
            theApp.one_shot_source = theApp.last_aligned_face
            theApp.one_shot_source_vector = theApp.last_encoded_vector
    if(symbol == key.DOWN):
        if theApp.one_shot_mode:
            theApp.cur_canned_face = (theApp.cur_canned_face - 1 + len(canned_faces)) % len(canned_faces)
            theApp.one_shot_face = theApp.canned_aligned_faces[theApp.cur_canned_face]
        else:
            cur_vector = (cur_vector - 1 + num_vectors) % num_vectors
    if(symbol == key.UP):
        if theApp.one_shot_mode:
            theApp.cur_canned_face = (theApp.cur_canned_face + 1) % len(canned_faces)
            theApp.one_shot_face = theApp.canned_aligned_faces[theApp.cur_canned_face]
        else:
            cur_vector = (cur_vector + 1) % num_vectors
    elif(symbol == key.LEFT or symbol == key.RIGHT):
        if theApp.one_shot_mode:
            theApp.one_shot_mode = False
            do_clear = True
        else:
            theApp.one_shot_mode = True
            theApp.one_shot_face = theApp.last_aligned_face
            theApp.one_shot_source = theApp.last_aligned_face
            theApp.one_shot_source_vector = theApp.last_encoded_vector
            theApp.cur_canned_face = -1
            do_clear = True
    elif(symbol == key.SPACE):
        print("SPACEBAR")
        theApp.write_last_aligned();
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
    parser.add_argument('--input-image', dest='input_image', default="images/startup_face.jpg",
                        help="use this input image instead of camera")
    parser.add_argument('--no-camera', dest='no_camera', default=False, action='store_true',
                        help="disable camera")
    parser.add_argument('--debug-outputs', dest='debug_outputs', default=False, action='store_true',
                        help="write diagnostic output files each frame")
    args = parser.parse_args()

    input_image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    theApp.input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    theApp.use_camera = not args.no_camera
    theApp.model_name = args.model

    if args.anchor_offset is not None:
        anchor_indexes = "0,1,2,3"
        offsets = get_json_vectors(args.anchor_offset)
        dim = len(offsets[0])
        offset_indexes = anchor_indexes.split(",")
        vector_offsets = [ -1 * offset_from_string(offset_indexes[0], offsets, dim) ]
        for i in range(len(offset_indexes) - 1):
            vector_offsets.append(offset_from_string(offset_indexes[i+1], offsets, dim))

    if args.debug_outputs:
        theApp.setDebugOutputs(args.debug_outputs)

    clock.schedule(draw)
    pyglet.app.run()
