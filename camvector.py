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
from ali.interface import AliModel
from plat.utils import offset_from_string, vectors_from_json_filelist, json_list_to_array
from plat.bin.atvec import do_roc
from plat.grid_layout import grid2img
from plat.sampling import real_glob
from plat.fuel_helper import get_dataset_iterator
from PIL import Image
from scipy.misc import imread, imsave, imresize
import subprocess

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

#number of predefined attribute vectors, initial one
num_vectors   = 3
cur_vector    = 0
do_clear = True

# global app of messy state
theApp = None

# the actual attribute vectors for VAE and GAN
vector_offsets = None
vector_offsets2 = None

# default window sizes
window_height = 800
window_width = 1280

# actual pyglet windows
window1 = None
window2 = None

# camera settings
# cam_width = 720
# cam_height = 512
cam_width = 400
cam_height = 300

# constants - arrow up/down mode
ARROW_MODE_VECTOR_SOURCE = 1
ARROW_MODE_VECTOR_DEST = 2
ARROW_MODE_IMAGE_SOURCE = 3

# images for pre-defined vectors
vector_files = [
    "images/OneShot.png",
    "images/Happy.png",
    "images/Angry.png",
    "images/Sunglasses.png",
]

# small images for pre-defined vectors
small_vector_files = [
    "images/OneShotSmall.png",
    "images/HappySmall.png",
    "images/AngrySmall.png",
    "images/SunglassesSmall.png",
]

# starter images
canned_faces = [
    "images/startup_face.jpg",
    "images/yann.png",
    "images/fei_fei.png",
    "images/bengio.jpg",
    "images/demis.jpg",
    "images/geoffrey.jpg",
]

# initialize and return camera handle
def setup_camera(device_number):
    cam = cv2.VideoCapture(device_number)
    result1 = cam.set(cv2.CAP_PROP_FRAME_WIDTH,cam_width)
    result2 = cam.set(cv2.CAP_PROP_FRAME_HEIGHT,cam_height)
    result3 = cam.set(cv2.CAP_PROP_FPS,1)
    return cam

# given a camera handle, return image in RGB format
def get_camera_image(camera):
    retval, img = camera.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# convert an RGB image to a pyglet texture for display
def image_to_texture(img):
    sy,sx,number_of_channels = img.shape
    number_of_bytes = sy*sx*number_of_channels
    img  = np.flipud(img)
    img = img.ravel()
    image_texture = (GLubyte * number_of_bytes)( *img.astype('uint8') )
    pImg = pyglet.image.ImageData(sx,sy,'RGB',
           image_texture,pitch=sx*number_of_channels)
    return pImg

# return aligned image
def get_aligned(img):
    success, im_resize, rect = doalign.align_face_buffer(img, 256, max_extension_amount=0)
    return im_resize

# encode image into latent space of model
def encode_from_image(rawim, dmodel, scale_factor=None):
    if scale_factor is not None:
        rawim = imresize(rawim, 1.0 / scale_factor)
    mixedim = np.asarray([[rawim[:,:,0], rawim[:,:,1], rawim[:,:,2]]])
    entry = (mixedim / 255.0).astype('float32')
    encoded = dmodel.encode_images(entry)[0]
    return encoded

# value mapping utility
def pr_map(value, istart, istop, ostart, ostop):
    return ostart + (ostop - ostart) * ((value - istart) / float(istop - istart));

# helper functions to handle arrow key presses in various modes
def canned_face_up(cur_index):
    return (cur_index + 1) % len(canned_faces)

def canned_face_down(cur_index):
    return (cur_index - 1 + len(canned_faces)) % len(canned_faces)

# big nasty switch statement of keypress
def do_key_press(symbol, modifiers):
    global cur_vector
    print("SO: {}".format(symbol))
    if(symbol == key.R):
        theApp.camera_recording = not theApp.camera_recording
        print("Camera recording is now {}".format(theApp.camera_recording))
    elif(symbol == key.A):
        theApp.arrow_mode = ARROW_MODE_IMAGE_SOURCE
    elif(symbol == key.S):
        theApp.arrow_mode = ARROW_MODE_VECTOR_SOURCE
    elif(symbol == key.D):
        theApp.arrow_mode = ARROW_MODE_VECTOR_DEST
    elif(symbol == key.G):
        theApp.gan_mode = not theApp.gan_mode
        theApp.last_recon_face = None
        theApp.reset_aligned_face()
        print("GAN mode is now {}".format(theApp.gan_mode))
    elif(symbol == key.DOWN):
        if theApp.arrow_mode == ARROW_MODE_IMAGE_SOURCE:
            theApp.cur_canned_face = canned_face_down(theApp.cur_canned_face)
        elif theApp.arrow_mode == ARROW_MODE_VECTOR_DEST:
            theApp.cur_vector_dest = canned_face_down(theApp.cur_vector_dest)
        elif theApp.arrow_mode == ARROW_MODE_VECTOR_SOURCE:
            theApp.cur_vector_source = canned_face_down(theApp.cur_vector_source)
    elif(symbol == key.UP):
        if theApp.arrow_mode == ARROW_MODE_IMAGE_SOURCE:
            theApp.cur_canned_face = canned_face_up(theApp.cur_canned_face)
        elif theApp.arrow_mode == ARROW_MODE_VECTOR_DEST:
            theApp.cur_vector_dest = canned_face_up(theApp.cur_vector_dest)
        elif theApp.arrow_mode == ARROW_MODE_VECTOR_SOURCE:
            theApp.cur_vector_source = canned_face_up(theApp.cur_vector_source)
    elif(symbol == key.LEFT):
        cur_vector = (cur_vector - 1 + num_vectors) % num_vectors
    elif(symbol == key.RIGHT):
        cur_vector = (cur_vector + 1) % num_vectors
    elif(symbol == key.Z):
        # three vectors mode
        theApp.one_shot_mode = False
        theApp.arrow_mode = ARROW_MODE_IMAGE_SOURCE
        do_clear = True
    elif(symbol == key.X):
        # one_shot mode
        theApp.one_shot_mode = True
        theApp.cur_vector_source = theApp.cur_canned_face
        theApp.cur_vector_dest = theApp.cur_canned_face
        snapshot(None)
        do_clear = True
    elif(symbol == key.SPACE):
        print("SPACEBAR")
        snapshot(None);
    elif(symbol == key.ESCAPE):
        print("ESCAPE")
        cv2.destroyAllWindows()
        if theApp.use_camera:
            cv2.VideoCapture(0).release()
        sys.exit(0)

def get_date_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

pipeline_dir = "pipeline/{}".format(get_date_str())
os.makedirs(pipeline_dir)
aligned_dir = "{}/aligned".format(pipeline_dir)
recon_dir = "{}/recon".format(pipeline_dir)
atstrip1_dir = "{}/atstrip1".format(pipeline_dir)
atstrip2_dir = "{}/atstrip2".format(pipeline_dir)
roc_dir = "{}/roc".format(pipeline_dir)
os.makedirs(aligned_dir)
os.makedirs(recon_dir)
os.makedirs(atstrip1_dir)
os.makedirs(atstrip2_dir)
os.makedirs(roc_dir)
command = "CUDA_VISIBLE_DEVICES=1 \
/usr/local/anaconda2/envs/enhance/bin/python \
    ../neural-enhance3/enhance.py --model dlib_256_neupup1 --zoom 1 \
    --device gpu0 \
    --rendering-tile 256 \
    --rendering-overlap 1 \
    --input-directory {} \
    --output-directory {} \
    --watch".format(atstrip1_dir, atstrip2_dir)
with open("enhance.sh", "w+") as text_file:
    text_file.write(command)
# p=subprocess.Popen(command, shell=True)

# global images that get displayed on win2
win2_aligned_im = None
win2_smile_im = None
win2_surprised_im = None
win2_angry_im = None
win2_oneshot_a1 = None
win2_oneshot_a2 = None
win2_oneshot_b1 = None
win2_oneshot_b2 = None
win2_oneshot_c1 = None
win2_oneshot_c2 = None

# watcher to load win2 images on filesystem change
class InputFileHandler(FileSystemEventHandler):
    def process(self, infile):
        global win2_aligned_im, win2_smile_im, win2_surprised_im, win2_angry_im
        # basename = os.path.basename(infile)
        # if basename[0] == '.' or basename[0] == '_':
        #     print("Skipping infile: {}".format(infile))
        #     return;
        print("Processing infile: {}".format(infile))
        # aligned_file = "{}/{}".format(aligned_dir, basename)
        # win2_aligned_im = imread(aligned_file, mode='RGB')
        img = imread(infile, mode='RGB')
        win2_smile_im = img[0:256,0:256,0:3]
        win2_surprised_im = img[0:256,256:512,0:3]
        win2_angry_im = img[0:256,512:768,0:3]
        win2_aligned_im = img[0:256,768:1024,0:3]
        # if theApp is not None:
        #     draw2(None)

    def on_modified(self, event):
        if not event.is_directory:
            self.process(event.src_path)

event_handler = InputFileHandler()
observer = Observer()
observer.schedule(event_handler, path=atstrip2_dir, recursive=False)
observer.start()

class MainApp():
    last_recon_face = None
    one_shot_source_vector = None
    debug_outputs = False
    framecount = 0
    timecount  = 0
    use_camera = True
    camera = None
    model_name = None
    model_name2 = None
    dmodel = None
    dmodel2 = None
    one_shot_mode = False
    gan_mode = False
    cur_canned_face = 0
    cur_vector_source = 0
    cur_vector_dest  = 0
    cur_aligned_face_number = 0
    last_saved_aligned_face_number = 0
    last_draw1_aligned_face_number = 0
    scale_factor = None
    arrow_mode = ARROW_MODE_IMAGE_SOURCE
    camera_recording = False

    """Just a container for unfortunate global state"""
    def __init__(self):
        self.cur_frame = 0
        self.vector_textures = []
        self.small_vector_textures = []
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
                self.vector_y1 = int(0)
                self.vector_y3 = int(window_height - h)
            self.vector_textures.append(image_to_texture(vector_im))
        for i in range(len(small_vector_files)):
            png = Image.open(small_vector_files[i])
            if png.mode == "RGBA":
                png.load()
                vector_im = Image.new("RGB", png.size, (0, 0, 0))
                vector_im.paste(png, mask=png.split()[3]) # 3 is the alpha channel
            else:
                vector_im = png
            vector_im = np.asarray(vector_im)
            # vector_im = imread(vector_files[i], mode='RGB')
            if i == 1:
                h, w, c = vector_im.shape
                self.small_vector_x = int((window_width - w) / 2)
                self.small_vector_y = int((window_height - h) / 2)
                self.small_vector_y1 = int((256 - h) / 2)
                self.small_vector_y3 = int(window_height-(256/2) - h/2)
            self.small_vector_textures.append(image_to_texture(vector_im))
        self.cur_canned_face = 0
        self.canned_aligned = []
        self.canned_encoded = []
        self.canned_textures = []
        self.canned_small_textures = []
        self.canned_smaller_textures = []
        self.num_canned = len(canned_faces)
        for i in range(self.num_canned):
            canned_face = imread(canned_faces[i])
            self.canned_aligned.append(get_aligned(canned_face))
            self.canned_encoded.append(None)
            self.canned_textures.append(None)
            self.canned_small_textures.append(None)
            self.canned_smaller_textures.append(None)

    def setDebugOutputs(self, mode):
        self.debug_outputs = mode

    def reset_aligned_face(self):
        self.last_saved_aligned_face_number = 0
        self.last_draw1_aligned_face_number = 0

    def write_cur_aligned(self, debugfile=False, datestr=None):
        if debugfile:
            datestr = "debug"
        elif datestr is None:
            datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "{}/{}.png".format(aligned_dir,datestr)
        if not debugfile and os.path.exists(filename):
            return
        imsave(filename, self.canned_aligned[theApp.cur_canned_face])

        filename = "{}/{}.png".format(recon_dir, datestr)    
        if not debugfile and os.path.exists(filename):
            return
        imsave(filename, self.canned_aligned[self.cur_canned_face])

    def get_encoded(self, dmodel_cur, image_index, scale_factor):
        if self.canned_encoded[image_index] is None:
            self.canned_encoded[image_index] = encode_from_image(self.canned_aligned[image_index], dmodel_cur, scale_factor)
            print("Initialized {} im {} to {}".format(image_index, self.canned_aligned[image_index][0:0,0:3,0:3], self.canned_encoded[image_index][:5]))
        return self.canned_encoded[image_index]

    def get_recon_strip(self, dmodel_cur, scale_factor):
        global vector_offsets, vector_offsets2, cur_vector

        if dmodel_cur is None or (not self.gan_mode and vector_offsets is None) or (self.gan_mode and vector_offsets2 is None):
            decode_list = []
            for i in range(5):
                decode_list.append(self.canned_aligned[self.cur_canned_face])
            decoded = np.array(decode_list)
            decoded_array = np.concatenate(decoded, axis=1)
            return decoded_array

        encoded_source_image = self.get_encoded(dmodel_cur, self.cur_canned_face, scale_factor)

        decode_list = []
        if self.gan_mode:
            vector_index_start = 0
            cur_vector_offsets = vector_offsets2
            deblur_vector = None
        else:
            vector_index_start = 1
            cur_vector_offsets = vector_offsets
            deblur_vector = vector_offsets[0]
        if self.one_shot_mode:
            encoded_vector_source = self.get_encoded(dmodel_cur, self.cur_vector_source, scale_factor)
            encoded_vector_dest = self.get_encoded(dmodel_cur, self.cur_vector_dest, scale_factor)
            attribute_vector = encoded_vector_dest - encoded_vector_source
        else:
            attribute_vector = cur_vector_offsets[vector_index_start+cur_vector]
        for i in range(5):
            vector_scalar = pr_map(i, 0, 5, -1.5, 1.5)
            cur_anchor = encoded_source_image + vector_scalar * attribute_vector
            if not self.gan_mode:
                cur_anchor += deblur_vector
            decode_list.append(cur_anchor)
        decoded = dmodel_cur.sample_at(np.array(decode_list))
        n, c, y, x = decoded.shape
        decoded_strip = np.concatenate(decoded, axis=2)
        decoded_array = (255 * np.dstack(decoded_strip)).astype(np.uint8)
        if self.scale_factor is not None:
            decoded_array = imresize(decoded_array, self.scale_factor)
        elif self.gan_mode:
            decoded_array = imresize(decoded_array, 2.0)
        return decoded_array

    # get a triple of effects image. not for one-shot
    def write_recon_triple(self, dmodel_cur, datestr, scale_factor):
        global vector_offsets, vector_offsets2
        global win2_aligned_im, win2_smile_im, win2_surprised_im, win2_angry_im

        if dmodel_cur is None or (not self.gan_mode and vector_offsets is None) or (self.gan_mode and vector_offsets2 is None):
            win2_smile_im = self.canned_aligned[self.cur_canned_face]
            win2_surprised_im = self.canned_aligned[self.cur_canned_face]
            win2_angry_im = self.canned_aligned[self.cur_canned_face]
            win2_aligned_im = self.canned_aligned[self.cur_canned_face]
            return None

        encoded_source_image = self.get_encoded(dmodel_cur, self.cur_canned_face, scale_factor)
        decode_list = []
        if self.gan_mode:
            vector_index_start = 0
            cur_vector_offsets = vector_offsets2
            deblur_vector = None
        else:
            vector_index_start = 1
            cur_vector_offsets = vector_offsets
            deblur_vector = vector_offsets[0]
        for i in range(4):
            if i == 3:
                cur_anchor = encoded_source_image
            else:
                vector_scalar = 1.5
                attribute_vector = cur_vector_offsets[vector_index_start+i]
                cur_anchor = encoded_source_image + vector_scalar * attribute_vector
                if not self.gan_mode:
                    cur_anchor += deblur_vector
            decode_list.append(cur_anchor)
        decoded = dmodel_cur.sample_at(np.array(decode_list))
        n, c, y, x = decoded.shape
        decoded_strip = np.concatenate(decoded, axis=2)
        decoded_array = (255 * np.dstack(decoded_strip)).astype(np.uint8)

        if scale_factor is not None:
            decoded_array = imresize(decoded_array, scale_factor)
        elif self.gan_mode:
            decoded_array = imresize(decoded_array, 2.0)

        win2_smile_im = decoded_array[0:256,0:256,0:3]
        win2_surprised_im = decoded_array[0:256,256:512,0:3]
        win2_angry_im = decoded_array[0:256,512:768,0:3]
        win2_aligned_im = decoded_array[0:256,768:1024,0:3]

        if not self.gan_mode:
            filename = "{}/{}_attrib.png".format(atstrip1_dir, datestr)
            if os.path.exists(filename):
                return
            imsave(filename, decoded_array)

    # get a triple of effects image. not for one-shot
    def write_oneshot_sixpack(self, dmodel_cur, datestr, scale_factor):
        global win2_oneshot_a1, win2_oneshot_a2, win2_oneshot_b1, win2_oneshot_b2, win2_oneshot_c1, win2_oneshot_c2
        global vector_offsets

        index_a = self.cur_canned_face
        index_b = canned_face_up(index_a)
        index_c = canned_face_up(index_b)
        canned_indexes = [index_a, index_b, index_c]
        print("Canned indexes {}".format(canned_indexes))
        if dmodel_cur is None or (not self.gan_mode and vector_offsets is None) or (self.gan_mode and vector_offsets2 is None):
            win2_oneshot_a1 = self.canned_aligned[index_a]
            win2_oneshot_a2 = self.canned_aligned[index_a]
            win2_oneshot_b1 = self.canned_aligned[index_b]
            win2_oneshot_b2 = self.canned_aligned[index_b]
            win2_oneshot_c1 = self.canned_aligned[index_c]
            win2_oneshot_c2 = self.canned_aligned[index_c]
            return None

        encoded_vector_source = self.get_encoded(dmodel_cur, self.cur_vector_source, scale_factor)
        encoded_vector_dest = self.get_encoded(dmodel_cur, self.cur_vector_dest, scale_factor)
        attribute_vector = encoded_vector_dest - encoded_vector_source

        deblur_vector = vector_offsets[0]

        decode_list = []
        for i in range(0,3):
            encoded_source_image = self.get_encoded(dmodel_cur, canned_indexes[i], scale_factor)
            cur_anchor = encoded_source_image
            # if not self.gan_mode:
            #     cur_anchor += deblur_vector
            decode_list.append(cur_anchor)
            vector_scalar = 1.5
            cur_anchor = encoded_source_image + vector_scalar * attribute_vector
            # if not self.gan_mode:
            #     cur_anchor += deblur_vector
            decode_list.append(cur_anchor)

        decoded = dmodel_cur.sample_at(np.array(decode_list))
        n, c, y, x = decoded.shape
        decoded_strip = np.concatenate(decoded, axis=2)
        decoded_array = (255 * np.dstack(decoded_strip)).astype(np.uint8)

        if scale_factor is not None:
            decoded_array = imresize(decoded_array, scale_factor)
        elif self.gan_mode:
            decoded_array = imresize(decoded_array, 2.0)

        win2_oneshot_a1 = decoded_array[0:256,0:256,0:3]
        win2_oneshot_a2 = decoded_array[0:256,256:512,0:3]
        win2_oneshot_b1 = decoded_array[0:256,512:768,0:3]
        win2_oneshot_b2 = decoded_array[0:256,768:1024,0:3]
        win2_oneshot_c1 = decoded_array[0:256,1024:1280,0:3]
        win2_oneshot_c2 = decoded_array[0:256,1280:1536,0:3]

        if not self.gan_mode:
            filename = "{}/{}_oneshot.png".format(atstrip1_dir, datestr)
            if os.path.exists(filename):
                return
            imsave(filename, decoded_array)

    def get_small_texture(self, image_index, super_small=False):
        if super_small:
            if self.canned_smaller_textures[image_index] is None:
                small_source = imresize(self.canned_aligned[image_index], (138, 138))
                self.canned_smaller_textures[image_index] = image_to_texture(small_source)
            return self.canned_smaller_textures[image_index]
        else:
            if self.canned_small_textures[image_index] is None:
                small_source = imresize(self.canned_aligned[image_index], (164, 164))
                self.canned_small_textures[image_index] = image_to_texture(small_source)
            return self.canned_small_textures[image_index]

    def draw1(self, dt):
        global window1, cur_vector, do_clear

        # clear window only sometimes
        if do_clear or True:
            window1.clear()
            do_clear = False

        if self.cur_frame == 5:
            print("Fake key presses")
            # do_key_press(key.LEFT, None)

        # initialize camera and dmodel after warming up
        if self.camera is None and self.use_camera and self.cur_frame > 10:
            print("Initializing camera")
            self.camera = setup_camera(self.camera_device)

        if self.dmodel is None and self.model_name and self.cur_frame > 20:
            print("Initializing model {}".format(self.model_name))
            self.dmodel = DiscGenModel(filename=self.model_name)
            theApp.reset_aligned_face()

        if self.dmodel2 is None and self.model_name2 and self.cur_frame > 30:
            print("Initializing model {}".format(self.model_name2))
            self.dmodel2 = AliModel(filename=self.model_name2)
            print("Dmodel2 is {}".format(self.dmodel2))
            theApp.reset_aligned_face()

        if self.cur_frame == 35:
            print("Fake key presses")
            # do_key_press(key.G, None)
            # do_key_press(key.LEFT, None)

        # get source image
        if theApp.arrow_mode == ARROW_MODE_IMAGE_SOURCE:
            face_index = theApp.cur_canned_face
        elif theApp.arrow_mode == ARROW_MODE_VECTOR_SOURCE:
            face_index = theApp.cur_vector_source
        else:
            face_index = theApp.cur_vector_dest
        if self.camera is not None and face_index == 0 and theApp.camera_recording:
            candidate = get_aligned(get_camera_image(self.camera))
            if candidate is not None:
                theApp.canned_aligned[face_index] = candidate
                theApp.canned_encoded[face_index] = None
                theApp.canned_textures[face_index] = None
                theApp.canned_small_textures[face_index] = None
                theApp.canned_smaller_textures[face_index] = None

        align_im = theApp.canned_aligned[theApp.cur_canned_face]

        if self.one_shot_mode:
            vector_index = 0
        else:
            vector_index = cur_vector + 1
        self.vector_textures[vector_index].blit(self.vector_x, self.vector_y)

        if True:
            theApp.last_draw1_aligned_face_number = theApp.cur_aligned_face_number
            if self.one_shot_mode:
                source_tex = self.get_small_texture(self.cur_vector_source)
                source_x, source_y = self.vector_x + 176, self.vector_y + 11
                source_tex.blit(source_x, source_y)

                dest_tex = self.get_small_texture(self.cur_vector_dest)
                source_x, source_y = self.vector_x + 688, self.vector_y + 11
                dest_tex.blit(source_x, source_y)

            if self.canned_textures[self.cur_canned_face] is None:
                self.canned_textures[self.cur_canned_face] = image_to_texture(self.canned_aligned[self.cur_canned_face])

            self.canned_textures[self.cur_canned_face].blit(window_width / 2 - 128, window_height - 256)

            if self.gan_mode and self.dmodel2 is not None:
                recon = self.get_recon_strip(self.dmodel2, 2)
            else:
                recon = self.get_recon_strip(self.dmodel, self.scale_factor)
            if recon is not None:
                self.last_recon_face = recon
                recon_tex = image_to_texture(recon)
                recon_tex.blit(0, 0)

        # if self.debug_outputs:
        #     self.write_last_aligned(debugfile=True)

        self.framecount += pyglet.clock.get_fps()
        self.timecount  += dt
        self.cur_frame += 1
        return

    def draw_oneshot_small(self, vector_y, source_tex, dest_tex):
        self.small_vector_textures[0].blit(self.small_vector_x, vector_y)
        source_x, source_y = self.small_vector_x + 90, vector_y + 24
        source_tex.blit(source_x, source_y)
        dest_x, dest_y = self.small_vector_x + 540, vector_y + 24
        dest_tex.blit(dest_x, dest_y)

    def draw2(self, dt):
        window2.clear()
        global win2_aligned_im, win2_smile_im, win2_surprised_im, win2_angry_im
        global win2_oneshot_a1, win2_oneshot_a2, win2_oneshot_b1, win2_oneshot_b2, win2_oneshot_c1, win2_oneshot_c2
        if self.one_shot_mode:
            if win2_oneshot_a1 is not None:
                oneshot_tex = image_to_texture(win2_oneshot_a1)
                oneshot_tex.blit(0, window_height-256)
            if win2_oneshot_a2 is not None:
                oneshot_tex = image_to_texture(win2_oneshot_a2)
                oneshot_tex.blit(window_width-256, window_height-256)
            if win2_oneshot_b1 is not None:
                oneshot_tex = image_to_texture(win2_oneshot_b1)
                oneshot_tex.blit(0, int((window_height-256)/2))
            if win2_oneshot_b2 is not None:
                oneshot_tex = image_to_texture(win2_oneshot_b2)
                oneshot_tex.blit(window_width-256, int((window_height-256)/2))
            if win2_oneshot_c1 is not None:
                oneshot_tex = image_to_texture(win2_oneshot_c1)
                oneshot_tex.blit(0, 0)
            if win2_oneshot_c2 is not None:
                oneshot_tex = image_to_texture(win2_oneshot_c2)
                oneshot_tex.blit(window_width-256, 0)

            source_tex = self.get_small_texture(self.cur_vector_source, True)
            dest_tex = self.get_small_texture(self.cur_vector_dest, True)
            # top
            self.draw_oneshot_small(self.small_vector_y3, source_tex, dest_tex)
            # middle
            self.draw_oneshot_small(self.small_vector_y, source_tex, dest_tex)
            # bottom
            self.draw_oneshot_small(self.small_vector_y1, source_tex, dest_tex)
        else:
            self.small_vector_textures[1].blit(self.small_vector_x, self.small_vector_y3)
            self.small_vector_textures[2].blit(self.small_vector_x, self.small_vector_y)
            self.small_vector_textures[3].blit(self.small_vector_x, self.small_vector_y1)
            if win2_aligned_im is not None:
                win2_aligned_tex = image_to_texture(win2_aligned_im)
                win2_aligned_tex.blit(0, 0)
                win2_aligned_tex.blit(0, int((window_height-256)/2))
                win2_aligned_tex.blit(0, window_height-256)
            if win2_smile_im is not None:
                win2_smile_tex = image_to_texture(win2_smile_im)
                win2_smile_tex.blit(window_width-256, window_height-256)
            if win2_surprised_im is not None:
                win2_surprised_tex = image_to_texture(win2_surprised_im)
                win2_surprised_tex.blit(window_width-256, int((window_height-256)/2))
            if win2_angry_im is not None:
                win2_angry_tex = image_to_texture(win2_angry_im)
                win2_angry_tex.blit(window_width-256, 0)

# Draw Loop
def draw1(dt):
    global window1
    if window1 == None:
        return
    window1.switch_to()
    theApp.draw1(dt)

def draw2(dt):
    global window2
    if window2 == None:
        return
    window2.switch_to()
    theApp.draw2(dt)

# snapshot the current state and write a file to the processing queue
def snapshot(dt):
    print("SNAPSHOT: saving")

    datestr = get_date_str()
    theApp.write_cur_aligned(datestr=datestr)
    if not theApp.one_shot_mode:
        if theApp.gan_mode and theApp.dmodel2 is not None:
            theApp.write_recon_triple(theApp.dmodel2, datestr, 2)
        else:
            theApp.write_recon_triple(theApp.dmodel, datestr, theApp.scale_factor)
    else:
        if theApp.gan_mode and theApp.dmodel2 is not None:
            theApp.write_oneshot_sixpack(theApp.dmodel2, datestr, 2)
        else:
            theApp.write_oneshot_sixpack(theApp.dmodel, datestr, theApp.scale_factor)


theApp = MainApp()

if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser(description='Let get NIPSy')
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--model2", dest='model2', type=str, default=None,
                        help="path to the saved ali model")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-offset2', dest='anchor_offset2', default=None,
                        help="use json file as source of each ali anchors offsets")
    parser.add_argument('--anchor-index', dest='anchor_index', default="0,1,2,3",
                        help="indexes to offsets in anchor-offset")    
    parser.add_argument('--anchor-index2', dest='anchor_index2', default="0,1,2,3",
                        help="indexes to offsets in anchor-offset2")    
    parser.add_argument('--no-camera', dest='no_camera', default=False, action='store_true',
                        help="disable camera")
    parser.add_argument('--skip1', dest='skip1', default=False, action='store_true',
                        help="no window 1")
    parser.add_argument('--skip2', dest='skip2', default=False, action='store_true',
                        help="no window 2")
    parser.add_argument('--debug-outputs', dest='debug_outputs', default=False, action='store_true',
                        help="write diagnostic output files each frame")
    parser.add_argument('--full1', dest='full1', default=None, type=int,
                        help="Index to screen to use for window one in fullscreen mode")
    parser.add_argument('--full2', dest='full2', default=None, type=int,
                        help="Index to screen to use for window two in fullscreen mode")
    parser.add_argument('--camera', dest='camera', default=1, type=int,
                        help="Camera device number")
    parser.add_argument("--encoded-vectors", type=str, default=None,
                        help="Comma separated list of json arrays")
    parser.add_argument('--dataset', dest='dataset', default=None,
                        help="Source dataset (for labels).")
    parser.add_argument('--labels', dest='labels', default=None,
                        help="Text file with 0/1 labels.")
    parser.add_argument('--split', dest='split', default="valid",
                        help="Which split to use from the dataset (train/nontrain/valid/test/any).")
    parser.add_argument('--roc-limit', dest='roc_limit', default=10000, type=int,
                        help="Limit roc operation to this many vectors")
    parser.add_argument('--scale-factor', dest='scale_factor', default=None, type=float,
                        help="Scale up outputs of model")

    args = parser.parse_args()

    theApp.camera_device = args.camera
    theApp.scale_factor = args.scale_factor

    display = pyglet.window.get_platform().get_default_display()
    screens = display.get_screens()
    if args.skip1:
        window1 = None
    elif args.full1 is not None:
        window1 = pyglet.window.Window(fullscreen=True, screen=screens[args.full1])
    else:
        window1 = pyglet.window.Window(window_width, window_height, resizable=False)
        window1.set_location(0, 0)

    if args.skip2:
        window2 = None
    elif args.full2 is not None:
        window2 = pyglet.window.Window(fullscreen=True, screen=screens[args.full2])
    else:
        window2 = pyglet.window.Window(window_width, window_height, resizable=False)
        window2.set_location(100, 100)

    if window1 is not None:
        @window1.event
        def on_key_press(symbol, modifiers):
            do_key_press(symbol, modifiers)
    if window2 is not None:
        @window2.event
        def on_key_press(symbol, modifiers):
            do_key_press(symbol, modifiers)

    theApp.use_camera = not args.no_camera
    theApp.model_name = args.model
    theApp.model_name2 = args.model2

    if args.anchor_offset is not None:
        anchor_index = args.anchor_index
        offsets = vectors_from_json_filelist(real_glob(args.anchor_offset))
        dim = len(offsets[0])
        offset_indexes = anchor_index.split(",")
        vector_offsets = [ -1 * offset_from_string(offset_indexes[0], offsets, dim) ]
        for i in range(len(offset_indexes) - 1):
            vector_offsets.append(offset_from_string(offset_indexes[i+1], offsets, dim))

    if args.anchor_offset2 is not None:
        anchor_index = args.anchor_index2
        offsets = vectors_from_json_filelist(real_glob(args.anchor_offset2))
        dim = len(offsets[0])
        offset_indexes = anchor_index.split(",")
        vector_offsets2 = []
        for i in range(len(offset_indexes)):
            vector_offsets2.append(offset_from_string(offset_indexes[i], offsets, dim))

    if args.encoded_vectors is not None and (args.dataset is not None or args.labels is not None):
        encoded = json_list_to_array(args.encoded_vectors)
        num_rows, z_dim = encoded.shape
        if args.dataset:
            attribs = np.array(list(get_dataset_iterator(args.dataset, args.split, include_features=False, include_targets=True)))
        else:
            attribs = get_attribs_from_file(args.labels)
        encoded = encoded[:args.roc_limit]
        attribs = attribs[:args.roc_limit]
        print("encoded vectors: {}, attributes: {} ".format(encoded.shape, attribs.shape))

        attribute_index = 31
        chosen_vector = vector_offsets[1]
        dim = len(chosen_vector)
        threshold = None
        outfile = "{}/{}.png".format(roc_dir, get_date_str())
        do_roc(chosen_vector, encoded, attribs, attribute_index, threshold, outfile)

    if args.debug_outputs:
        theApp.setDebugOutputs(args.debug_outputs)

    snapshot(None)
    pyglet.clock.schedule_interval(draw1, 1)
    pyglet.clock.schedule_interval(snapshot, 15)
    pyglet.clock.schedule_interval(draw2, 1)
    pyglet.app.run()
