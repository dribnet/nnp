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
from plat.utils import get_json_vectors, offset_from_string, vectors_from_json_filelist
from plat.grid_layout import grid2img
from plat.sampling import real_glob
from PIL import Image
from scipy.misc import imread, imsave, imresize
import subprocess

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

num_vectors   = 3
cur_vector    = 0
do_clear = True

theApp = None
vector_offsets = None
vector_offsets2 = None
window_height = 800
window_width = 1280
window1 = None
window2 = None
cam_width = 720
cam_height = 512

vector_files = [
    "images/OneShot.png",
    "images/Happy.png",
    "images/Surprised.png",
    "images/Angry.png",
]

small_vector_files = [
    "images/OneShotSmall.png",
    "images/HappySmall.png",
    "images/SurprisedSmall.png",
    "images/AngrySmall.png",
]

canned_faces = [
    "images/bengio.jpg",
    "images/demis.jpg",
    "images/fei_fei.jpg",
    "images/geoffrey.jpg",
    "images/yann.jpg",
]

def setup_camera(device_number):
    cam = cv2.VideoCapture(device_number)
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

def do_key_press(symbol, modifiers):
    global cur_vector
    print("SO: {}".format(symbol))
    if(symbol == key.S):
        if theApp.one_shot_mode:
            theApp.one_shot_source = theApp.last_aligned_face
            theApp.one_shot_source_vector = theApp.last_encoded_vector
    if(symbol == key.G):
        theApp.gan_mode = not theApp.gan_mode
        theApp.last_encoded_vector = None
        theApp.last_recon_face = None
        theApp.reset_aligned_face()
        print("GAN mode is now {}".format(theApp.gan_mode))
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

def get_date_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

pipeline_dir = "pipeline/{}".format(get_date_str())
os.makedirs(pipeline_dir)
aligned_dir = "{}/aligned".format(pipeline_dir)
recon_dir = "{}/recon".format(pipeline_dir)
atstrip1_dir = "{}/atstrip1".format(pipeline_dir)
atstrip2_dir = "{}/atstrip2".format(pipeline_dir)
os.makedirs(aligned_dir)
os.makedirs(recon_dir)
os.makedirs(atstrip1_dir)
os.makedirs(atstrip2_dir)
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

win2_aligned_im = None
win2_smile_im = None
win2_surprised_im = None
win2_angry_im = None

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
    model_name2 = None
    dmodel = None
    dmodel2 = None
    one_shot_mode = False
    gan_mode = False
    cur_canned_face = -1;
    cur_aligned_face_number = 0
    last_saved_aligned_face_number = 0
    last_draw1_aligned_face_number = 0

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
        self.cur_canned_face = -1
        self.canned_aligned_faces = []
        for i in range(len(canned_faces)):
            canned_face = imread(canned_faces[i])
            self.canned_aligned_faces.append(get_aligned(canned_face))

    def setDebugOutputs(self, mode):
        self.debug_outputs = mode

    def reset_aligned_face(self):
        self.last_saved_aligned_face_number = 0
        self.last_draw1_aligned_face_number = 0

    def write_last_aligned(self, debugfile=False, datestr=None):
        if self.last_aligned_face is None:
            return
        if debugfile:
            datestr = "debug"
        elif datestr is None:
            datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "{}/{}.png".format(aligned_dir,datestr)
        if not debugfile and os.path.exists(filename):
            return
        imsave(filename, self.last_aligned_face)

        if self.last_recon_face is None:
            return
        filename = "{}/{}.png".format(recon_dir, datestr)    
        if not debugfile and os.path.exists(filename):
            return
        imsave(filename, self.last_recon_face)

    def get_recon_strip(self, rawim, dmodel_cur):
        global vector_offsets, vector_offsets2, cur_vector

        if dmodel_cur is None or (not self.gan_mode and vector_offsets is None) or (self.gan_mode and vector_offsets2 is None):
            decode_list = []
            for i in range(5):
                decode_list.append(rawim)
            decoded = np.array(decode_list)
            decoded_array = np.concatenate(decoded, axis=1)
            return decoded_array

        encoded = encode_from_image(rawim, dmodel_cur)
        self.last_encoded_vector = encoded
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
            if self.one_shot_source_vector is not None:
                # compute attribute vector
                attribute_vector = encoded - self.one_shot_source_vector
            else:
                # smile is debug ?
                attribute_vector = cur_vector_offsets[vector_index_start]
            # override encoded to be one_shot_face
            encoded = encode_from_image(self.one_shot_face, dmodel_cur)
        else:
            attribute_vector = cur_vector_offsets[vector_index_start+cur_vector]
        for i in range(5):
            scale_factor = pr_map(i, 0, 5, -1.5, 1.5)
            cur_anchor = encoded + scale_factor * attribute_vector
            if not self.gan_mode:
                cur_anchor += deblur_vector
            decode_list.append(cur_anchor)
        decoded = dmodel_cur.sample_at(np.array(decode_list))
        n, c, y, x = decoded.shape
        decoded_strip = np.concatenate(decoded, axis=2)
        decoded_array = (255 * np.dstack(decoded_strip)).astype(np.uint8)
        if self.gan_mode:
            decoded_array = imresize(decoded_array, 2.0)
        return decoded_array

    # get a triple of effects image. not for one-shot
    def write_recon_triple(self, rawim, dmodel_cur, datestr):
        global vector_offsets, vector_offsets2
        global win2_aligned_im, win2_smile_im, win2_surprised_im, win2_angry_im

        if dmodel_cur is None or (not self.gan_mode and vector_offsets is None) or (self.gan_mode and vector_offsets2 is None):
            return None

        encoded = encode_from_image(rawim, dmodel_cur)
        self.last_encoded_vector = encoded
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
                cur_anchor = encoded
            else:
                scale_factor = 1.5
                attribute_vector = cur_vector_offsets[vector_index_start+i]
                cur_anchor = encoded + scale_factor * attribute_vector
                if not self.gan_mode:
                    cur_anchor += deblur_vector
            decode_list.append(cur_anchor)
        decoded = dmodel_cur.sample_at(np.array(decode_list))
        n, c, y, x = decoded.shape
        decoded_strip = np.concatenate(decoded, axis=2)
        decoded_array = (255 * np.dstack(decoded_strip)).astype(np.uint8)

        if self.gan_mode:
            decoded_array = imresize(decoded_array, 2.0)

        win2_smile_im = decoded_array[0:256,0:256,0:3]
        win2_surprised_im = decoded_array[0:256,256:512,0:3]
        win2_angry_im = decoded_array[0:256,512:768,0:3]
        win2_aligned_im = decoded_array[0:256,768:1024,0:3]

        if not self.gan_mode:
            filename = "{}/{}.png".format(atstrip1_dir, datestr)
            if os.path.exists(filename):
                return
            imsave(filename, decoded_array)


    def draw1(self, dt):
        global window1, cur_vector, do_clear

        # clear window only sometimes
        if do_clear:
            window1.clear()
            do_clear = False

        # initialize camera and dmodel after warming up
        if self.camera is None and self.use_camera and self.cur_frame > 10:
            print("Initializing camera")
            self.camera = setup_camera(self.camera_device)

        if self.cur_frame == 5:
            print("Fake key presses")
            # do_key_press(key.LEFT, None)

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
        if self.camera:
            img = get_camera_image(self.camera)
        else:
            img = self.input_image

        align_im = get_aligned(img)
        if align_im is not None:
            self.last_aligned_face = align_im
            theApp.cur_aligned_face_number = theApp.cur_aligned_face_number + 1

        if self.one_shot_mode:
            vector_index = 0
        else:
            vector_index = cur_vector + 1
        self.vector_textures[vector_index].blit(self.vector_x, self.vector_y)

        if self.last_aligned_face is not None and theApp.cur_aligned_face_number != theApp.last_draw1_aligned_face_number:
            theApp.last_draw1_aligned_face_number = theApp.cur_aligned_face_number
            if self.one_shot_mode:
                aligned_small = imresize(self.last_aligned_face, (180, 180))
                align_tex = image_to_texture(aligned_small)
                # align_tex.blit(3 * window_width / 4 - 128, int((window_height - 256) / 2))
                align_tex.blit(3 * window_width / 4 - 128 - 32, int((window_height - 180) / 2))
                small_source = imresize(self.one_shot_source, (180, 180))
                one_shot_source_tex = image_to_texture(small_source)
                one_shot_source_tex.blit(window_width / 4 - 128 + 96, int((window_height - 180) / 2))
                one_shot_face_tex = image_to_texture(self.one_shot_face)
                one_shot_face_tex.blit(window_width / 2 - 128, window_height - 256)
            else:
                align_tex = image_to_texture(self.last_aligned_face)
                align_tex.blit(window_width / 2 - 128, window_height - self.last_aligned_face.shape[0])

            if self.gan_mode and self.dmodel2 is not None:
                recon = self.get_recon_strip(self.last_aligned_face, self.dmodel2)
            else:
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

    def draw2(self, dt):
        global win2_aligned_im, win2_smile_im, win2_surprised_im, win2_angry_im
        if self.one_shot_mode:
            self.small_vector_textures[0].blit(self.small_vector_x, self.small_vector_y3)
            self.small_vector_textures[0].blit(self.small_vector_x, self.small_vector_y)
            self.small_vector_textures[0].blit(self.small_vector_x, self.small_vector_y1)
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
    window1.switch_to()
    theApp.draw1(dt)

def draw2(dt):
    global window2
    window2.switch_to()
    theApp.draw2(dt)

# snapshot the current state and write a file to the processing queue
def snapshot(dt):
    if theApp.last_aligned_face is None:
        print("skipping snapshot - no aligned face")
        return

    if theApp.last_saved_aligned_face_number == theApp.cur_aligned_face_number:
        print("skipping snapshot - no new face")
        return

    theApp.last_saved_aligned_face_number = theApp.cur_aligned_face_number

    print("SNAPSHOT: saving")

    datestr = get_date_str()
    theApp.write_last_aligned(datestr=datestr)
    if not theApp.one_shot_mode:
        if theApp.gan_mode and theApp.dmodel2 is not None:
            theApp.write_recon_triple(theApp.last_aligned_face, theApp.dmodel2, datestr)
        else:
            theApp.write_recon_triple(theApp.last_aligned_face, theApp.dmodel, datestr)


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
    parser.add_argument('--input-image', dest='input_image', default="images/startup_face.jpg",
                        help="use this input image instead of camera")
    parser.add_argument('--no-camera', dest='no_camera', default=False, action='store_true',
                        help="disable camera")
    parser.add_argument('--debug-outputs', dest='debug_outputs', default=False, action='store_true',
                        help="write diagnostic output files each frame")
    parser.add_argument('--full1', dest='full1', default=None, type=int,
                        help="Index to screen to use for window one in fullscreen mode")
    parser.add_argument('--full2', dest='full2', default=None, type=int,
                        help="Index to screen to use for window two in fullscreen mode")
    parser.add_argument('--camera', dest='camera', default=1, type=int,
                        help="Camera device number")
    args = parser.parse_args()

    theApp.camera_device = args.camera

    display = pyglet.window.get_platform().get_default_display()
    screens = display.get_screens()
    if args.full1 is not None:
        window1 = pyglet.window.Window(fullscreen=True, screen=screens[args.full1])
    else:
        window1 = pyglet.window.Window(window_width, window_height, resizable=False)
        window1.set_location(0, 0)

    @window1.event
    def on_key_press(symbol, modifiers):
        do_key_press(symbol, modifiers)

    if args.full2 is not None:
        window2 = pyglet.window.Window(fullscreen=True, screen=screens[args.full2])
    else:
        window2 = pyglet.window.Window(window_width, window_height, resizable=False)
        window1.set_location(100, 100)

    input_image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    theApp.input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

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

    if args.debug_outputs:
        theApp.setDebugOutputs(args.debug_outputs)

    pyglet.clock.schedule_interval(draw1, 1/60.0)
    pyglet.clock.schedule_interval(snapshot, 5)
    pyglet.clock.schedule_interval(draw2, 1)
    pyglet.app.run()

@window1.event
def on_key_press(symbol, modifiers):
    do_key_press(symbol, modifiers)
