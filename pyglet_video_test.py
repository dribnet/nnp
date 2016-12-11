import pyglet
from pyglet import clock
import numpy as np
from scipy.misc import imread, imsave, imresize
from pyglet.gl import *

pyglet.have_avbin=False

# AVBin is needed for video playback
# http://avbin.github.io/AVbin/Download.html

# Loading Sound & Video using Pyglet
# https://pyglet.readthedocs.io/en/pyglet-1.2-maintenance/programming_guide/media.html

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

# Setup
window = pyglet.window.Window()

# Load a video source

video_file = 'videos/test1.mp4'
source = pyglet.media.load(video_file)
image_file = 'videos/test1.png'
#source = pyglet.media.load('videos/test2.mp4')
# source = pyglet.media.load('videos/test3.mp4')

im = imread(image_file, mode='RGB')
tex = image_to_texture(im)

video = source.video_format
player = pyglet.media.Player()
player.queue(source)
# player.queue(source)
# player.queue(source)
player.play()

print 'Video is %spx by %spx and %ss long' % (video.width, video.height, source.duration)

# Draw Loop
def draw(dt):
    global player
    window.clear()
    # print 'Current time is %ss' % (player.time)

    # Try to loop back to start if video reaches the end
    # Seems to freeze up if you try to seek to 0, or when the video ends 
    # if(player.time > source.duration-1):
    if player.time >= player.source.duration - 1:
        player.pause()
        player = pyglet.media.Player()
        source = pyglet.media.load(video_file)
        player.queue(source)
        # player.seek(0.01)
        player.play()


        # Player does not actually seek to 0.01 in this case, but instead starts at 3s into the video 
        # player.seek(0.01)
        #player.play()
    
    if player.time >= 0.4:
        player.get_texture().blit(0, 0)
    else:
        tex.blit(0,0)
    return

clock.schedule(draw)
pyglet.app.run()
