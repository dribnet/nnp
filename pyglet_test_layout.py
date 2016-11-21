import math
import pyglet
from pyglet import clock
from pyglet.window import key

class layout():
    def __init__(self, image_path, cols, height, window):
        self.image_path = image_path
        self.cols = cols
        self.window = window
        self.height = height # as a percentage of screen height (eg. 0.3)
        self.setup()

    def setup(self):
        # loading images
        self.image     = pyglet.image.load(self.image_path)
        self.image_seq = pyglet.image.ImageGrid(self.image, 1, self.cols)
        # setting up sprites
        self.start     = pyglet.sprite.Sprite(self.image_seq[0] )
        self.sequence  = []

        for i in range(1, len(self.image_seq)):
            self.sequence.append(pyglet.sprite.Sprite(self.image_seq[i])) 

        self.result    = self.sequence[-1]
        return 

    def change_image(self, image_path, cols):
        self.image_path = image_path
        self.cols = cols
        self.setup()

    def draw(self):
        height = window.height*self.height - self.start.height/2
        left   = window.width/2 - len(self.sequence)*self.sequence[0].width/2
        right  = window.width-left
        space  = 30

        # draw start image 
        self.start.y = height
        self.start.x = left-space - self.start.width
        self.start.draw()
        
        # draw sequence
        for i in range(len(self.sequence)):
            seq = self.sequence[i]
            seq.x = left + seq.width*i
            seq.y = height
            seq.draw()
        
        # draw & animate result 
        frame = int(abs(math.sin(timecount*2))*self.cols-1)

        self.result = self.sequence[frame]
        self.result.y = height
        self.result.x = right+space
        self.result.draw()
        return

# Setup
window = pyglet.window.Window(resizable=True)
framecount = 0
timecount  = 0
num_sets   = 3
cur_set    = 0

layout_happy     = layout('images/happy_0.png', 8, 3/4., window)
layout_angry     = layout('images/angry_0.png', 8, 2/4., window)
layout_surprised = layout('images/surprised_0.png', 8, 1/4., window)

# Draw Loop
def draw(dt):
    global window, framecount, timecount
    window.clear()

    layout_happy.draw()
    layout_angry.draw()
    layout_surprised.draw()
    
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
    
    print "Switching to set " + str(cur_set)

    layout_happy.change_image('images/happy_' + str(cur_set) + '.png', 8)
    layout_angry.change_image('images/angry_' + str(cur_set) + '.png', 8)
    layout_surprised.change_image('images/surprised_' + str(cur_set) + '.png', 8)

@window.event
def on_key_press(symbol, modifiers):
    if(symbol == key.LEFT):
        change_set(-1)
    elif(symbol == key.RIGHT):
        change_set(1)

clock.schedule(draw)
pyglet.app.run()
