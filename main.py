from model import *
import os, sys
import numpy as np
import json

from nbt.world import WorldFolder
from world_discover import get_block

try:
    import nbt
except ImportError:
    # nbt not in search path. Let's see if it can be found in the parent folder
    extrasearchpath = os.path.realpath(os.path.join(__file__,os.pardir,os.pardir))
    if not os.path.exists(os.path.join(extrasearchpath,'nbt')):
        raise
    sys.path.append(extrasearchpath)
    



world = WorldFolder("myWorld")

# load the block ids
f = open("blocks_ids.json", "r")
blocks_ids = json.load(f)


size_box = 32
window_size = 1
for x in range(window_size, size_box-window_size):
    for y in range(-64+window_size, 120-window_size):
        for z in range(window_size, size_box - window_size):
            window = np.zeros((window_size * 2 + 1, window_size * 2 + 1, window_size * 2 + 1))
            for wx in range(x-window_size, x+window_size+1):
                for wy in range(y-window_size, y+window_size+1):
                    for wz in range(z-window_size, z+window_size+1):
                        block = get_block(world, (wx, wy, wz))
                        wix = wx - (x-window_size)
                        wiy = wy - (y-window_size)
                        wiz = wz - (z-window_size)
                        window[wix, wiy, wiz] = blocks_ids[str(block)]
            print(window)


