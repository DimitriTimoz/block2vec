import json
import urllib
import os
import urllib.request 

# this script convert  this json
# https://github.com/PrismarineJS/minecraft-data/blob/master/data/pc/1.20.2/blocks.json
# to a lighter json file`
#

urllib.request.urlretrieve("https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/pc/1.20.2/blocks.json", "blocks.json")

f = open("blocks.json", "r")
blocks = json.load(f)
os.remove("blocks.json")
ids = {}
for block in blocks:
    id = block["id"]
    name = "minecraft:" + block["name"]
    ids[name] = id
    
f = open("blocks_ids.json", "w")
json.dump(ids, f)