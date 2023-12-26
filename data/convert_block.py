import json
import urllib
import os
import urllib.request 

# this script convert  this json
# https://github.com/PrismarineJS/minecraft-data/blob/master/data/pc/1.20.2/blocks.json
# to a lighter json file`
#

VERSION = "1.20.2"

urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/pc/{VERSION}/blocks.json", "blocks.json")
urllib.request.urlretrieve(f"https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/pc/{VERSION}/biomes.json", "biomes.json")

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

f = open("biomes.json", "r")
biomes = json.load(f)
os.remove("biomes.json")
ids = {}
for biome in biomes:
    id = biome["id"]
    name = "minecraft:" + biome["name"]
    ids[name] = id

f = open("biomes_ids.json", "w")
json.dump(ids, f)