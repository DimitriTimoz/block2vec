from model import *
import world

# Hyperparameters
embedding_dim = 10
epochs = 50

with world.World('myWorld') as myWorld:
    # All locations are stored as tuples
    myBlockPos = (15, 10, 25)
    # Get the block object at the given location
    myBlock = myWorld.get_block(myBlockPos)
    print(myBlock)
