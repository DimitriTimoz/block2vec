from model import *
import locale, os, sys
import numpy as np

# local module
try:
    import nbt
except ImportError:
    # nbt not in search path. Let's see if it can be found in the parent folder
    extrasearchpath = os.path.realpath(os.path.join(__file__,os.pardir,os.pardir))
    if not os.path.exists(os.path.join(extrasearchpath,'nbt')):
        raise
    sys.path.append(extrasearchpath)
    
from nbt.world import WorldFolder

def to_section(y):
    return y // 16

def extract_bits(value, start, length):
    mask = (1 << length) - 1
    return (value >> start) & mask

def parse_block_data(data, bits_per_value):
    bits_per_long = 64
    block_states = []
    total_bits = len(data) * bits_per_long
    for bit_index in range(0, total_bits, bits_per_value):
        start_long = bit_index // bits_per_long
        start_offset = bit_index % bits_per_long
        end_offset = start_offset + bits_per_value

        # Check if the bits are contained within a single long
        if end_offset <= bits_per_long:
            state = extract_bits(data[start_long], start_offset, bits_per_value)
        else:
            # Bits span across two integers
            first_part = extract_bits(data[start_long], start_offset, bits_per_long - start_offset)
            
            # Check if the next long is within bounds
            if start_long + 1 < len(data):
                second_part = extract_bits(data[start_long + 1], 0, end_offset - bits_per_long)
                state = (second_part << (bits_per_long - start_offset)) | first_part
            else:
                # Handle the case where the second part is out of bounds
                state = first_part << (end_offset - bits_per_long)

        block_states.append(state)

    return block_states

world = WorldFolder("myWorld")

for region in world.iter_regions():
    for chunk_columns in region.iter_chunks():
        for chunk in chunk_columns["sections"]:
            y_col = chunk["Y"]
            block_states = chunk["block_states"]
            palette = block_states["palette"]
            bits_per_value = max(int(np.ceil(np.log2(len(palette)))), 1)
            if "data" in block_states.keys():
                data = block_states["data"]
                block_states_indices = parse_block_data(data, bits_per_value)
                # Now map these indices to blocks using the palette
                blocks = [palette[index] if index < len(palette) else None for index in block_states_indices]
                for block in blocks:
                    if block is None:
                        continue
                    print(block["Name"])

