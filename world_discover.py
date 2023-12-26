from nbt.region import InconceivedChunk
import numpy as np

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

def get_block_in_chunk(chunk, cx, cy, cz):
    block_states = chunk["block_states"]
    palette = block_states["palette"]
    bits_per_value = max(int(np.ceil(np.log2(len(palette)))), 1)
    if "data" in block_states.keys():
        data = block_states["data"]
        block_states_indices = parse_block_data(data, bits_per_value)
        # Now map these indices to blocks using the palette
        blocks = [palette[index] if index < len(palette) else None for index in block_states_indices]
        index = (((cy * 16) + cz) * 16) + cx
        if blocks[index] is None:
            print("none")
            return {"Name": "minecraft:air"}
        return blocks[index]
    else:
        block = block_states[0][0]
        return block

def get_block(world, pos):
    x, y, z = pos
    rx,cx = divmod(x, 32)
    rz,cz = divmod(z, 32)
    if (rx,rz) not in world.regions and (rx,rz) not in world.regionfiles:
        return
    region = world.get_region(rx,rz)
    try:
        nbt = region.get_nbt(cx,cz)
    except InconceivedChunk:
        return "minecraft:air"
    chunk_column = nbt.tags[9]
    y += 64
    cy = y // 16
    chunk = chunk_column[cy]
    cx, cy, cz = (x % 16, y % 16, z % 16)
    block = get_block_in_chunk(chunk, cx, cy, cz)
    block = block["Name"]
    return block