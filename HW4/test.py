import struct
offset =  0
start =offset
max_length = 0
max_id = 0
prev_id = -1
with open("postings.txt","r") as f:
    while True:
        f.seek(offset)
        _id, pos, zone = struct.unpack("III", f.read(12))
        if _id == 6807771:
            print _id, pos, zone, offset
        offset+=12