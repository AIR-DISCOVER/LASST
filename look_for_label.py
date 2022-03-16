from plyfile import PlyData

label_to_class = {
    0: "null",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refridgerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "nightstand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "other structure",
    39: "other furniture",
    40: "other properties",
}

a = PlyData.read("/home/tb5zhh/data/full/train/scene0402_00.ply")

labels = []
classes = []
for i in a.elements[0]['label']:
    if i not in labels:
        labels.append((i))
        classes.append((i, label_to_class[i]))

print(classes[::2])
print(classes[1::2])