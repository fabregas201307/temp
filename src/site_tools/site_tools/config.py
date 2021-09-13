# Needs to be set the same as classes.json from labelbox
# Background class needs to be remapped to 5
LABELS = ["Landscape", "Asphalt", "Concrete", "Gravel", "Rooftop", "Background"]

LABELMAP = {
    0: (11, 102, 35),  # Forest Green
    1: (49, 51, 53),  # Dark Gray
    2: (215, 188, 106),  # Morning Yellow
    3: (150, 34, 125),  # Sunset Read
    4: (2, 108, 181),  # Pacific Blue
    5: (255, 255, 255),  # White
}

# Color (BGR) to class
INV_LABELMAP = {
    (11, 102, 35): 0,
    (49, 51, 53): 1,
    (215, 188, 106): 2,
    (150, 34, 125): 3,
    (2, 108, 181): 4,
    (255, 255, 255): 5,
}

LABELMAP_RGB = {k: (v[2], v[1], v[0]) for k, v in LABELMAP.items()}

INV_LABELMAP_RGB = {v: k for k, v in LABELMAP_RGB.items()}
