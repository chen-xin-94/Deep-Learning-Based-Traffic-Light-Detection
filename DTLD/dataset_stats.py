#!/usr/bin/env python
"""
Sample script to show some numbers for the dataset.

Example usage:
    python dataset_stats.py input_yaml
"""

import sys
import logging
from read_label_file import get_all_labels


def quick_stats(input_yaml):
    """
    Prints statistic data for the traffic light yaml files.

    :param input_yaml: Path to yaml file of published traffic light set
    """
    LABEL_DICT =  {
    "Off" : 0,
    "Red" : 1,
    "Yellow" : 2,
    "Red-yellow" : 3,
    "Green":4
    }
    LABEL_DICT_R={v: k for k, v in LABEL_DICT.items()}

    LABEL_DICT_P =  {
        "Circle" : 0,
        "Straight" : 1,
        "Left" : 2,
        "StraightLeft" : 3,
        "Right": 4,
        "Pedestrian": 8,
        "Bike": 9
        }
        # classes are saved as in bstld

    LABEL_DICT_P_R={v: k for k, v in LABEL_DICT_P.items()}

    images = get_all_labels(input_yaml)

    widths = []
    heights = []
    sizes = []

    num_images = len(images)
    num_lights = 0
    appearances =  {
        "Off" : 0,
        "Red" : 0,
        "Yellow" : 0,
        "Red-yellow" : 0,
        "Green":0
        }
    appearances_P =  {
        "Circle" : 0,
        "Straight" : 0,
        "Left" : 0,
        "StraightLeft" : 0,
        "Right": 0,
        "Pedestrian": 0,
        "Bike": 0
        }

    large=0
    medium=0
    small=0

    for image in images:
        num_lights += len(image['objects'])
        for box in image['objects']:
            class_str=str(box['class_id'])[-2]
            class_str_P=str(box['class_id'])[-1]

            appearances[LABEL_DICT_R[int(class_str)]]+=1
            appearances_P[LABEL_DICT_P_R[int(class_str_P)]]+=1

            widths.append(box['width'])
            heights.append(box['height'])

            size=box['width']*box['height']
            sizes.append(size)

            if size<=(32*32):
                small+=1
            elif size>(96*96):
                large+=1
            else:
                medium+=1

    avg_width = sum(widths) / float(len(widths))
    avg_height = sum(heights) / float(len(heights))
    avg_size = sum(sizes) / float(len(sizes))

    median_width = sorted(widths)[len(widths) // 2]
    median_height = sorted(heights)[len(heights) // 2]
    median_size = sorted(sizes)[len(sizes) // 2]

    print('Number of images:', num_images)
    print('Number of traffic lights:', num_lights, '\n')

    print('Small images:', small)
    print('Medium images:', medium)
    print('Large images:', large, '\n')

    print('Minimum width:', min(widths))
    print('Average width:', avg_width)
    print('median width:', median_width)
    print('maximum width:', max(widths), '\n')

    print('Minimum height:', min(heights))
    print('Average height:', avg_height)
    print('median height:', median_height)
    print('maximum height:', max(heights), '\n')

    print('Minimum size:', min(sizes))
    print('Average size:', avg_size)
    print('median size:', median_size)
    print('maximum size:', max(sizes), '\n')

    print('Labels:')
    for key, label in appearances.items():
        print('\t{}: {}'.format(key, label))

    print()

    print('Labels_Pictogram:')
    for key, label in appearances_P.items():
        print('\t{}: {}'.format(key, label))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    quick_stats(sys.argv[1])
