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
    "horizontal" : 1,
    "vertical" : 2,
    "horizontalbus" : 6,
    "verticalbus" : 7
    }
    LABEL_DICT_R={v: k for k, v in LABEL_DICT.items()}

    images = get_all_labels(input_yaml)

    appearances =  {
        "horizontal" : 0,
        "vertical" : 0,
        "horizontalbus" : 0,
        "verticalbus" : 0
        }

    for image in images:

        for box in image['objects']:
            class_str=str(box['class_id'])[2]
            appearances[LABEL_DICT_R[int(class_str)]]+=1

    print('Labels:')
    for key, label in appearances.items():
        print('\t{}: {}'.format(key, label))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    quick_stats(sys.argv[1])
