#!/usr/bin/env python3
"""
Sample script to receive traffic light labels and images
of the Bosch Small Traffic Lights Dataset.

Example usage:
    python read_label_file.py input_yaml
"""

import os
import sys
import yaml


def get_all_labels(input_yaml):
    """ Gets all labels within label file

    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    Args:
        input_yaml->str: Path to yaml file
        riib->bool: If True, change path to labeled pictures
        clip->bool: If True, clips boxes so they do not go out of image bounds
    Returns: Labels for traffic lights
    """
    assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
    with open(input_yaml, 'rb') as iy_handle:
        images = yaml.load(iy_handle)

    if not images or not isinstance(images[0], dict) or 'path' not in images[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

    return images


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    get_all_labels(sys.argv[1])
