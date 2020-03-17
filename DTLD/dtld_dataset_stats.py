# TOO SLOW, USE ANOTHER METHODE
import cv2
import yaml 
import numpy as np
import os.path
import copy
import time
import argparse

LABEL_DICT =  {
    "Off" : 0,
    "Red" : 1,
    "Yellow" : 2,
    "Red-yellow" : 3,
    "Green":4
    }
LABEL_DICT_R={v: k for k, v in LABEL_DICT.items()}

class DriveuObject():
  """ Class describing a label object in the dataset by rectangle

  Attributes:
    x (int):          X coordinate of upper left corner of bouding box label
    y (int):          Y coordinate of upper left corner of bouding box label
    width (int):      Width of bounding box label
    height (int):     Height of bounding box label
    class_id (int):   6 Digit class idenntity of bounding box label (Digit explanation see documentation pdf)
    uniqie_id (int):  Unique ID of the object
    track_id (string) Track ID of the object (representing one real-world TL instance)

  """

  def __init__(self):
    self.x = 0
    self.y = 0
    self.width = 0
    self.height = 0
    self.class_id = 0
    self.unique_id = 0
    self.track_id = 0


  def colorFromClassId(self):
    """ Color for bounding box visualization

    Returns:
      Color-Vector (BGR) for traffic light visualization

    """
     #Second last digit indicates state/color
    if str(self.class_id)[-2] == "1":
      return (0,0,255)
    elif str(self.class_id)[-2] == "2":
      return (0,255,255)
    elif str(self.class_id)[-2] == "3":
      return (0,165,255)
    elif str(self.class_id)[-2] == "4":
      return (0,255,0)
    else:
      return (255,255,255)

class DriveuImage():
  """ Class describing one image in the DriveU Database

  Attributes:
    file_path (string):         Path of the left camera image
 
    timestamp (float):          Timestamp of the image

    objects (DriveuObject)      Labels in that image
    """
  def __init__(self):
    self.file_path = ''

    self.timestamp = 0

    self.objects = []

  def getImage(self):
    """ Method loading the left unrectified color image in 8 Bit

    Returns:
      8 Bit BGR color image

    """
    if os.path.isfile(self.file_path):
      """Load image from file path, do debayering and shift"""
      img = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
      img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
      # Images are saved in 12 bit raw -> shift 4 bits
      img = np.right_shift(img, 4)
      img = img.astype(np.uint8)

      return True, img

    else:

      print ("Image " + str(self.file_path) + "not found")

      # return False, img

  def getLabeledImage(self):
    """Method loading the left unrectified color image and drawing labels in it

    Returns:
      Labeled 8 Bit BGR color image

    """

    status, img = self.getImage()

    for o in self.objects:
      cv2.rectangle(img, (o.x, o.y), (o.x + o.width, o.y + o.height), o.colorFromClassId(), 2)

    return img


class DriveuDatabase():
  """ Class describing the DriveU Dataset

  Attributes:
    images (List of DriveuImage)  All images of the dataset
    file_path (string):           Path of the dataset (.yml)
    """
  def __init__(self, file_path):
    self.images = []
    self.file_path = file_path

  def open(self, data_base_dir):
    """Method loading the dataset

    """

    if os.path.exists(self.file_path) is not None:
      print ('Opening DriveuDatabase from File: ' + str(self.file_path))
      images = yaml.load(open(self.file_path, 'rb').read())

      for i, image_dict in enumerate(images):

        image = DriveuImage()
        if data_base_dir != '':
            inds = [i for i, c in enumerate(image_dict['path']) if c == '/']
            image.file_path = data_base_dir + '/' + image_dict['path'][inds[-4]:]

            print (image.file_path)
        else:
            image.file_path = image_dict['path']

        image.timestamp = image_dict['time_stamp']

        for o in image_dict['objects']:

          object = DriveuObject()
          object.x = o['x']
          object.y = o['y']
          object.width = o['width']
          object.height = o['height']
          object.class_id = o['class_id']
          object.unique_id = o['unique_id']
          object.track_id = o['track_id']

          cpy = copy.copy(object)

          image.objects.append(cpy)

        copy_image = copy.copy(image)
        self.images.append(copy_image)

    else:
      print ('Opening DriveuDatabase from File: ' + str(self.file_path) + 'failed. File or Path incorrect.')

def parse_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--label_file', default='')
  parser.add_argument('--data_base_dir', default='')
  return parser.parse_args()


def main(args):

  database = DriveuDatabase(args.label_file)

  database.open(args.data_base_dir)

  widths = []
  heights = []
  sizes = []

  num_images = len(database.images)
  num_lights = 0
  large=0
  medium=0
  small=0

  appearances =  {
    "Off" : 0,
    "Red" : 0,
    "Yellow" : 0,
    "Red-yellow" : 0,
    "Green":0
    }

  for image in database.images:
    num_lights += len(image.objects)
    for box in image.objects:
      class_str=str(box.class_id)[-2]
      appearances[LABEL_DICT_R[int(class_str)]]+=1
      widths.append(box.width)
      heights.append(box.height)

      size=box.width*box.height
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

if __name__ == '__main__':
  main(parse_args())


