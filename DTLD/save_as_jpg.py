import cv2
import yaml 
import numpy as np
import os
import copy
import argparse

# CLASS_DICT =  {
#     "Off" : 0,
#     "Red" : 1,
#     "Yellow" : 2,
#     "Red-yellow" : 3,
#     "Green":4
#     }

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

      return img

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

            # print (image.file_path)
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

  counter=0
  len_database=len(database.images)

  for img in database.images:

    img_color = img.getImage()

    jpeg_path=img.file_path.rstrip('.tiff')+'.jpg'
    jpeg_path=jpeg_path.replace('/DTLD/DTLD','/DTLD/DTLD_JPEG')

    # print(jpeg_path)
    jpeg_dir_path=os.path.dirname(jpeg_path)
    
    if not os.path.exists(jpeg_dir_path):
      os.makedirs(jpeg_dir_path)
    cv2.imwrite(jpeg_path,img_color)

    if counter % 10 == 0:
      print("Percent done", (counter/len_database)*100)
    counter += 1.


if __name__ == '__main__':
  # tf.app.run()
  main(parse_args())
