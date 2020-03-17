'''
Usage: python ./dtld_to_tfrecord.py --input_yaml input_file_name.yaml --output_path output_file_name.record
'''

import tensorflow as tf
import yaml
import os, sys
import io
from PIL import Image
# from utilities import dataset_util
sys.path.append('C:/Users/xinch/Documents/Python/TensorFlow/models/research/object_detection')
from utils import dataset_util
import hashlib
from random import shuffle

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('input_yaml', '', 'Path to labeling YAML')
flags.DEFINE_string('width', '0', 'Ignore boxes with width less than this value')
FLAGS = flags.FLAGS
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

LABEL_DICT_P_R={v: k for k, v in LABEL_DICT_P.items()}

CLASS_DICT={
  "40":"Green_Circle",
  "41":"Green_Straight",
  "42":"Green_Left",
  "44":"Green_Right",
  "48":"Green_Pedestrian",
  "740":"Green_Bus",
  #"49":"Green_Bike",
  "10":"Red_Circle",
  "11":"Red_Straight",
  "12":"Red_Left",
  "14":"Red_Right",
  "18":"Red_Pedestrian",
  #"19":"Red_Bike",
  #"0":"Off",
  "710":"Red_Bus",
  "720":"Yellow_Bus",
  "20":"Yellow_Circle",
  "30":"Red_Yellow_Circle"
}

CLASS_ID_DICT={
  "40":1,
  "41":2,
  "42":3,
  "44":4,
  "48":5,
  "740":6,
  "10":7,
  "11":8,
  "12":9,
  "14":10,
  "18":11,
  "710":12,
  "720":13,
  "20":14,
  "30":15,
}

def create_tf_example(example,appearances,width_limit):
  
  filename = example['path'] # Filename of the image. Empty if image is not from file
  filename = filename.encode()

  with tf.gfile.GFile(example['path'], 'rb') as fid:
    encoded_image = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_image)
  image = Image.open(encoded_jpg_io)
  width, height = image.size
  key = hashlib.sha256(encoded_image).hexdigest()

  image_format = 'jpg'.encode() 

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
        # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
        # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)


  for box in example['objects']:
    if box['width'] <= width_limit:
      continue
    # adding box, one image may have multiple detected boxes
    class_id=str(box['class_id'])[-2]
    class_id_P=str(box['class_id'])[-1]
    class_id_B=str(box['class_id'])[2]

    if class_id == '0': # ignore "Off"
      continue
    if class_id_P == '3' or class_id_P == '9': # ignore "StraightLeft" and "Cyclist"
      continue
    if class_id == '2' and class_id_P != '0': # only consider yellow CIRCLE
      continue
    if class_id == '3' and class_id_P !='0': # ignore all pictograms of "Red-yellow" except "Red_Yellow_Circle"
      continue
    if class_id_B=='7' and class_id == '3': # ignore all pictograms of "Red-yellow" except "Red_Yellow_Circle"
      continue


    if class_id_B == '7' and class_id_P == '0':
      class_id_str = class_id_B + class_id + class_id_P
    else:
      class_id_str = class_id + class_id_P

    if box['x'] + box['width'] > width or box['y']+ box['height'] > height:
      continue
    
    xmins.append(float(box['x']) / width)
    xmaxs.append(float(box['x'] + box['width']) / width)
    ymins.append(float(box['y']) / height)
    ymaxs.append(float(box['y']+ box['height']) / height)

    classes_text.append((CLASS_DICT[class_id_str]).encode())
    classes.append(CLASS_ID_DICT[class_id_str])
    appearances[CLASS_DICT[class_id_str]]+=1

  if classes: # if classes are not empty
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example,appearances

  else:
    return '',appearances

def main(_):
      
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
      
  INPUT_YAML = FLAGS.input_yaml
  examples = yaml.load(open(INPUT_YAML, 'rb').read())

  len_examples = len(examples)
  print("Loaded ", len(examples), "examples")

  # for i in range(len(examples)):
  #   examples[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(INPUT_YAML), examples[i]['path']))
  
  shuffle(examples)
  
  counter = 0
  counter_pic=0
  appearances={k:0 for k in CLASS_DICT.values()}
  width_limit = int(FLAGS.width)

  for example in examples:
    tf_example,appearances = create_tf_example(example,appearances,width_limit)
    if tf_example:
      writer.write(tf_example.SerializeToString())
      counter_pic+=1

    if counter % 100 == 0:
      print("Percent done", (counter/len_examples)*100)  
    counter += 1.
    
  writer.close()
  # print appearances
  print()
  print("number of pictures: "+str(counter_pic))
  print('appearances') 
  for key, label in appearances.items():
      print('\t{}: {}'.format(key, label))

if __name__ == '__main__':
  tf.app.run()
