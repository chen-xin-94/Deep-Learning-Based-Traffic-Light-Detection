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
FLAGS = flags.FLAGS

LABEL_DICT =  {
  "Circle" : 0,
  "Straight" : 1,
  "Left" : 2,
  "StraightLeft" : 3,
  "Right": 4,
  "Bus":7, # in digital 3
  "Pedestrian": 8, 
  "Bike": 9
  }
  # classes are saved as in bstld

LABEL_DICT_R={v: k for k, v in LABEL_DICT.items()}

def create_tf_example(example,appearances):
  
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
    # adding box, one image may have multiple detected boxes
    class_id_d3=int(str(box['class_id'])[2])
    if  class_id_d3==7:
      class_id=-1
    else:
      class_id=str(box['class_id'])[-1]
      class_id=int(class_id)
      
    if class_id == 3: # ignore "StraightLeft"
      continue
    if class_id == 9: # ignore "Bike"
      continue
    if box['x'] + box['width'] > width or box['y']+ box['height'] > height:
      continue


    xmins.append(float(box['x']) / width)
    xmaxs.append(float(box['x'] + box['width']) / width)
    ymins.append(float(box['y']) / height)
    ymaxs.append(float(box['y']+ box['height']) / height)
    
    if class_id_d3==7:
      classes_text.append((LABEL_DICT_R[class_id_d3]).encode())
      appearances[LABEL_DICT_R[class_id_d3]]+=1
    else:
      classes_text.append((LABEL_DICT_R[class_id]).encode())
      appearances[LABEL_DICT_R[class_id]]+=1      
    # to match with bstld_label_map

    if  class_id_d3==7:
      #classes.append(class_id_d3)
      classes.append(int(6))


    elif class_id==0 or class_id==1 or class_id==2:
       classes.append(class_id+1)

    elif class_id == 4:
      classes.append(int(4))
 
    elif class_id == 8:
      classes.append(int(5))

    # elif class_id == 9:
    #   classes.append(int(6))


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
  appearances={k:0 for k in LABEL_DICT.keys()}

  for example in examples:
    tf_example,appearances = create_tf_example(example,appearances)

    writer.write(tf_example.SerializeToString())

    if counter % 10 == 0:
      print("Percent done", (counter/len_examples)*100)
    counter += 1.
  writer.close()

  # print appearances
  print()
  print('appearances') 
  for key, label in appearances.items():
      print('\t{}: {}'.format(key, label))


if __name__ == '__main__':
  tf.app.run()
