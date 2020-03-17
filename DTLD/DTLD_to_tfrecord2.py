'''
Usage: python ./dtld_to_tfrecord.py --input_yaml input_file_name.yaml --output_path output_file_name.record
'''

import tensorflow as tf
import yaml
import os, sys
import io
from PIL import Image
# from utilities import dataset_util

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
  # classes are saved as in bstld

LABEL_DICT_R={v: k for k, v in LABEL_DICT.items()}

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
    class_id = str(box['class_id'])[-2]
    if box['width'] < 0:
      continue
    if class_id =='0': # ignore "off"
      continue
    if box['x'] + box['width'] > width or box['y']+ box['height'] > height:
      continue

    xmins.append(float(box['x']) / width)
    xmaxs.append(float(box['x'] + box['width']) / width)
    ymins.append(float(box['y']) / height)
    ymaxs.append(float(box['y']+ box['height']) / height)

    label_text=LABEL_DICT_R[int(class_id)]
    classes_text.append(label_text.encode())
    appearances[label_text]+=1

    # to match with bstld_label_map
    if class_id == '3':
      classes.append(int(1))
    elif class_id == '1':
      classes.append(int(4))
    elif class_id == '2':
      classes.append(int(3))
    elif class_id == '4':
      classes.append(int(2))


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
  appearances={k:0 for k in LABEL_DICT.keys()}
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
