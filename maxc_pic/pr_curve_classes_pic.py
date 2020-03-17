import os
import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_name','faster_rcnn_inception_v2','model name')

def main(argv):
    folder_name = 'evalpath'
    model_name =FLAGS.model_name

    list_dir_all = os.listdir(folder_name+'/'+model_name)
    Green_Bus = []
    Green_Circle = []
    Green_Left = []
    Green_Pedestrian = []
    Green_Right = []
    Green_Straight= []

    for file in list_dir_all:
        if 'Green_Bus.txt' in file:
            Green_Bus.append(file)
        if 'Green_Circle.txt' in file:
            Green_Circle.append(file)
        if 'Green_Left.txt' in file:
            Green_Left.append(file)      
        if 'Green_Pedestrian.txt' in file:
            Green_Pedestrian.append(file)
        if 'Green_Right.txt' in file:
            Green_Right.append(file)
        if 'Green_Straight.txt' in file:
            Green_Straight.append(file)

    prefix=folder_name+'/'+model_name+'/'
    print('Loading')
    precision_Green_Bus = np.loadtxt(prefix+Green_Bus[0])
    recall_Green_Bus = np.loadtxt(prefix+Green_Bus[1])

    precision_Green_Circle = np.loadtxt(prefix+Green_Circle[0])
    recall_Green_Circle = np.loadtxt(prefix+Green_Circle[1])

    precision_Green_Left = np.loadtxt(prefix+Green_Left[0])
    recall_Green_Left = np.loadtxt(prefix+Green_Left[1])

    precision_Green_Pedestrian = np.loadtxt(prefix+Green_Pedestrian[0])
    recall_Green_Pedestrian = np.loadtxt(prefix+Green_Pedestrian[1])

    precision_Green_Right = np.loadtxt(prefix+Green_Right[0])
    recall_Green_Right = np.loadtxt(prefix+Green_Right[1])
        
    precision_Green_Straight = np.loadtxt(prefix+Green_Straight[0])
    recall_Green_Straight= np.loadtxt(prefix+Green_Straight[1])

    print('All Loaded')

    # plot
    fig=plt.figure(dpi=300)
    plt.step(recall_Green_Bus,precision_Green_Bus,label='Green_Bus')
    plt.step(recall_Green_Circle,precision_Green_Circle,label='Green_Circle')
    plt.step(recall_Green_Left,precision_Green_Left,label='Green_Left')
    plt.step(recall_Green_Pedestrian,precision_Green_Pedestrian,label='Green_Pedestrian')
    plt.step(recall_Green_Right,precision_Green_Right,label='Green_Right')
    plt.step(recall_Green_Straight,precision_Green_Straight,label='Green_Straight')

    # plt.title(model_name) # TODO
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.legend(loc='lower left')
    fig.savefig(model_name+'_classes_prcurve',dpi=300)
    print('saved')
    
if __name__ == '__main__':
  app.run(main)