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

    green = []
    red = []
    yellow = []
    Redyellow = []

    for file in list_dir_all:
        if 'Green.txt' in file:
            green.append(file)
        if 'Red.txt' in file:
            red.append(file)
        if 'Yellow.txt' in file:
            yellow.append(file)      
        if 'Red-yellow.txt' in file: #TODO
            Redyellow.append(file)

    prefix=folder_name+'/'+model_name+'/'

    print('Loading')
    precision_green = np.loadtxt(prefix+green[0])
    recall_green = np.loadtxt(prefix+green[1])

    precision_red = np.loadtxt(prefix+red[0])
    recall_red = np.loadtxt(prefix+red[1])

    precision_yellow = np.loadtxt(prefix+yellow[0])
    recall_yellow = np.loadtxt(prefix+yellow[1])

    precision_Redyellow = np.loadtxt(prefix+Redyellow[0])
    recall_Redyellow = np.loadtxt(prefix+Redyellow[1])

    print('All Loaded')

    # plot
    fig=plt.figure(dpi=300)
    plt.step(recall_green,precision_green,label='Green',color='green')
    plt.step(recall_red,precision_red,label='Red',color='red')
    plt.step(recall_yellow,precision_yellow,label='Yellow',color='yellow')
    plt.step(recall_Redyellow,precision_Redyellow,label='Red-yellow',color='black')

    # plt.title(model_name) # TODO
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.legend(loc='lower left')
    fig.savefig(model_name+'_classes_prcurve',dpi=300)
    print('saved')

if __name__ == '__main__':
  app.run(main)