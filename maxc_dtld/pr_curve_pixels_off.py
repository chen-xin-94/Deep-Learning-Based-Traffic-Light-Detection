import os
import numpy as np
import matplotlib.pyplot as plt

folder_name = 'evalpath'
model_name = 'faster_rcnn_inception_v2'

list_dir_all = os.listdir(folder_name+'/'+model_name+'_test')

green = []
red = []
yellow = []
off = []

list_dir_all_b5 = os.listdir(folder_name+'/'+model_name+'_b5')
green_b5 = []
red_b5 = []
yellow_b5 = []
off_b5 = []

list_dir_all_b10 = os.listdir(folder_name+'/'+model_name+'_b10')
green_b10 = []
red_b10 = []
yellow_b10 = []
off_b10 = []



for file in list_dir_all:
    if 'Green.txt' in file:
        green.append(file)
    if 'Red.txt' in file:
        red.append(file)
    if 'Yellow.txt' in file:
        yellow.append(file)
    if 'Off.txt' in file:
        off.append(file)

for file_b5 in list_dir_all_b5:
    if 'Green.txt' in file_b5:
        green_b5.append(file_b5)
    if 'Red.txt' in file_b5:
        red_b5.append(file_b5)
    if 'Yellow.txt' in file_b5:
        yellow_b5.append(file_b5)
    if 'Off.txt' in file_b5:
        off_b5.append(file_b5)
        
for file_b10 in list_dir_all_b10:
    if 'Green.txt' in file_b10:
        green_b10.append(file_b10)
    if 'Red.txt' in file_b10:
        red_b10.append(file_b10)
    if 'Yellow.txt' in file_b10:
        yellow_b10.append(file_b10)
    if 'Off.txt' in file_b10:
        off_b10.append(file_b10)

prefix=folder_name+'/'+model_name+'/'
print ('Loading')
precision_green = np.loadtxt(prefix+green[0])
recall_green = np.loadtxt(prefix+green[1])
precision_red = np.loadtxt(prefix+red[0])
recall_red = np.loadtxt(prefix+red[1])
precision_yellow = np.loadtxt(prefix+yellow[0])
recall_yellow = np.loadtxt(prefix+yellow[1])
precision_off = np.loadtxt(prefix+off[0])
recall_off = np.loadtxt(prefix+off[1])

prefix5=folder_name+'/'+model_name+'_b5/'
precision_green_b5 = np.loadtxt(prefix5+green_b5[0])
recall_green_b5 = np.loadtxt(prefix5+green_b5[1])
precision_red_b5 = np.loadtxt(prefix5+red_b5[0])
recall_red_b5 = np.loadtxt(prefix5+red_b5[1])
precision_yellow_b5 = np.loadtxt(prefix5+yellow_b5[0])
recall_yellow_b5 = np.loadtxt(prefix5+yellow_b5[1])
precision_off_b5 = np.loadtxt(prefix5+off_b5[0])
recall_off_b5 = np.loadtxt(prefix5+off_b5[1])

prefix10=folder_name+'/'+model_name+'_b10/'
precision_green_b10 = np.loadtxt(prefix10+green_b10[0])
recall_green_b10 = np.loadtxt(prefix10+green_b10[1])
precision_red_b10 = np.loadtxt(prefix10+red_b10[0])
recall_red_b10 = np.loadtxt(prefix10+red_b10[1])
prefix10=folder_name+'/'+model_name+'_b10/'
precision_yellow_b10 = np.loadtxt(prefix10+yellow_b10[0])
recall_yellow_b10 = np.loadtxt(prefix10+yellow_b10[1])
precision_off_b10 = np.loadtxt(prefix10+off_b10[0])
recall_off_b10 = np.loadtxt(prefix10+off_b10[1])
print ('All loaded')

# plot
fig=plt.figure(dpi=300)
plt.step(recall_green,precision_green,label='all')
plt.step(recall_green_b5,precision_green_b5,label='width > 5')
plt.step(recall_green_b10,precision_green_b10,label='width > 10')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')

fig2=plt.figure(dpi=300)
plt.step(recall_red,precision_red,label='all')
plt.step(recall_red_b5,precision_red_b5,label='width > 5')
plt.step(recall_red_b10,precision_red_b10,label='width > 10')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')

fig3=plt.figure(dpi=300)
plt.step(recall_yellow,precision_yellow,label='all')
plt.step(recall_yellow_b5,precision_yellow_b5,label='width > 5')
plt.step(recall_yellow_b10,precision_yellow_b10,label='width > 10')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')

fig4=plt.figure(dpi=300)
plt.step(recall_off,precision_off,label='all')
plt.step(recall_off_b5,precision_off_b5,label='width > 5')
plt.step(recall_off_b10,precision_off_b10,label='width > 10')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')

fig.savefig(model_name+'_pixels_prcurve_G',dpi=300)
fig2.savefig(model_name+'_pixels_prcurve_R',dpi=300)
fig3.savefig(model_name+'_pixels_prcurve_Y',dpi=300)
fig4.savefig(model_name+'_pixels_prcurve_O',dpi=300)
print ('Saved')