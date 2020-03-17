import os
import numpy as np
import matplotlib.pyplot as plt

folder_name = 'evalpath'
model_name = 'faster_rcnn_inception_v2'

list_dir_all = os.listdir(folder_name+'/'+model_name)

Red_Circle = []
Yellow_Circle = []

list_dir_all_b5 = os.listdir(folder_name+'/'+model_name+'_b5_2') # only _2 is tested on the correct test set
Red_Circle_b5 = []
Yellow_Circle_b5 = []

list_dir_all_b10 = os.listdir(folder_name+'/'+model_name+'_b10')
Red_Circle_b10 = []
Yellow_Circle_b10 = []


for file in list_dir_all:
    if 'Red_Circle.txt' in file:
        Red_Circle.append(file)
    if 'IOUYellow_Circle.txt' in file:
        Yellow_Circle.append(file)

for file_b5 in list_dir_all_b5:
    if 'Red_Circle.txt' in file_b5:
        Red_Circle_b5.append(file_b5)
    if 'IOUYellow_Circle.txt' in file_b5:
        Yellow_Circle_b5.append(file_b5)
        
for file_b10 in list_dir_all_b10:
    if 'Red_Circle.txt' in file_b10:
        Red_Circle_b10.append(file_b10)
    if 'IOUYellow_Circle.txt' in file_b10:
        Yellow_Circle_b10.append(file_b10)

prefix=folder_name+'/'+model_name+'/'
print ('Loading')
precision_Red_Circle = np.loadtxt(prefix+Red_Circle[0])
recall_Red_Circle = np.loadtxt(prefix+Red_Circle[1])
precision_Yellow_Circle = np.loadtxt(prefix+Yellow_Circle[0])
recall_Yellow_Circle = np.loadtxt(prefix+Yellow_Circle[1])

prefix5=folder_name+'/'+model_name+'_b5/'
precision_Red_Circle_b5 = np.loadtxt(prefix5+Red_Circle_b5[0])
recall_Red_Circle_b5 = np.loadtxt(prefix5+Red_Circle_b5[1])
precision_Yellow_Circle_b5 = np.loadtxt(prefix5+Yellow_Circle_b5[0])
recall_Yellow_Circle_b5 = np.loadtxt(prefix5+Yellow_Circle_b5[1])

prefix10=folder_name+'/'+model_name+'_b10/'
precision_Red_Circle_b10 = np.loadtxt(prefix10+Red_Circle_b10[0])
recall_Red_Circle_b10 = np.loadtxt(prefix10+Red_Circle_b10[1])
precision_Yellow_Circle_b10 = np.loadtxt(prefix10+Yellow_Circle_b10[0])
recall_Yellow_Circle_b10 = np.loadtxt(prefix10+Yellow_Circle_b10[1])

print ('All loaded')
# plot
fig=plt.figure(dpi=300)
plt.step(recall_Red_Circle,precision_Red_Circle,label='all')
plt.step(recall_Red_Circle_b5,precision_Red_Circle_b5,label='width > 5')
plt.step(recall_Red_Circle_b10,precision_Red_Circle_b10,label='width > 10')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')

fig2=plt.figure(dpi=300)
plt.step(recall_Yellow_Circle,precision_Yellow_Circle,label='all')
plt.step(recall_Yellow_Circle_b5,precision_Yellow_Circle_b5,label='width > 5')
plt.step(recall_Yellow_Circle_b10,precision_Yellow_Circle_b10,label='width > 10')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')

fig.savefig(model_name+'_pixels_prcurve_RC',dpi=300)
fig2.savefig(model_name+'_pixels_prcurve_YC',dpi=300)
print ('Saved')