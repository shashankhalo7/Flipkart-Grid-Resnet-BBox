import sys
import subprocess
import pandas as pd
import os.path
import os
from fastai import *
from fastai.vision import *

def denorm_y(y):return (y+1)*240
def denorm_x(x):return (x+1)*320
path=Path(__file__).parent
path_test = sys.argv[1]
learn = load_learner(path,'model_learn.pkl')
learner_top = load_learner(path,'model_learner_top.pkl')


file_list = os.listdir(path_test)
df = pd.DataFrame({'image_name': file_list})

x1=list()
y1=list()
x2=list()
y2=list()

for row in df.iterrows():
    fname=row[1]['image_name']
    img = open_image(f'{path_test}/{fname}')
    top=learner_top.predict(img)
    bottom=learn.predict(img)
    top_coord=np.array(top[1])
    bottom_coord=np.array(bottom[1])
    x1.append(denorm_x(top_coord[0][1]))
    y1.append(denorm_y(top_coord[0][0]))
    x2.append(denorm_x(bottom_coord[0][1]))
    y2.append(denorm_y(bottom_coord[0][0]))


df['x1']=x1
df['x2']=x2
df['y1']=y1
df['y2']=y2

df.to_csv('round3_final_prediction.csv')
