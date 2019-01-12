import time
from game2048.displays import Display
from MYADDmodel import model
from MYADDtrain import ModelWrapper, Guides
from game2048.game import Game
from keras.models import load_model
from game2048.MYADDagent import MyOwnAgent as TestAgent
import keras
import numpy as np
import tensorflow as tf
from collections import namedtuple
import random
from MYADDini import grid_ohe

mygame=Game(4,2048)
mymodel=model=load_model("logs/iris_model.h5")
myTrain=ModelWrapper(mymodel,100000)


ohe_board=grid_ohe(mygame.board)

a=time.time()
direction=myTrain.predict(ohe_board).argmax()
b=time.time()

print("time is %f" %(b-a))
