#import keras
from MYADDmodel import model
from MYADDtrain import ModelWrapper, Guides
from game2048.game import Game
from keras.models import load_model


mymodel=model=load_model("logs/iris_model.h5")
myTrain=ModelWrapper(mymodel,100000)




for i in range (0,300000):
    mygame=Game()
    if i%10==0:
        myTrain.memory=Guides(100000)
    #print("the end is %d, score is %d"%(mygame.end,mygame.score))
    while (mygame.end==0):
        myTrain.move(mygame)
        #print("the while is working at %d"%i)
    myTrain.train(1000)
    print("round complete %i, the score is %d" %(i,mygame.score))
    if i%100==0:
        mp="logs/iris_model.h5"
        model.save(mp)
