import keras
import numpy as np
import tensorflow as tf
from collections import namedtuple
import random
from MYADDini import grid_ohe



Guide=namedtuple('Guide',('state','action'))
class Guides:
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]
        self.position=0

    def push(self,*args):
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position]=Guide(*args)
        self.position=(self.position+1)%self.capacity

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def ready(self,batch_size):
        return len(self.memory)>=batch_size

    def _len_(self):
        return len(self.memory)


class ModelWrapper:
    def __init__(self,model,capacity):
        self.model=model
        self.memory=Guides(capacity)
        self.writer=tf.summary.FileWriter("logs",tf.get_default_graph())          
        self.training_step=0

    def predict(self,board):
        return self.model.predict(np.expand_dims(board,axis=0))       

    def move(self,game):
        ohe_board=grid_ohe(game.board)

        from game2048.expectimax import board_to_move
        suggest=board_to_move(game.board)

        direction=self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board,suggest)
        #print("the move work correct")

    def train(self,batch):
        if self.memory.ready(batch):
            print("this train is on!!")
            guides=self.memory.sample(batch)
            X=[]
            Y=[]
            for guide in guides:
                X.append(guide.state)
                ohe_action=[0]*4
                ohe_action[guide.action]=1
                Y.append(ohe_action)
            loss,acc=self.model.train_on_batch(np.array(X),np.array(Y))
            #self.writer.add_scalar('loss',float(loss),self.training_step)
            #self.writer.add_scalar('acc',float(acc),self.training_step)
            self.training_step+=1
            print("at step %d, the acc is %f, the loss is %f" %(self.training_step,acc,loss))
        else: print("sorry, not ready yet")
