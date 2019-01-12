from game2048.agents import Agent
from keras.models import load_model
import numpy as np
from MYADDini import grid_ohe

model=load_model("logs/iris_model.h5")

class MyOwnAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.search_func = model.predict

    def step(self):
        direction = self.search_func(np.expand_dims(grid_ohe(self.game.board),axis=0)).argmax()
        return direction
