# Septian Wijaya - 16/398528/PA/17489

import numpy as np
import MLP_iris_module 
from pathlib import Path

#Fetch Data
path = Path(__file__).parents[0]
inputFile = str(path) + "\\Iris.csv"
data = np.genfromtxt(inputFile, skip_header=True, delimiter=',') 

#Run MLP on Data
mlp = MLP_iris_module.MLP(data=data, input_size=4, hidden_layers_size=5,out_layer_size=2, epoch=2000, learn_rate=0.1).run()
