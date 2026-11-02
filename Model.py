#imports
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as pyl

#variables

    #main data
main_data = pd.read_csv('ml-model-ps-by-ash/arcane_blast_logs.csv')
cols      = main_data.columns
nrml_data = (main_data[cols]-main_data[cols].min())/(main_data[cols].max() - main_data[cols].min()) #normalized data in a DataFrame
dmg_arr = nrml_data['Damage_Dealt'].to_numpy()
m         = 5000
alpha = 0.1
    #derived features
delta_x       = main_data.loc[:, 'Player_X'] - main_data.loc[:, 'Enemy_X']
delta_y       = main_data.loc[:, 'Player_Y'] - main_data.loc[:, 'Enemy_Y']
distance      = (delta_x**2 + delta_y**2)**0.5 #distance 
invert_distance = distance**(-1)
nrml_distance = (invert_distance-invert_distance.min())/(invert_distance.max() - invert_distance.min())
b = 0
    #other parameters
# exp = -1.0
# data_to_plot = {
#     'nrml_distance'   : np.array(nrml_distance)**exp,
#     'nrml_damage'     : np.array(nrml_data['Damage_Dealt']),
#     'nrml_spell_pow'  : np.array(nrml_data['Base_Spell_Power']),
#     'nrml_enemy_armor': np.array(nrml_data['Enemy_Armor']),
#     'nrml_delta_y'    : (delta_y - delta_y.min())/(delta_y.max() - delta_y.min()),
#     'nrml_delta_x'    : (delta_x - delta_x.min())/(delta_x.max() - delta_x.min())
# }

# #code

# fig, ax = pyl.subplots()
# ax.scatter('nrml_distance', 'nrml_damage', data = data_to_plot)
# ax.set_xlabel('nrml_distance')
# ax.set_ylabel('nrml_damage')
# pyl.show()

X = nrml_data[cols.drop(['Player_X','Player_Y','Enemy_X','Enemy_Y', 'Damage_Dealt'])]
X['nrml_distace'] = nrml_distance
X = X.to_numpy()
w = np.zeros(X.shape[1])

#functions
    #features

    #model
def f_wb(X_i, w, b):
    return np.dot(X_i, w) + b

    #square dot function
def sqrt_function(X, w, b, y, m):
    cost = 0
    for i in range(m):
        cost += (f_wb(X[i], w, b)-y)**2
    return 1/(2*m)*cost

    #finding change for w
def f1(X, w, b, y, m, j):
    temp = 0
    for i in range(m):
        temp += (f_wb(X[i], w, b) - y[i])*X[i][j]
    return temp

    #defining change for b
def f2(X, w, b, y, m):
    temp = 0
    for i in range(m):
        temp += (f_wb(X[i], w, b) -y[i])
    return temp

iter = 1000
    #gradient descent algo
def grad_des(X, w, b, y, alpha, iter):
    for h in range(iter):
        for j in range(len(w)):
            w[j] -= alpha*f1(X, w, b, y, m, j)
            b -= alpha*f2(X, w, b, y, m)
        if h%100 == 0:
            print(w, b)
        
    return w, b

print(grad_des(X, w, b, dmg_arr, alpha, iter))