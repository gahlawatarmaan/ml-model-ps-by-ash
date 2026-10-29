#imports
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as pyl

#variables

    #main data
main_data = pd.read_csv('D:/Armaans stuff/Projects/PS Arcane Bolt/arcane_blast_logs.csv')
cols      = main_data.columns
nrml_data = (main_data[cols]-main_data[cols].min())/(main_data[cols].max() - main_data[cols].min()) #normalized data in a DataFrame
m         = 5000
    #derived features
delta_x       = main_data.loc[:, 'Player_X'] - main_data.loc[:, 'Enemy_X']
delta_y       = main_data.loc[:, 'Player_Y'] - main_data.loc[:, 'Enemy_Y']
distance      = (delta_x**2 + delta_y**2)**0.5 #distance 
nrml_distance = (distance-distance.min())/(distance.max() - distance.min())

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

#functions
    #features
    