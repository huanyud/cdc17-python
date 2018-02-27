import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib

# To use Type 1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

seed = 4
num_of_source = 9
num_of_sink = 30
search_radius = 50

# Randomly generate agent and location coordinates
random.seed(seed)
rand_pos_agent = []
for i in range(3):
    for j in range(3):
        rand_pos_agent.append(random.choice([[x, y] for x in range(33*i, 33*(i+1)) for y in range(33*j, 33*(j+1))]))
rand_pos_location = random.sample([[x, y] for x in range(100) for y in range(100)], num_of_sink)
   
# Plot the agents and locations on the plane
fig = plt.figure()
ax = fig.add_subplot(111)
scatter_loc = plt.scatter(np.array(rand_pos_location)[:,0], np.array(rand_pos_location)[:,1])
scatter_uav = plt.scatter(np.array(rand_pos_agent)[:,0], np.array(rand_pos_agent)[:,1], marker='x')
for i in range(num_of_source):
    circle = Circle((rand_pos_agent[i][0], rand_pos_agent[i][1]), radius=search_radius, fill=False)
    ax.add_patch(circle)
plt.axis([0, 100, 0, 100])
ax.set_aspect('equal')
plt.legend((scatter_loc, scatter_uav),
            ('Potential locations', 'Stationary sensors'),
            scatterpoints=1,
            loc='upper left',
            ncol=1)
plt.savefig('figure_example_spatial_field.pdf', bbox_inches='tight')
