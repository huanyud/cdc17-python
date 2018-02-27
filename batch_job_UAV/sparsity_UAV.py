import networkx as nx
import random
import math
import heapq
import queue
import matplotlib.pyplot as plt
import csv
import time
import numpy as np
from matplotlib.patches import Circle
import matplotlib

# To use Type 1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

seed = 1
rand_pos_agent = []
rand_pos_location = []

def generate_graph(num_of_source=100, num_of_sink=1000, supply_each=50, plane_size=100):
    G = nx.MultiDiGraph()
    
    # Add source nodes
    for i in range(num_of_source):
        G.add_node('s' + str(i), demand = -supply_each, total_supply = supply_each, unused_supply = supply_each)
    grand_total_supply = supply_each * num_of_source
    
    # Add sink nodes
    random.seed(seed)
    rand_prior_prob = [0.0] * num_of_sink
    rand_detect_prob = [0.0] * num_of_sink
    for i in range(num_of_sink):
        rand_prior_prob[i] = random.uniform(0.01, 0.99)
    sum_rand_prior_prob = sum(rand_prior_prob)
    rand_prior_prob = [a / sum_rand_prior_prob for a in rand_prior_prob]
    for i in range(num_of_sink):
        rand_detect_prob[i] = random.uniform(0.01, 0.99)
    for i in range(num_of_sink):
        G.add_node('t' + str(i), prior_prob = rand_prior_prob[i], detect_prob = rand_detect_prob[i])
        
    # Add global sink
    G.add_node('gt', demand = grand_total_supply)
    
    # Randomly generate agent and location coordinates
    global rand_pos_agent
    global rand_pos_location
    random.seed(seed)
    for i in range(10):
        for j in range(10):
            rand_pos_agent.append(random.choice([[x, y] for x in range(10*i, 10*(i+1)) for y in range(10*j, 10*(j+1))]))
    rand_pos_location = random.sample([[x, y] for x in range(plane_size) for y in range(plane_size)], num_of_sink)
    
    # Add edges between sinks and the global sink
    for i in range(num_of_sink):
        sink = 't'+str(i)
        G.add_edge(sink, 'gt', key = 1, flow = 0, weight = -G.node[sink]['prior_prob'] * G.node[sink]['detect_prob'], capacity = 1) 
    
    return [G, grand_total_supply]



def update_graph(G, search_radius):
    edge_quota = 0
    # Add edges between sources and sinks
    for idx_a, a in enumerate(rand_pos_agent):
        for idx_l, l in enumerate(rand_pos_location):
            if (a[0]-l[0])**2 + (a[1]-l[1])**2 <= search_radius**2:
                G.add_edge('s' + str(idx_a), 't' + str(idx_l), key = 0, flow = 0, weight = 0)
                edge_quota += 1
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(np.array(rand_pos_location)[:,0], np.array(rand_pos_location)[:,1])
    plt.scatter(np.array(rand_pos_agent)[:,0], np.array(rand_pos_agent)[:,1], marker='x')
    for i in range(num_of_source):
        circle = Circle((rand_pos_agent[i][0], rand_pos_agent[i][1]), radius=search_radius, fill=False)
        ax.add_patch(circle)
    plt.axis([0, 100, 0, 100])
    ax.set_aspect('equal')
    plt.show()
    '''
    
    return [G, edge_quota]




def assign_extra_demand(nodeName):
    # Initialization
    visited_set = set() # Record visited nodes
    Q = queue.Queue()
    Q.put(nodeName)
    in_Q_set = set() # Record nodes currently in Q (since Python has no queue.contains())
    in_Q_set.add(nodeName)
    pred = {} # Record predecessors for building augmenting path
    for i in range(num_of_source):
        pred['s'+str(i)] = None
    for i in range(num_of_sink):
        pred['t'+str(i)] = None
    # Loop
    while Q.empty() is False:
        nd = Q.get()
        in_Q_set.remove(nd)
        visited_set.add(nd)
        # If the node is a sink
        if nd[0] == 't':
            for source in G.predecessors(nd):
                if source not in in_Q_set and source not in visited_set and source not in elim_set:
                    Q.put(source)
                    in_Q_set.add(source)
                    pred[source] = nd
        # If the node is a source, but it has no unused supply
        elif nd[0] == 's' and G.node[nd]['unused_supply'] == 0:
            for sink in G.neighbors(nd):
                if G.edge[nd][sink][0]['flow'] > 0 and sink not in in_Q_set and sink not in visited_set and sink not in elim_set:
                    Q.put(sink)
                    in_Q_set.add(sink)
                    pred[sink] = nd
        # If the node is a source, and it has unused supply
        else: #(nd[0] == 's' and G.node[nd]['unused_supply'] > 0):
            # Decrement unused_supply of the source by 1
            G.node[nd]['unused_supply'] -= 1 
            # Recursively build augmenting path
            cur = nd
            while True:
                pre = pred[cur]
                if pre == None:
                    break
                # For source
                if cur[0] == 's':
                    old_flow = G.edge[cur][pre][0]['flow']
                    G.add_edge(cur, pre, key = 0, flow = old_flow + 1)
                # For sink
                else:
                    old_flow = G.edge[pre][cur][0]['flow']
                    G.add_edge(pre, cur, key = 0, flow = old_flow - 1)
                cur = pre
            # There is no node to be eliminated! And remember to break the outer loop!
            visited_set = set()
            break
    return visited_set



num_of_source = 100
num_of_sink = 1000
supply_each = 50
search_radius_list = [15, 17.5, 20, 22.5, 25, 27.5, 30]
edge_quota_list = [0] * len(search_radius_list)
time_list_our_algo = [0.0] * len(search_radius_list)
time_list_capacity_scaling = [0.0] * len(search_radius_list)
[G_origin, _] = generate_graph(num_of_source, num_of_sink, supply_each)
for kk in range(0, len(search_radius_list)):
    G = G_origin.copy()
    # Update the graph
    # Update edges between sources and sinks
    [G, edge_quota] = update_graph(G, search_radius_list[kk])
    edge_quota_list[kk] = edge_quota
    grand_total_supply = num_of_source * supply_each

    # Run our algorithm
    start_time_our_algo = time.time()

    heap_of_weights = []
    for i in range(num_of_sink):
        heapq.heappush(heap_of_weights, (G.edge['t'+str(i)]['gt'][1]['weight'], 't'+str(i), 1))
    elim_set = set()
    final_sink_demand = {}
    for i in range(num_of_sink):
        final_sink_demand['t'+str(i)] = 0

    while grand_total_supply > 0 and len(elim_set) != num_of_source + num_of_sink:
        top_element_heap = heapq.heappop(heap_of_weights)
        sink = top_element_heap[1]
        if sink in elim_set:
            continue
        potential_elim_set = assign_extra_demand(sink)
        if len(potential_elim_set) == 0:
            grand_total_supply -= 1
            final_sink_demand[sink] += 1
            old_order = top_element_heap[2]
            old_weight = top_element_heap[0]
            G.edge[sink]['gt'][old_order]['flow'] = 1
            new_weight = old_weight * G.node[sink]['detect_prob']
            new_order = old_order + 1
            G.add_edge(sink, 'gt', key = new_order, flow = 0, weight = new_weight, capacity = 1)
            heapq.heappush(heap_of_weights, (new_weight, sink, new_order))
        else:
            elim_set = elim_set.union(potential_elim_set)

    time_elapsed_our_algo = time.time() - start_time_our_algo
    flow_cost_our_algo = 0

    edge_list_2 = [edge for edge in G.edges(data=True) if edge[0][0] == 't' and edge[2]['flow'] > 0]
    for edge in edge_list_2:
        flow_cost_our_algo += edge[2]['weight']
    
    # Run NetworkX's capacity scaling algorithm
    start_time_capacity_scaling = time.time()
    [flow_cost_capacity_scaling, _] = nx.capacity_scaling(G)
    time_elapsed_capacity_scaling = time.time() - start_time_capacity_scaling
    
    # Store results
    time_list_our_algo[kk] = time_elapsed_our_algo
    time_list_capacity_scaling[kk] = time_elapsed_capacity_scaling
        
    

# Save data to a txt file
outfile = open('result_sparsity_UAV.txt', 'a')
outfile.write('----#source:' + str(num_of_source) + ', #sink: ' + str(num_of_sink) + ' (seed: ' + str(seed) + ')----\n')
outfile.write('num of arcs between sources and sinks: ' + str(edge_quota_list) + '\n')
outfile.write('time_our_algo: ' + str(time_list_our_algo) + ', time_capacity_scaling: ' + str(time_list_capacity_scaling) + '\n')
outfile.close()

# Plot figure
plt.figure()
plt.plot(edge_quota_list, time_list_our_algo, 'rs-')
plt.plot(edge_quota_list, time_list_capacity_scaling, 'b^--')
plt.rc('text', usetex=True)
plt.xlabel(r'$|\mathcal{A}|$')
plt.ylabel('Time elapsed (sec)')
plt.legend(['New algorithm', 'Capacity scaling'], loc=2)
#plt.xlim(9500, 50500)
plt.xticks(edge_quota_list)
plt.savefig('figure_sparsity_UAV.pdf', bbox_inches='tight')
