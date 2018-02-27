import networkx as nx
import random
import math
import heapq
import queue
import matplotlib.pyplot as plt
import csv
import time

random.seed(1)

def generate_graph(num_of_source, num_of_sink, supply_each=10, arc_prob=0.5):
    G = nx.MultiDiGraph()

    IS_RANDOM_SUPPLY = False
    grand_total_supply = 0
    rand_supply = [0] * num_of_source
    for i in range(num_of_source):  
        if IS_RANDOM_SUPPLY:
            rand_supply[i] = random.randint(1, 10)
        else:
            rand_supply[i] = supply_each
    for i in range(num_of_source):
        G.add_node('s' + str(i), demand = -rand_supply[i], total_supply = rand_supply[i], unused_supply = rand_supply[i])
        grand_total_supply += rand_supply[i]

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

    G.add_node('gt', demand = grand_total_supply)

    for i in range(num_of_source):
        for j in range(num_of_sink):
            if random.random() > arc_prob:
                G.add_edge('s' + str(i), 't' + str(j), key = 0, flow = 0, weight = 0)

    for i in range(num_of_source):
        if not G.neighbors('s' + str(i)):
            G.add_edge('s' + str(i), 't' + str(random.randint(0, num_of_sink - 1)), key = 0, flow = 0, weight = 0)

    for i in range(num_of_sink):
        if not G.predecessors('t' + str(i)):
            source = 's' + str(random.randint(0, num_of_source - 1))
            G.add_edge(source, 't' + str(i), key = 0, flow = 0, weight = 0) 

    for i in range(num_of_sink):
        sink = 't'+str(i)
        G.add_edge(sink, 'gt', key = 1, flow = 0, weight = -G.node[sink]['prior_prob'] * G.node[sink]['detect_prob'], capacity = 1)
    
    return [G, grand_total_supply]

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



num_of_source = 200
num_of_sink = 300
supply_each_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
time_list_our_algo = [0.0] * len(supply_each_list)
time_list_capacity_scaling = [0.0] * len(supply_each_list)
[G, _] = generate_graph(num_of_source, num_of_sink)
for kk in range(0, len(supply_each_list)):
    # Update the graph
    # Update node supply
    supply_each = supply_each_list[kk]
    for i in range(num_of_source):
        G.node['s'+str(i)]['demand'] = -supply_each
        G.node['s'+str(i)]['total_supply'] = supply_each
        G.node['s'+str(i)]['unused_supply'] = supply_each
    grand_total_supply = supply_each * num_of_source
    G.node['gt']['demand'] = grand_total_supply

    G_copy = G.copy()
    
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
    
    G = G_copy

# Save data to a txt file
outfile = open('result_supply.txt', 'a')
outfile.write('----#source:' + str(num_of_source) + ', #sink: ' + str(num_of_sink) + '----\n')
outfile.write('supply_at_each_source: ' + str(supply_each_list) + '\n')
outfile.write('time_our_algo: ' + str(time_list_our_algo) + ', time_capacity_scaling: ' + str(time_list_capacity_scaling) + '\n')
outfile.close()

# Plot figure
plt.plot(supply_each_list, time_list_our_algo, 'rs-')
plt.plot(supply_each_list, time_list_capacity_scaling, 'b^--')
plt.xlabel('Supply at each source')
plt.ylabel('Time elapsed (sec)')
plt.legend(['New algorithm', 'Capacity scaling'], loc=2)
plt.xlim(5, 95)
plt.xticks(supply_each_list)
plt.savefig('figure_supply.pdf')
