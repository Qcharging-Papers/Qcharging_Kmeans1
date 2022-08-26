# Libraries
from timeit import repeat
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

# Modules
from simulator.network import parameter as para
from simulator.node.utils import find_receiver


def q_max_function(q_table, state):
    temp = [max(row) if index != state else -float("inf") for index, row in enumerate(q_table)]
    return np.asarray(temp)


def reward_function(network, mc, q_learning, state, time_stem, receive_func=find_receiver):
    alpha = q_learning.alpha
    charging_time = get_charging_time(network, mc, q_learning, time_stem=time_stem, state=state, alpha=alpha)
    w, nb_target_alive = get_weight(network, mc, q_learning, state, charging_time, receive_func)
    p = get_charge_per_sec(network, q_learning, state)
    p_hat = p / np.sum(p)
    E = np.asarray([network.node[request["id"]].energy for request in q_learning.list_request])
    e = np.asarray([request["avg_energy"] for request in q_learning.list_request])
    second = nb_target_alive / len(network.target)
    third = np.sum(w * p_hat)
    first = np.sum(e * p / E)
    return first, second, third, charging_time


def init_function(nb_action=81):
    return np.zeros((nb_action + 1, nb_action + 1), dtype=float)

def get_weight(net, mc, q_learning, action_id, charging_time, receive_func=find_receiver):
    p = get_charge_per_sec(net, q_learning, action_id)
    all_path = get_all_path(net, receive_func)
    time_move = distance.euclidean(q_learning.action_list[mc.state],
                                   q_learning.action_list[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in q_learning.list_request]
    for request_id, request in enumerate(q_learning.list_request):
        temp = (net.node[request["id"]].energy - time_move * request["avg_energy"]) + (
                p[request_id] - request["avg_energy"]) * charging_time
        if temp < 0:
            list_dead.append(request["id"])
    for request_id, request in enumerate(q_learning.list_request):
        nb_path = 0
        for path in all_path:
            if request["id"] in path:
                nb_path += 1
        w[request_id] = nb_path
    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    for path in all_path:
        if para.base in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive


def get_path(net, sensor_id, receive_func=find_receiver):
    path = [sensor_id]
    if distance.euclidean(net.node[sensor_id].location, para.base) <= net.node[sensor_id].com_ran:
        path.append(para.base)
    else:
        receive_id = receive_func(net=net, node=net.node[sensor_id])
        if receive_id != -1:
            path.extend(get_path(net, receive_id, receive_func))
    return path


def get_all_path(net, receive_func=find_receiver):
    list_path = []
    for sensor_id, target_id in enumerate(net.target):
        list_path.append(get_path(net, sensor_id, receive_func))
    return list_path


def get_charge_per_sec(net, q_learning, state):
    return np.asarray(
        [para.alpha / (distance.euclidean(net.node[request["id"]].location,
                                          q_learning.action_list[state]) + para.beta) ** 2 for
         request in q_learning.list_request])

def get_charging_time(network=None, mc = None, q_learning=None, time_stem=0, state=None, alpha=0.1):
    # request_id = [request["id"] for request in network.mc.list_request]
    time_move = distance.euclidean(mc.current, q_learning.action_list[state]) / mc.velocity
    energy_min = network.node[0].energy_thresh + alpha * network.node[0].energy_max
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    for node in network.node:
        d = distance.euclidean(q_learning.action_list[state], node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in network.mc_list:
            if other_mc.id != mc.id and other_mc.get_status() == "charging":
                d = distance.euclidean(other_mc.current, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2)*(other_mc.end_time - time_stem)
            elif other_mc.id != mc.id and other_mc.get_status() == "moving" and other_mc.state != len(q_learning.q_table) - 1:
                d = distance.euclidean(other_mc.end, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2)*(other_mc.end_time - other_mc.arrival_time)
        if node.energy - time_move * node.avg_energy + p1 < energy_min and p - node.avg_energy > 0:
            s1.append((node.id, p, p1))
        if node.energy - time_move * node.avg_energy + p1 > energy_min and p - node.avg_energy < 0:
            s2.append((node.id, p, p1))
    t = []

    for index, p, p1 in s1:
        t.append((energy_min - network.node[index].energy + time_move * network.node[index].avg_energy - p1) / (
                p - network.node[index].avg_energy))
    for index, p, p1 in s2:
        t.append((energy_min - network.node[index].energy + time_move * network.node[index].avg_energy - p1) / (
                p - network.node[index].avg_energy))
    dead_list = []
    for item in t:
        nb_dead = 0
        for index, p, p1 in s1:
            temp = network.node[index].energy - time_move * network.node[index].avg_energy + p1 + (
                        p - network.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        for index, p, p1 in s2:
            temp = network.node[index].energy - time_move * network.node[index].avg_energy + p1 + (
                        p - network.node[index].avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        dead_list.append(nb_dead)
    if dead_list:
        arg_min = np.argmin(dead_list)
        return t[arg_min]
    return 0

def network_clustering(optimizer, network=None, nb_cluster=81):
    X = []
    Y = []
    for node in network.node:
        node.set_check_point(200)
        X.append(node.location)
        Y.append(node.avg_energy**0.5)
    X = np.array(X)
    Y = np.array(Y)
    # print(Y)
    d = np.linalg.norm(Y)
    Y = Y/d
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X, sample_weight=Y)
    charging_pos = []
    for pos in kmeans.cluster_centers_:
        charging_pos.append((int(pos[0]), int(pos[1])))
    charging_pos.append(para.depot)
    # print(charging_pos, file=open('log/centroid.txt', 'w'))
    node_distribution_plot(network=network, charging_pos=charging_pos)
    network_plot(network=network, charging_pos=charging_pos)
    return charging_pos

def network_clustering_v2(optimizer, network=None, nb_cluster=81):
    X = []
    Y = []
    min_node = 1000
    for node in network.node:
        node.set_check_point(200)
        if node.avg_energy != 0:
            min_node = min(min_node, node.avg_energy)
    for node in network.node:
        repeat = int(node.avg_energy/min_node)
        for _ in range(repeat):
            X.append(node.location)
            Y.append(node.avg_energy)
    X = np.array(X)
    Y = np.array(Y)
    # print(Y)
    d = np.linalg.norm(Y)
    Y = Y/d
    # print(d)
    # print(Y)
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(X)
    charging_pos = []
    for pos in kmeans.cluster_centers_:
        charging_pos.append((int(pos[0]), int(pos[1])))
    charging_pos.append(para.depot)
    # print(charging_pos, file=open('log/centroid.txt', 'w'))
    # node_distribution_plot(network=network, charging_pos=charging_pos)
    network_plot(network=network, charging_pos=charging_pos)
    return charging_pos
    
def node_distribution_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.node:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    plt.hist(c_node, bins=100)
    plt.savefig('fig/node_distribution.png')

def network_plot(network, charging_pos):
    x_node = []
    y_node = []
    c_node = []
    for node in network.node:
        x_node.append(node.location[0])
        y_node.append(node.location[1])
        c_node.append(node.avg_energy)
    x_centroid = []
    y_centroid = []
    for centroid in charging_pos:
        x_centroid.append(centroid[0])
        y_centroid.append(centroid[1])
    c_node = np.array(c_node)
    d = np.linalg.norm(c_node)
    c_node = c_node / d * 80
    plt.scatter(x_node, y_node, s = c_node)
    plt.scatter(x_centroid, y_centroid, c='red', marker='^')
    plt.savefig('fig/network_plot.png')
