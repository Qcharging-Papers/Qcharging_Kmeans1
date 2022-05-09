import numpy as np
from scipy.spatial import distance

from optimizer.utils import init_function, q_max_function, reward_function, network_clustering, network_clustering_v2
from simulator.node.utils import find_receiver


class Q_learningv2:
    def __init__(self, init_func=init_function, nb_action=81, alpha=0, q_alpha=0.5, q_gamma=0.5, load_checkpoint=False):
        self.action_list = []
        self.nb_action = nb_action
        self.q_table = init_func(nb_action=nb_action)
        self.state = nb_action
        self.charging_time = [0.0 for _ in range(nb_action + 1)]
        self.reward = np.asarray([0.0 for _ in range(nb_action + 1)])
        self.reward_max = [0.0 for _ in range(nb_action + 1)]
        self.list_request = []
        self.alpha = alpha
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma

    def update(self, mc, network, time_stem, alpha=0.5, gamma=0.5, q_max_func=q_max_function, reward_func=reward_function):
        if not len(self.list_request):
            return self.action_list[self.state], 0.0
        self.set_reward(mc=mc,time_stem=time_stem, reward_func=reward_func, network=network)
        self.q_table[self.state] = (1 - self.q_alpha) * self.q_table[self.state] + self.q_alpha * (
                self.reward + self.q_gamma * self.q_max(q_max_func))
        self.choose_next_state(mc, network)
        if self.state == len(self.action_list) - 1:
            charging_time = (mc.capacity - mc.energy) / mc.e_self_charge
        else:
            charging_time = self.charging_time[self.state]
        print("mc ", mc.id, "next state =", self.action_list[self.state], self.state, charging_time)
        # print(self.charging_time)
        return self.action_list[self.state], charging_time

    def q_max(self, q_max_func=q_max_function):
        return q_max_func(q_table=self.q_table, state=self.state)

    def set_reward(self, mc = None, time_stem=0, reward_func=reward_function, network=None):
        first = np.asarray([0.0 for _ in self.action_list], dtype=float)
        second = np.asarray([0.0 for _ in self.action_list], dtype=float)
        third = np.asarray([0.0 for _ in self.action_list], dtype=float)
        for index, row in enumerate(self.q_table):
            temp = reward_func(network=network, mc=mc, q_learning=self, state=index, time_stem=time_stem, receive_func=find_receiver)
            first[index] = temp[0]
            second[index] = temp[1]
            third[index] = temp[2]
            self.charging_time[index] = temp[3]
        first = first / np.sum(first)
        second = second / np.sum(second)
        third = third / np.sum(third)
        self.reward = first + second + third
        self.reward_max = list(zip(first, second, third))

    def choose_next_state(self, mc, network):
        # next_state = np.argmax(self.q_table[self.state])
        if mc.energy < 10:
            self.state = len(self.q_table) - 1
        else:
            self.state = np.argmax(self.q_table[self.state])
            # print(self.reward_max[self.state])
            # print(self.action_list[self.state])
    
    def net_partition(self, net=None, netpart_func=network_clustering):
        self.action_list = network_clustering_v2(self, network=net, nb_cluster=self.nb_action)
