import csv
import pandas as pd
import random
import pickle
import os
from ast import literal_eval
from scipy import mean
from scipy.stats import sem, t
from tabulate import tabulate

from optimizer.qlearning_kmeans import Q_learningv2
from simulator.mobilecharger.mobilecharger import MobileCharger
from simulator.network.network import Network
from simulator.node.node import Node

def get_experiment(simulation_type):
    while True:
        try:
            experiment_type = input('Enter Experiment type: ')
            experiment_index = int(input('Enter Experiment index: '))
            if simulation_type == 'start':
                df = pd.read_csv("data/" + experiment_type + ".csv")
                return df, experiment_type, experiment_index
            else:
                checkpoint_file = 'checkpoint/checkpoint_{}_{}.pkl'.format(experiment_type, experiment_index)
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                return checkpoint, experiment_type, experiment_index
        except Exception as e:
            if simulation_type=='start':
                print('Experiment does not exist! Please try again.')
            else:
                print('Experiment checkpoint does not exist! Please try a again.')



def start_simulating():
    print('[Simulator] Starting new experiment...')
    df, experiment_type, experiment_index = get_experiment('start')


    try:
        os.makedirs('log')
    except FileExistsError:
        pass
    try:
        os.makedirs('fig')
    except FileExistsError:
        pass
    output_file = open("log/q_learning_Kmeans.csv", "w")
    result = csv.DictWriter(output_file, fieldnames=["nb_run", "lifetime", "dead_node"])
    result.writeheader()


    # Read data from experiment datasheet
    com_ran = df.commRange[experiment_index]
    prob = df.freq[experiment_index]
    nb_mc = df.nb_mc[experiment_index]
    alpha = df.q_alpha[experiment_index]
    clusters = df.charge_pos[experiment_index]
    package_size = df.package[experiment_index]
    q_alpha = df.qt_alpha[experiment_index]
    q_gamma = df.qt_gamma[experiment_index]
    energy = df.energy[experiment_index]
    energy_max = df.energy[experiment_index]
    node_pos = list(literal_eval(df.node_pos[experiment_index]))
    life_time = []
    for nb_run in range(3):
        random.seed(nb_run)

        # Initialize Sensor Nodes
        list_node = []
        for i in range(len(node_pos)):
            location = node_pos[i]
            node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                        energy_thresh=0.4 * energy, prob=prob)
            list_node.append(node)
        
        # Initialize Mobile Chargers
        mc_list = []
        for id in range(nb_mc):
            mc = MobileCharger(id, energy=df.E_mc[experiment_index], capacity=df.E_max[experiment_index],
                            e_move=df.e_move[experiment_index],
                            e_self_charge=df.e_mc[experiment_index], velocity=df.velocity[experiment_index], depot_state = clusters)
            mc_list.append(mc)
        
        # Initialize Targets
        target = [int(item) for item in df.target[experiment_index].split(',')]
        
        # Construct Network
        net_log_file = "log/network_log_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
        MC_log_file = "log/MC_log_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
        experiment = "{}_{}_{}".format(experiment_type, experiment_index, nb_run)
        net = Network(list_node=list_node, mc_list=mc_list, target=target, package_size=package_size, experiment=experiment)
        
        # Initialize Q-learning Optimizer
        q_learning = Q_learningv2(nb_action=clusters, alpha=alpha, q_alpha=q_alpha, q_gamma=q_gamma)
        
        print("[Simulator] Initializing experiment({}, {}), repetition {}:\n".format(experiment_type, experiment_index, nb_run))
        print("[Simulator] Network:")
        print(tabulate([['Sensors', len(net.node)], ['Targets', len(net.target)], ['Package Size', package_size], ['Sending Freq', prob], ['MC', nb_mc]], headers=['Parameters', 'Value']), '\n')
        print("[Simulator] Optimizer:")
        print(tabulate([['Alpha', q_learning.q_alpha], ['Gamma', q_learning.q_gamma], ['Theta', q_learning.alpha]], headers=['Parameters', 'Value']), '\n')
        
        # Define log file
        file_name = "log/q_learning_Kmeans_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
        with open(file_name, "w") as information_log:
            writer = csv.DictWriter(information_log, fieldnames=["time", "nb_dead_node", "nb_package"])
            writer.writeheader()
        
        temp = net.simulate(optimizer=q_learning, t=0, dead_time=0)
        life_time.append(temp[0])
        result.writerow({"nb_run": nb_run, "lifetime": temp[0], "dead_node": temp[1]})

    confidence = 0.95
    h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
    result.writerow({"nb_run": mean(life_time), "lifetime": h, "dead_node": 0})

def resume_simulating():
    print('[Simulator] Resuming Experiment...')
    checkpoint, experiment_type, experiment_index = get_experiment('resume')

    print('[Simulator] Resuming experiment ({}, {}) repetition {}, at {}s.'.format(experiment_type, experiment_index, checkpoint['nb_run'], checkpoint['time']))
    
    net         = checkpoint['network']
    optimizer   = checkpoint['optimizer']
    time        = checkpoint['time']
    dead_time   = checkpoint['dead_time']
    nb_run      = checkpoint['nb_run']
    log_file    = "log/q_learning_Kmeans_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
    lifetime    = net.simulate(optimizer=optimizer, t=time, dead_time=dead_time)

# Read experiment data into Dataframe

print('----------------------------------------------------------------------------------------------------------------------------------------------------------')
print(' █████   ███   █████ ███████████    █████████  ██████   █████     █████████   ███                             ████             █████                      ')
print('░░███   ░███  ░░███ ░░███░░░░░███  ███░░░░░███░░██████ ░░███     ███░░░░░███ ░░░                             ░░███            ░░███                       ')
print(' ░███   ░███   ░███  ░███    ░███ ░███    ░░░  ░███░███ ░███    ░███    ░░░  ████  █████████████   █████ ████ ░███   ██████   ███████    ██████  ████████ ')
print(' ░███   ░███   ░███  ░██████████  ░░█████████  ░███░░███░███    ░░█████████ ░░███ ░░███░░███░░███ ░░███ ░███  ░███  ░░░░░███ ░░░███░    ███░░███░░███░░███')
print(' ░░███  █████  ███   ░███░░░░░███  ░░░░░░░░███ ░███ ░░██████     ░░░░░░░░███ ░███  ░███ ░███ ░███  ░███ ░███  ░███   ███████   ░███    ░███ ░███ ░███ ░░░ ')
print('  ░░░█████░█████░    ░███    ░███  ███    ░███ ░███  ░░█████     ███    ░███ ░███  ░███ ░███ ░███  ░███ ░███  ░███  ███░░███   ░███ ███░███ ░███ ░███     ')
print('    ░░███ ░░███      █████   █████░░█████████  █████  ░░█████   ░░█████████  █████ █████░███ █████ ░░████████ █████░░████████  ░░█████ ░░██████  █████    ')
print('     ░░░   ░░░      ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░    ░░░░░     ░░░░░░░░░  ░░░░░ ░░░░░ ░░░ ░░░░░   ░░░░░░░░ ░░░░░  ░░░░░░░░    ░░░░░   ░░░░░░  ░░░░░     ')                                                                                                                                                                         
print('`-----------------------------------------------------------Qlearning Kmeans Optimization----------------------------------------------------------------/')
print('Select one way to run Simulator:')
print('\t1. Start')
print('\t2. Resume (Requires checkpoint.pkl and log file)')
simulation_type = int(input('Confirm your selection (1/2): '))
print('------------------------------------------------------------------------------')
if (simulation_type == 1):
    start_simulating()
else:
    resume_simulating()

