import csv
import pandas as pd
import random
import pickle
from os.path import exists
from ast import literal_eval
from scipy import mean
from scipy.stats import sem, t

from optimizer.qlearning_kmeans import Q_learningv2
from simulator.mobilecharger.mobilecharger import MobileCharger
from simulator.network.network import Network
from simulator.node.node import Node



def start_simulating():
    print('Starting new experiment...')
    experiment_type = input('Enter Experiment type(node/target/MC/prob/cluster): ')
    df = pd.read_csv("data/" + experiment_type + ".csv")
    experiment_index = int(input('Enter Experiment index(0..4): '))

    # | Experiment_index      Experiment_type|    0    |    1    |    2    |    3     |    4   |
    # |--------------------------------------|---------|---------|---------|----------|--------|
    # | **node**                             |   700   |   800   | __900__ |   1000   |   1100 |
    # | **target**                           |   500   |   550   | __600__ |   650    |   700  |
    # | **MC**                               |   2     | __3__   |   4     |   5      |   6    |
    # | **prob**                             |   0.5   | __0.6__ |   0.7   |   0.8    |   0.9  |
    # | **package**                          | __500__ |   550   |   600   |   650    |   700  |
    # | **cluster**                          |   40    |   50    |   60    |   70     | __80__ |

    # Define output file
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
                            e_self_charge=df.e_mc[experiment_index], velocity=df.velocity[experiment_index])
            mc_list.append(mc)
        
        # Initialize Targets
        target = [int(item) for item in df.target[experiment_index].split(',')]
        
        # Construct Network
        net = Network(list_node=list_node, mc_list=mc_list, target=target, package_size=package_size)
        
        # Initialize Q-learning Optimizer
        q_learning = Q_learningv2(nb_action=clusters, alpha=alpha, q_alpha=q_alpha, q_gamma=q_gamma)
        
        print("Experiment {}, index {}, repeat {}:\n".format(experiment_type, experiment_index, nb_run))
        print("Network:\n\t{} Sensors, {} Targets, Package Frequency: {}, Package size: {}Bytes, Number of MCs: {}".format(len(net.node), len(net.target), prob, package_size, nb_mc))
        print("Optimizer Q_learning Kmeans:\n\tQ-alpha: {}, Q-gamma: {}, Theta: {}, Number of clusters: {}".format(q_learning.q_alpha, q_learning.q_gamma, q_learning.alpha, clusters))
        
        # Define log file
        file_name = "log/q_learning_Kmeans_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
        with open(file_name, "w") as information_log:
            writer = csv.DictWriter(information_log, fieldnames=["time", "nb_dead_node", "nb_package"])
            writer.writeheader()
        
        temp = net.simulate(exp_type=experiment_type, exp_index=experiment_index, nb_run=nb_run, optimizer=q_learning, t=0, dead_time=0, file_name=file_name)
        life_time.append(temp[0])
        result.writerow({"nb_run": nb_run, "lifetime": temp[0], "dead_node": temp[1]})

    confidence = 0.95
    h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
    result.writerow({"nb_run": mean(life_time), "lifetime": h, "dead_node": 0})

def resume_simulating():
    experiment_type  = None
    experiment_index = None
    print('Resuming Experiment...')
    while True:
        experiment_type = input('Enter Experiment type(node/target/MC/prob/cluster): ')
        experiment_index = int(input('Enter Experiment index(0..4): '))
        if exists('checkpoint/checkpoint_{}_{}.pkl'.format(experiment_type, experiment_index)):
            print('Checkpoint found!!!')
            break
        else:
            print('No checkpoint found!!!')
    print('------------------------------------------------------------------------------')

    checkpoint = None
    checkpoint_file = 'checkpoint/checkpoint_{}_{}.pkl'.format(experiment_type, experiment_index)
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)

    print('Resuming Experiment {} index {} repeat {} at {}s...'.format(experiment_type, experiment_index, checkpoint['nb_run'], checkpoint['time']))
    
    net         = checkpoint['network']
    optimizer   = checkpoint['optimizer']
    time        = checkpoint['time']
    dead_time   = checkpoint['dead_time']
    nb_run      = checkpoint['nb_run']
    log_file    = "log/q_learning_Kmeans_{}_{}_{}.csv".format(experiment_type, experiment_index, nb_run)
    lifetime    = net.simulate(exp_type=experiment_type, exp_index=experiment_index, nb_run=nb_run, optimizer=optimizer, t=time, dead_time=dead_time, file_name=log_file)

# Read experiment data into Dataframe
print('.------------------------------------------------------------------------------.')
print('| _       ______  _____ _   __   _____ _                 __      __            |')
print('|| |     / / __ \/ ___// | / /  / ___/(_)___ ___  __  __/ /___ _/ /_____  _____|')
print('|| | /| / / /_/ /\__ \/  |/ /   \__ \/ / __ `__ \/ / / / / __ `/ __/ __ \/ ___/|')
print('|| |/ |/ / _, _/___/ / /|  /   ___/ / / / / / / / /_/ / / /_/ / /_/ /_/ / /    |')
print('||__/|__/_/ |_|/____/_/ |_/   /____/_/_/ /_/ /_/\__,_/_/\__,_/\__/\____/_/     |')
print('|                                                                              |')
print('|-------------------------------------v1.2.1-----------------------------------|')
print('`---------------------------Qlearning Kmeans Optimization----------------------/')
print('Select one way to run Simulator:')
print('\t1. Start')
print('\t2. Resume (Requires checkpoint.pkl and log file)')
simulation_type = int(input('Confirm your selection (1/2): '))
print('------------------------------------------------------------------------------')
if (simulation_type == 1):
    start_simulating()
else:
    resume_simulating()

