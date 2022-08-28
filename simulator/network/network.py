import csv
from scipy.spatial import distance

from simulator.network import parameter as para
from simulator.network.utils import uniform_com_func, to_string, count_package_function, set_checkpoint


class Network:
    def __init__(self, list_node=None, mc_list=None, target=None, package_size=400, experiment=None):
        self.node = list_node
        self.set_neighbor()
        self.set_level()
        self.mc_list = mc_list
        self.target = target
        self.charging_pos = []
        self.package_size = package_size

        self.active = False
        self.package_lost = False

        self.experiment = experiment
        self.net_log_file = "log/net_log_" + self.experiment + ".csv"
        self.mc_log_file = "log/mc_log_" + self.experiment + ".csv"


    def set_neighbor(self):
        for node in self.node:
            for other in self.node:
                if other.id != node.id and distance.euclidean(node.location, other.location) <= node.com_ran:
                    node.neighbor.append(other.id)

    def set_level(self):
        queue = []
        for node in self.node:
            if distance.euclidean(node.location, para.base) < node.com_ran:
                node.level = 1
                queue.append(node.id)
        while queue:
            for neighbor_id in self.node[queue[0]].neighbor:
                if not self.node[neighbor_id].level:
                    self.node[neighbor_id].level = self.node[queue[0]].level + 1
                    queue.append(neighbor_id)
            queue.pop(0)



    def communicate(self, func=uniform_com_func):
        return func(self)

    def run_per_second(self, t, optimizer):
        state = self.communicate()
        request_id = []
        for index, node in enumerate(self.node):
            if node.energy < node.energy_thresh:
                node.request(optimizer=optimizer, t=t)
                request_id.append(index)
            else:
                node.is_request = False
        if request_id:
            for index, node in enumerate(self.node):
                if index not in request_id and (t - node.check_point[-1]["time"]) > 50:
                    node.set_check_point(t)
            
        if optimizer and self.active:
            for mc in self.mc_list:
                # if mc.id*300 < t:
                #     mc.run(network=self, time_stem=t, net=self, optimizer=optimizer)
                mc.run(network=self, time_stem=t, net=self, optimizer=optimizer)
        return state

    def simulate_max_time(self, optimizer=None, t=0, dead_time=0, max_time=2000000):
        nb_dead = self.count_dead_node()
        nb_package = self.count_package()
        dead_time = dead_time

        if t == 0:
            with open(self.net_log_file, "w") as information_log:
                writer = csv.DictWriter(information_log, fieldnames=['time_stamp', 'number_of_dead_nodes', 'number_of_monitored_target', 'lowest_node_energy', 'lowest_node_location', 'avg_energy', 'MC_0_status', 'MC_1_status', 'MC_2_status', 'MC_0_location', 'MC_1_location', 'MC_2_location'])
                writer.writeheader()
            
            with open(self.mc_log_file, "w") as mc_log:
                writer = csv.DictWriter(mc_log, fieldnames=['time_stamp', 'id', 'starting_point', 'destination_point', 'decision_id', 'charging_time', 'moving_time'])
                writer.writeheader()
        
        t = t
        while t <= max_time and nb_package==len(self.target):
            t = t + 1
            if (t - 1) % 100 == 0:
                print("[Network] Simulating time: {}s, lowest energy node: {:.4f} at {}".format(t, self.node[self.find_min_node()].energy, self.node[self.find_min_node()].location))
                print('\t\tNumber of dead nodes: {}'.format(self.count_dead_node()))
                print('\t\tNumber of packages: {}'.format(self.count_package()))
                
                network_info = {
                    'time_stamp' : t,
                    'number_of_dead_nodes' : self.count_dead_node(),
                    'number_of_monitored_target' : self.count_package(),
                    'lowest_node_energy': round(self.node[self.find_min_node()].energy, 3),
                    'lowest_node_location': self.node[self.find_min_node()].location,     
                    'avg_energy': self.get_average_energy(),
                    'MC_0_status' : self.mc_list[0].get_status(),
                    'MC_1_status' : self.mc_list[1].get_status(),
                    'MC_2_status' : self.mc_list[2].get_status(),
                    'MC_0_location' : self.mc_list[0].current,
                    'MC_1_location' : self.mc_list[1].current,
                    'MC_2_location' : self.mc_list[2].current,
                }
                with open(self.net_log_file, 'a') as information_log:
                    node_writer = csv.DictWriter(information_log, fieldnames=['time_stamp', 'number_of_dead_nodes', 'number_of_monitored_target', 'lowest_node_energy', 'lowest_node_location', 'avg_energy', 'MC_0_status', 'MC_1_status', 'MC_2_status', 'MC_0_location', 'MC_1_location', 'MC_2_location'])
                    node_writer.writerow(network_info)
                for mc in self.mc_list:
                    print("\t\tMC #{} is {} at {}".format(mc.id, mc.get_status(), mc.current))

            if (t-1) % 500 == 0 and t > 1:
                set_checkpoint(t=t, network=self, optimizer=optimizer, dead_time=dead_time)

            ######################################
            if t == 200:
                optimizer.net_partition(net=self)
                self.active = True
            ######################################

            state = self.run_per_second(t, optimizer)
            current_dead = self.count_dead_node()
            current_package = self.count_package()
            if not self.package_lost:
                if current_package < len(self.target):
                    self.package_lost = True
                    dead_time = t
            if current_dead != nb_dead or current_package != nb_package:
                network_info = {
                    'time_stamp' : t,
                    'number_of_dead_nodes' : self.count_dead_node(),
                    'number_of_monitored_target' : self.count_package(),
                    'lowest_node_energy': round(self.node[self.find_min_node()].energy, 3),
                    'lowest_node_location': self.node[self.find_min_node()].location,
                    'avg_energy': self.get_average_energy(),
                    'MC_0_status' : self.mc_list[0].get_status(),
                    'MC_1_status' : self.mc_list[1].get_status(),
                    'MC_2_status' : self.mc_list[2].get_status(),
                    'MC_0_location' : self.mc_list[0].current,
                    'MC_1_location' : self.mc_list[1].current,
                    'MC_2_location' : self.mc_list[2].current,
                }
                with open(self.net_log_file, 'a') as information_log:
                    node_writer = csv.DictWriter(information_log, fieldnames=['time_stamp', 'number_of_dead_nodes', 'number_of_monitored_target', 'lowest_node_energy', 'lowest_node_location', 'MC_0_status', 'MC_1_status', 'MC_2_status', 'MC_0_location', 'MC_1_location', 'MC_2_location'])
                    node_writer.writerow(network_info)
                break


        print('\n[Network]: Finished with {} dead sensors, {} packages at {}s!'.format(self.count_dead_node(), self.count_package(), dead_time))
        return dead_time, nb_dead

    def simulate(self, optimizer=None, t=0, dead_time=0, max_time=2000000):
        life_time = self.simulate_max_time(optimizer=optimizer, t=t, dead_time=dead_time, max_time=max_time)
        return life_time

    def print_net(self, func=to_string):
        func(self)

    def find_min_node(self):
        min_energy = 10 ** 10
        min_id = -1
        for node in self.node:
            if node.energy < min_energy:
                min_energy = node.energy
                min_id = node.id
        return min_id

    def count_dead_node(self):
        count = 0
        for node in self.node:
            if node.energy <= 0:
                count += 1
        return count

    def count_package(self, count_func=count_package_function):
        count = count_func(self)
        return count

    def get_average_energy(self):
        total = 0
        for node in self.node:
            total += node.avg_energy
        return total/len(self.node)

    ##############################################################################################
    def simulate_lifetime(self, optimizer, file_name="log/energy_log.csv"):
        energy_log = open(file_name, "w")
        node_log = open('log/dead_node.csv', 'w')
        writer = csv.DictWriter(energy_log, fieldnames=["time", "mc energy", "min energy"])
        writer.writeheader()
        node_writer = csv.DictWriter(node_log, fieldnames=['time', 'dead_node'])
        node_writer.writeheader()
        node_log.close()
        t = 0
        while t <= 2000000:
            t = t + 1
            if (t - 1) % 100 == 0:
                node_log = open('log/dead_node.csv', 'a')
                node_writer = csv.DictWriter(node_log, fieldnames=['time', 'dead_node'])
                node_writer.writerow({"time": t, "dead_node": self.count_dead_node()})
                node_log.close()
                print('number of dead node: {}'.format(self.count_dead_node()))
                print("time = ", t, ", lowest energy node: ", self.node[self.find_min_node()].energy, "at",
                      self.node[self.find_min_node()].location)
                for mc in self.mc_list:
                    print("\tMC#{} at{} is {}".format(mc.id, mc.current, mc.get_status()))
            state = self.run_per_second(t, optimizer)
            if not (t - 1) % 50:
                for mc in self.mc_list:
                    writer.writerow(
                        {"time": t, "mc energy": mc.energy, "min energy": self.node[self.find_min_node()].energy})

        print(t, self.node[self.find_min_node()].energy)
        for mc in self.mc_list:
            print("\tMC#{} at{}".format(mc.id, mc.current))
            writer.writerow({"time": t, "mc energy": mc.energy, "min energy": self.node[self.find_min_node()].energy})
        energy_log.close()
        return t
