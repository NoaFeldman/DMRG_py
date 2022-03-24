import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

dir_name = 'results/time_results'

def plot_perforemance(server, job_opt, node_opt):
    res = {}
    for file in os.listdir(dir_name + '/' + server):
        print(dir_name + '/' + server + '/' + file)
        filename_splitted = file.split('_')
        if filename_splitted[3] == node_opt:
            if node_opt == 'fill':
                N = filename_splitted[8]
                curr_job_opt = filename_splitted[6]
            else:
                N = filename_splitted[7]
                curr_job_opt = filename_splitted[5]
            if  curr_job_opt == job_opt and N == '40':
                with open(dir_name + '/' + server + '/' + file, 'rb') as f:
                    curr_t = pickle.load(f)
                curr_cpu_nums = filename_splitted[2]
                if curr_cpu_nums in res.keys():
                    res[curr_cpu_nums] = res[curr_cpu_nums] + curr_t
                else:
                    res[curr_cpu_nums] = curr_t
    cpu_nums = [int(c) for c in res.keys()]
    times = res.values()
    times_sorted = [x for _, x in sorted(zip(cpu_nums, times))]
    cpu_nums_sorted = sorted(cpu_nums)
    if node_opt == 'fill':
        if server == 'astro':
            full_node_cpus = 96
        elif server == 'amd' or server == 'amd_flag':
            full_node_cpus = 48
        elif server == 'power':
            full_node_cpus = 28
        times_sorted = [times_sorted[i] / (full_node_cpus / cpu_nums_sorted[i]) for i in range(len(cpu_nums_sorted))]
    return cpu_nums_sorted, times_sorted


servers = ['amd', 'amd_flag', 'power', 'astro']
node_opts = ['single', 'fill']
job_opt = 'PEPS4'
legends = []
for server in servers:
    for node_opt in node_opts:
        cpu_nums, times = plot_perforemance(server, job_opt, node_opt)
        plt.plot(cpu_nums, times)
        legends.append(server + '_' + node_opt)
plt.legend(legends)
plt.xlabel('cpu number')
plt.ylabel('time[s]')
plt.title(job_opt)
plt.show()
