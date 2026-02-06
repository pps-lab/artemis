import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
import os

from datetime import datetime

def get_prover_time (path):
    res = 0
    if not os.path.isfile(path): 
        print(path)
        return res
    
    with open(path) as file:
        lines = [line.rstrip() for line in file]

    for line in lines:
        if (line.startswith("Proving time")):
            res = round(float(line.split(" ")[-1][:-2]), 1)

    return res

def get_verifier_time (path):
    res = 0
    if not os.path.isfile(path): 
        return res
    
    with open(path) as file:
        lines = [line.rstrip() for line in file]

    for line in lines:
        if (line.startswith("Verifying time")):
            if (line.split(" ")[-1][-2:] == 'ms'):
                res = round(float(line.split(" ")[-1][:-2]), 1) / 1000
            else:
                res = round(float(line.split(" ")[-1][:-2]), 1)
    return res

prover_times_nocom = []
prover_times_poly = []
prover_times_pos = []
prover_times_lite = []
prover_times_polyh = []

verifier_times_nocom = []
verifier_times_poly = []
verifier_times_pos = []
verifier_times_lite = []
verifier_times_polyh = []

names = ['mnist', 'snet', 'cifar10', 'dlrm', 'mobilenet', 'vgg', 'diffusion', 'gpt2']
commitments = ['nocom', 'poly', 'pos']
modelsizes = ['~10k', '~100k', '~280k', '~750k', '~3.5m', '~15m', '~19.5m', '~81m']

# for name in names:
#     for commitment in commitments:
#         filepath = os.path.join('../../logs', name + '_' + commitment + '_kzg.txt')
#         prover_times_nocom.append(get_prover_time(filepath))
#         verifier_times_nocom.append(get_verifier_time(filepath))
# nocom results
for name in names:
    filepath = os.path.join('logs', name + '_nocom_kzg.txt')
    prover_times_nocom.append(get_prover_time(filepath))
    verifier_times_nocom.append(get_verifier_time(filepath))

# poly results
for name in names:
    filepath = os.path.join('logs', name + '_poly_kzg.txt')
    prover_times_poly.append(get_prover_time(filepath))
    verifier_times_poly.append(get_verifier_time(filepath))

# pos results
for name in names:
    filepath = os.path.join('logs', name + '_pos_kzg.txt')
    prover_times_pos.append(get_prover_time(filepath))
    verifier_times_pos.append(get_verifier_time(filepath))

# lite results
for name in names:
    filepath = os.path.join('logs', name + '_lite_kzg.txt')
    prover_times_lite.append(get_prover_time(filepath))
    verifier_times_lite.append(get_verifier_time(filepath))

# polyh results
for name in names:
    filepath = os.path.join('logs', name + '_polyh_kzg.txt')
    prover_times_polyh.append(get_prover_time(filepath))
    verifier_times_polyh.append(get_verifier_time(filepath))

# cplink_overhead = [2.2, 4.1, 11.4, 23.0, 164, 408, 838, 2501]
# cplink_opt = [1.8, 1.6, 4.5, 8.8, 102, 152, 317, 921]
#poseidon_time = [54.9, 40.7, 292.3, 0]

cplink = prover_times_nocom
cplink_lite = prover_times_lite
names_sizes = []

# for (time, overhead) in zip(prover_times_nocom, cplink_overhead):
#     cplink.append(time + overhead)

# for (time, overhead) in zip(prover_times_lite, cplink_opt):
#     cplink_lite.append(time + overhead)
# cplink.append(0)
# cplink.append(0)
# cplink.append(0)
# cplink.append(0)

for (name, modelsize) in zip(names, modelsizes):
    names_sizes.append(name + modelsize)

# for val in prover_times_poly: 
#     print(val)

# for val in prover_times_pos: 
#     print(val)

# for val in verifier_times_poly: 
#     print(val)


# Create the bar graph
fontsize=2
plt.figure(figsize=(40, 30))
bar_width = 0.2
x = np.arange(len(names)) * 1.5
fig, ax = plt.subplots()
pos = ax.bar(x, prover_times_pos, width=bar_width, label='Poseidon', color='red', alpha = 1)
#cpl = ax.bar(x + bar_width, cplink, width=bar_width, label='CPLink', color='green', alpha = 1)
polyh = ax.bar(x + bar_width, prover_times_polyh, width=bar_width, label='Poly circuit (horner)', color='green', alpha = 1)
cplite = ax.bar(x + bar_width * 2, cplink_lite, width=bar_width, label='CPLink Optimized', color='purple', alpha = 1)
poly = ax.bar(x + bar_width * 3, prover_times_poly, width=bar_width, label='Poly circuit', color='orange', alpha = 1)
nocom = ax.bar(x + bar_width * 4, prover_times_nocom, width=bar_width, label='No commitment', color='skyblue', alpha = 1)

#bars = ax.bar(names, prover_times_vec)
#label = ax.bar_label(pos, fontsize=fontsize)


# fig2, ax2 = plt.subplots()
#pos = ax.bar(x, verifier_times_pos, width=bar_width, label='Poseidon', color='red', alpha = 1)
#cpl = ax2.bar(x + bar_width, cplink, width=bar_width, label='CPLink', color='green', alpha = 1)
#poly = ax.bar(x, verifier_times_poly, width=bar_width, label='Poly circuit', color='orange', alpha = 1)
#nocom = ax.bar(x + bar_width * 2, verifier_times_nocom, width=bar_width, label='No commitment', color='skyblue', alpha = 1)
#polyh = ax.bar(x + bar_width, verifier_times_polyh, width=bar_width, label='Poly circuit (horner)', color='green', alpha = 1)

#ax.bar_label(cpl, fontsize=fontsize)
label = ax.bar_label(pos, fontsize=fontsize)
ax.bar_label(poly, fontsize=fontsize)
ax.bar_label(polyh, fontsize=fontsize)
ax.bar_label(nocom, fontsize=fontsize)
ax.bar_label(cplite, fontsize=fontsize)

# fontsize = 6
# #bars = ax2.bar(names, prover_times_vec)
#label = ax2.bar_label(pos, fontsize=fontsize)

# ax2.bar_label(cpl, fontsize=fontsize)
# ax2.bar_label(poly, fontsize=fontsize)
# ax2.bar_label(nocom, fontsize=fontsize)

time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
#plt.xticks(x + 0.1, names)


# Add titles and labels
plt.title('Prover times KZG')
plt.xlabel('Models')
plt.xticks(x + bar_width, names_sizes, fontsize=5)
plt.yscale("log", base=2)
plt.ylabel('Prover times (s)')

# Add a legend to the graph
plt.legend()

# # Save the plot as a TikZ file
# tikzplotlib.save("bar_graph.tex")
plt.savefig(time + '.svg', format='svg')
# Show the plot (optional)
plt.show()


import tikzplotlib

tikzplotlib.save(time + '.tex')