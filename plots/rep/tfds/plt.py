import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=40

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = fontsize


fig = plt.gcf()

columns = ['worker', 'emulator_time', 'gpu_time']

# ssd, 24 workers, 8 v100
df = pd.read_csv('speedup.dat', sep='\t', names=columns)


def get_speedups(times, baseline):
	res = []
	for t in times:
		res.append(baseline/t)
	return res

emulator_baseline = 38.84659195
emulator_times = df['emulator_time'].tolist()
emulator_speedups = get_speedups(emulator_times, emulator_baseline)

gpu_baseline = 40.62999511
gpu_times = df['gpu_time'].tolist()
gpu_speedups = get_speedups(gpu_times, gpu_baseline)

workers = df['worker'].tolist()


plt.plot(workers, emulator_speedups, marker='^', markersize=10, linestyle='-', color='blue', label='Speedup')

# plt.plot(workers, gpu_speedups, marker='o', markersize=10, linestyle='-', color='green', label='Speedup-GPEmu')



plt.axhline(y=1, color='purple', linestyle='--', label='Baseline')



plt.ylim(0,5)
# Adding labels and title
plt.xticks(range(0,14,2), fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('# of workers', fontsize=fontsize)
plt.ylabel('Speedup', fontsize=fontsize)
plt.title('Training Time Speedup across TF-DS worker count',fontsize=fontsize, pad=20)

fig.set_size_inches(16, 6)
fig.set_dpi(100)
plt.legend(fontsize=fontsize)
plt.savefig('tfds.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()