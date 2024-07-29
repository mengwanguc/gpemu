import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.stats import pearsonr

title_font_size = 52
tick_font_size = 48
label_size = 40

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Times New Roman"
annotation_font = {'fontname':'Times New Roman', 'fontsize':title_font_size}

linewidth=5


all_times = []
all_batches = []
filenames = ['alexnet-p100.txt']
legends = ['Alexnet-P100']
colors = ['blue']

for filename in filenames:
	with open(filename) as f:
		df = pd.read_csv(filename, delimiter='\t')
		times = df['cpu2gpu_time'].to_list()
		batches = df['batch_size'].to_list()
		all_times.append(times)
		all_batches.append(batches)

# print(batches)
# print(times)

figure, axes = plt.subplots()
# axes.set_aspect( 1.1 )
x_range = [0,128]
y_range = [0,0.008]
axes.set_xlim(x_range)
axes.set_ylim(y_range)
plt.xticks([0,32,64,96,128], fontsize=tick_font_size)
plt.yticks([0,0.002,0.004,0.006,0.008], [0,0.002,0.004,0.006,0.008], fontsize=tick_font_size)

for i in range(len(filenames)):
	# plt.plot(all_batches[i], all_times[i], linewidth=linewidth, label=legends[i], color=colors[i])
	plt.plot(all_batches[i], all_times[i], marker='o', ms=4, label=legends[i], color=colors[i])
	correlation_coefficient, _ = pearsonr(all_batches[i][0:128], all_times[i][0:128])
	print(correlation_coefficient)
plt.xlabel('Batch size', fontsize=title_font_size)
plt.ylabel('Host-to-GPU time', fontsize=title_font_size)
plt.title('Per-batch Host-to-GPU Time v.s. Batch Size', fontsize=title_font_size, y=1.05)
# Time v.s. Batch Size

# Calculate Pearson correlation coefficient


# plt.legend(fontsize=label_size, markerfirst=False, borderpad=0.2, loc='upper left')
figure.set_size_inches(12, 8)
figure.set_dpi(100)
plt.savefig('smallbatch.eps', bbox_inches='tight')