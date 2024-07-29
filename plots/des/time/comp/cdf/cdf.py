import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

fontsize=40

plt.rcParams["font.family"] = 'Arial'
linewidth=5


filename = 'resnet50-batch128-v100.csv'
columns = ['cpu2gpu_time', 'forward_time', 'backward_time']

df = pd.read_csv(filename, delimiter='\t', columns=)
forward = [float(x) for x in df['forward_time'].to_list()]
backward = [float(x) for x in df['backward_time'].to_list()]
# # drop the first 5 values
# dl = dl[5:-1]
forward.insert(0,0)
backward.insert(0,0)


figure, axes = plt.subplots()
x_range = [0,0.3]
y_range = [0,1]
axes.set_xlim(x_range)
axes.set_ylim(y_range)
plt.xticks([0,0.1,0.2], ['0',0.1,0.2],fontsize=tick_font_size)
plt.yticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1], fontsize=tick_font_size)

forward_sorted = np.sort(forward)
forward_cdf = np.arange(0, len(forward_sorted)) / float(len(forward_sorted)-1)
plt.plot(forward_sorted, forward_cdf, linewidth=linewidth, label=legends[i], 'green')

backward_sorted = np.sort(backward)
backward_cdf = np.arange(0, len(backward_sorted)) / float(len(backward_sorted)-1)
plt.plot(backward_sorted, backward_cdf, linewidth=linewidth, label=legends[i], 'blue')


plt.xlabel('Per-batch compute time', fontsize=title_font_size)
plt.ylabel('CDF', fontsize=title_font_size)
plt.title('CDF of per-batch compute time', fontsize=title_font_size, y=1.05)

# plt.legend(fontsize=label_size, markerfirst=False, borderpad=0.2, bbox_to_anchor=(0.55, 0.7))
figure.set_size_inches(12, 8)
figure.set_dpi(100)
plt.savefig('cdf.eps', bbox_inches='tight')

plt.show()