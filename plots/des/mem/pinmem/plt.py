import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fontsize=40

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize


n_groups = 4
columns = ['epoch_time']

df = pd.read_csv('time.dat', sep='\t', header=None, names=columns)

epoch_time = df['epoch_time'].to_list()

colors = ['red', 'orange', 'green']
labels = ['Real\nGPU', 'Emu\nw/o PM', 'Emu\nw/ PM']

# create plot
fig, ax1 = plt.subplots(1, 1)

ax1.set_ylim(0,500)

bar_width = 0.5

  
for i in range(len(labels)):
    plt.bar(labels[i], epoch_time[i], color=colors[i], width=bar_width)


ax1.set_ylabel('Epoch time (s)', fontsize=fontsize)
# ax1.set_xlabel('Setup', fontsize=fontsize)

# plt.xticks(xtics, xlabels)

# legend reverse order
# ax1.legend(markerfirst=False, bbox_to_anchor=(0.95, 1.35))
# ax2.legend(loc=(0.01, 0.97), ncol=4, frameon=False, markerfirst=False)
# ax1.legend()

fig.set_size_inches(8, 6)
fig.set_dpi(100)

plt.savefig('figure.eps',  bbox_inches='tight')

plt.show()