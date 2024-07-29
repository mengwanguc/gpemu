import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

title_font_size = 25
tick_font_size = 25
textfontsize = 25
textfontcolor = 'purple'
anotation_color = textfontcolor

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 25
label_font = {'fontname':'Helvetica', 'fontsize':'25'}
annotation_font = {'fontname':'Helvetica', 'fontsize':title_font_size}


n_groups = 4
columns = ['cache', 'thread', 'gpu_count', 'io', 'cpu2gpu', 'gpu']
schemes = {}
local = {}
df = pd.read_csv('time.dat', sep='\t', header=None, names=columns)


io = df['io'].tolist()
cpu2gpu = df['cpu2gpu'].tolist()
gpu = df['gpu'].tolist()
thread = df['thread'].to_list()
gpu_count = df['gpu_count'].to_list()

thread_per_gpu = [thread[i]//gpu_count[i] for i in range(len(thread))]
total_time = [io[i]+cpu2gpu[i]+gpu[i] for i in range(len(io))]

# create plot
fig, ax1 = plt.subplots(1, 1)

ax1.set_ylim(0,1000)
ax1.set_xlim(0,20)
ax1.set_xticks([1,2,4,8,16])
ax1.tick_params(axis='y', labelsize=20)


ax1.set_ylabel('Per epoch training time (s)', fontsize=25)
ax1.set_xlabel('CPU # per GPU', fontsize=25)

plt.plot(thread_per_gpu, total_time, color='blue', marker='o', linewidth=2, markersize=8)

plt.title('Per epoch train time v.s. cpu # per gpu', pad=25)

# legend reverse order
# handles, labels = plt.gca().get_legend_handles_labels()
# ax1.legend(reversed(handles), reversed(labels), markerfirst=False, bbox_to_anchor=(0.95, 1.35))
# ax2.legend(loc=(0.01, 0.97), ncol=4, frameon=False, markerfirst=False)
# ax1.legend()

fig.set_size_inches(8, 8)
fig.set_dpi(100)

plt.savefig('figure.png',  bbox_inches='tight')