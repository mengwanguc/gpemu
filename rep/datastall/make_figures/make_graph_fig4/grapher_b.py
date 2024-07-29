import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

title_font_size = 18
tick_font_size = 18
textfontsize = 18
textfontcolor = 'purple'
anotation_color = textfontcolor

prep_time = 5.688

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 18
label_font = {'fontname':'Helvetica', 'fontsize':'18'}
annotation_font = {'fontname':'Helvetica', 'fontsize':title_font_size}


n_groups = 4
columns = ['cache', 'io', 'cpu2gpu', 'gpu']
schemes = {}
local = {}
df = pd.read_csv('time4(b).dat', sep='\t', header=None, names=columns)

sizes = [14,16,18,20,22,24]
io = df['io'].tolist()
cpu2gpu = df['cpu2gpu'].tolist()
gpu = df['gpu'].tolist()

# must be modified. 
ideal = [158.846, 109.486, 10.627, 1.145, 0.421, 0.000]
# create plot
fig, ax1 = plt.subplots(1, 1)

ax1.set_ylim(0,400)
ax1.set_xlim(-0.5,6)
# ax1.set_yticks([2800])
ax1.tick_params(axis='y', labelsize=20)

bar_width = 0.3
opacity = 1

xtics = df.index
xlabels = df['cache'].to_list()
xlabels = ['{:.0f}%\n{}g'.format(l*100, s) for l,s in zip(xlabels,sizes)]

bottom = [0 for i in gpu]
gpubars = ax1.bar(xtics, gpu, bar_width,
color='blue',
label='GPU compute time'
)

bottom = [bottom[i]+gpu[i] for i in range(len(bottom))]
cpu2gpubars = ax1.bar(xtics, cpu2gpu, bar_width, bottom=bottom,
color='orange',
label='CPU to GPU transfer time'
)

bottom = [bottom[i]+cpu2gpu[i] for i in range(len(bottom))]
prepbars = ax1.bar(xtics, prep_time, bar_width, bottom=bottom,
color='green',
label='prep stall time'
)

bottom = [bottom[i]+prep_time for i in range(len(bottom))]
iobars = ax1.bar(xtics, io, bar_width, bottom=bottom,
color='red',
label='fetch stall time'
)

bottom = [bottom[i] for i in range(len(bottom))]
idealbars = ax1.bar(xtics, ideal, 0.2, bottom=bottom,
color='pink',
label='ideal fetch stall time'
)
for bar in idealbars:
    bar.set_hatch('/')
# local_bot = ax1.bar(index+i*bar_width, local[i], bar_width, bottom=schemes[i],
# alpha=opacity,
# fill=False,
# edgecolor=colors[i],
# hatch='\\',
# label='RS{}-L'.format(i+1)
# )

ax1.set_ylabel('Per epoch training time (s)', fontsize=18)
ax1.set_xlabel('% of dataset cached', fontsize=18)

plt.xticks(xtics, xlabels)
plt.yticks(fontsize=18)
plt.title('Fetch Stall Time (Emulator)\nOn HDD + 24 workers + 8GPUs (emulated by p100)')

# legend reverse order
handles, labels = plt.gca().get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels), markerfirst=False, bbox_to_anchor=(0.4, 0.6))
# ax2.legend(loc=(0.01, 0.97), ncol=4, frameon=False, markerfirst=False)
# ax1.legend()

fig.set_size_inches(10, 10)
fig.set_dpi(100)

plt.savefig('Figure_4_HDD_merged.png',  bbox_inches='tight')
plt.show()