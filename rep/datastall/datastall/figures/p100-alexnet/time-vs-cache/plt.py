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
columns = ['cache', 'io', 'cpu2gpu', 'gpu']
schemes = {}
local = {}
df = pd.read_csv('time.dat', sep='\t', header=None, names=columns)

io = df['io'].tolist()
cpu2gpu = df['cpu2gpu'].tolist()
gpu = df['gpu'].tolist()

# create plot
fig, ax1 = plt.subplots(1, 1)

ax1.set_ylim(0,500)
ax1.set_xlim(-0.5,1.5)
# ax1.set_yticks([2800])
ax1.tick_params(axis='y', labelsize=20)


bar_width = 0.3
opacity = 1

  
xtics = df.index
xlabels = df['cache'].to_list()
xlabels = ['{:.0f}%'.format(l*100) for l in xlabels]

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
iobars = ax1.bar(xtics, io, bar_width, bottom=bottom,
color='red',
label='Data stall time'
)

# local_bot = ax1.bar(index+i*bar_width, local[i], bar_width, bottom=schemes[i],
# alpha=opacity,
# fill=False,
# edgecolor=colors[i],
# hatch='\\',
# label='RS{}-L'.format(i+1)
# )

ax1.set_ylabel('Per epoch training time (s)', fontsize=25)
ax1.set_xlabel('Cache percentage', fontsize=25)

plt.xticks(xtics, xlabels)

# legend reverse order
handles, labels = plt.gca().get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels), markerfirst=False, bbox_to_anchor=(0.95, 1.35))
# ax2.legend(loc=(0.01, 0.97), ncol=4, frameon=False, markerfirst=False)
# ax1.legend()

fig.set_size_inches(8, 8)
fig.set_dpi(100)

plt.savefig('figure.png',  bbox_inches='tight')