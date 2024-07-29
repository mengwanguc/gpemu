import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

title_font_size = 30
tick_font_size = 30
textfontsize = 39

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = textfontsize


columns = ['cache', 'io', 'cpu2gpu', 'gpu']
df = pd.read_csv('ds-cache-emu.csv', sep='\t', header=None, names=columns)

io = df['io'].tolist()
cpu2gpu = df['cpu2gpu'].tolist()
gpu = df['gpu'].tolist()

# must be modified. 
# TODO: How did Mike compute this ideal? need to verify
ideal = [158.846, 109.486, 10.627, 1.145, 0.000]
# create plot
fig, ax1 = plt.subplots(1, 1)

ax1.set_ylim(0,400)
ytics = range(0,450,100)
ax1.set_xlim(-0.5,5)
# ax1.set_yticks([2800])
ax1.tick_params(axis='y', labelsize=20)

bar_width = 0.3
opacity = 1

xtics = df.index
xlabels = df['cache'].to_list()
xlabels = ['{:.0f}%'.format(c*100) for c in xlabels]

bottom = [0 for i in gpu]
gpubars = ax1.bar(xtics, gpu, bar_width,
color='blue',
label='GPU compute'
)

bottom = [bottom[i]+gpu[i] for i in range(len(bottom))]
iobars = ax1.bar(xtics, io, bar_width, bottom=bottom,
color='red',
label='Fetch Stall'
)

bottom = [bottom[i] for i in range(len(bottom))]
idealbars = ax1.bar(xtics, ideal, 0.2, bottom=bottom,
color='pink',
label='Ideal fetch stall'
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

ax1.set_ylabel('Epoch Time (s)', fontsize=textfontsize)
ax1.set_xlabel('% of dataset cached', fontsize=textfontsize)


plt.xticks(xtics, xlabels, fontsize=textfontsize)
plt.yticks(ytics, fontsize=textfontsize)
# plt.title('Fetch Stall Time (Emulator)\nOn HDD + 24 workers + 8GPUs (emulated by p100)')
plt.title('(b) Fetch Stall v.s. Cache %', pad=20, fontsize=textfontsize)

# legend reverse order
handles, labels = plt.gca().get_legend_handles_labels()
ax1.legend(reversed(handles), reversed(labels), markerfirst=False, bbox_to_anchor=(0.3, 1), fontsize=textfontsize)
# ax2.legend(loc=(0.01, 0.97), ncol=4, frameon=False, markerfirst=False)
# ax1.legend()

fig.set_size_inches(11, 8)
fig.set_dpi(100)

plt.savefig('ds-cache.eps',  bbox_inches='tight')
plt.show()
