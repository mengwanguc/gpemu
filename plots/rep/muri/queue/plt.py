import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=40

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize

fig = plt.gcf()

columns = ['time', 'queue']

colors = ['red', 'green']
linestyles = ['-', '-']
filenames = ['srtf', 'muri']
labels =  ['SRTF', 'Muri-S']

x = []
y = []

for i in range(2):
    filename = '{}.csv'.format(filenames[i])
    df = pd.read_csv(filename, sep='\t', names=columns)
    times = df['time'].tolist()
    queues = df['queue'].tolist()
    plt.plot(times, queues, linestyle=linestyles[i], color=colors[i], label='{}'.format(labels[i]))



plt.ylim(0,10)
plt.xlim(0,100)
# Adding labels and title
xticks = range(0,101,50)
yticks = range(0,11,5)
plt.xticks(xticks, fontsize=fontsize)
plt.yticks(yticks, fontsize=fontsize)
plt.xlabel('Time (min)', fontsize=fontsize)
plt.ylabel('Queue length', fontsize=fontsize)
plt.title('(a) Queue length by time',fontsize=fontsize, pad=25)

fig.set_size_inches(8, 6)
fig.set_dpi(100)
plt.legend(fontsize=fontsize, markerfirst=False)
plt.savefig('fig.png',  bbox_inches='tight')
plt.savefig('fig.eps',  bbox_inches='tight')