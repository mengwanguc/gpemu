import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=40

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize


fig = plt.gcf()

sizes = []

columns = ['size', 'throughput']

# ssd, 24 workers, 8 v100
df = pd.read_csv('data.dat', sep='\t', names=columns)

size = df['size'].tolist()
throughput = df['throughput'].tolist()

plt.plot(size, throughput, marker='o', markersize=10, linestyle='-', color='blue', lw=3)


plt.ylim(0,100)
plt.xlim(0,1100)
# Adding labels and title
xticks = [256,512,1024]
yticks = [20,40,60,80]
plt.xticks(xticks, fontsize=fontsize)
plt.yticks(yticks, fontsize=fontsize)
plt.xlabel('File size per read (KB)', fontsize=fontsize)
plt.ylabel('Throughput (MB/s)', fontsize=fontsize)
plt.title('(b) HDD random read throughput',fontsize=fontsize, pad=25)

fig.set_size_inches(8, 6)
fig.set_dpi(100)
# plt.legend(fontsize=fontsize)
plt.savefig('thru.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()