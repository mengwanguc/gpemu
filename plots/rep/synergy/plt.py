import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

fontsize=40

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize

fig = plt.gcf()
fig.set_size_inches(20, 8)

linewidth=3

plt.subplot(1,2,1)

legends = ['Prop$_{GPEmu}$', 'Syn$_{GPEmu}$', 'Prop$_{Paper}$', 'Syn$_{Paper}$']
filenames = ['gpemu-prop.csv', 'gpemu-synergy.csv', 'paper-prop.csv', 'paper-synergy.csv']
colors = ['red', 'green', 'purple', 'blue']
linestyles = ['-', '-', ':', ':']
for i in range(4):
	df = pd.read_csv('las/'+filenames[i], delimiter=',', header=None, names=['jct', 'cdf'])
	jct = [float(x) for x in df['jct'].to_list()]
	cdf = [float(x) for x in df['cdf'].to_list()]
	print('{}\t{}'.format(legends[i], sum(jct)/len(jct)))
	jct.sort()
	cdf.sort()
	jct = [0] + jct
	cdf = [0] + cdf
	plt.plot(jct, cdf, linewidth=linewidth, linestyle=linestyles[i], label=legends[i], color=colors[i])
	

x_range = [0,12]
y_range = [0,1]
xticks = [0,4,8,12]
yticks = [0,0.2,0.4,0.6,0.8,1]
plt.ylim(y_range)
plt.xlim(x_range)
plt.xticks(xticks, xticks,fontsize=fontsize)
plt.yticks(yticks, yticks, fontsize=fontsize)

	

plt.xlabel('JCT (hours)', fontsize=fontsize)
plt.ylabel('CDF of JCT', fontsize=fontsize)
plt.title('CDF of JCT for LAS policy', fontsize=fontsize, y=1.05)

plt.legend(loc="upper center", bbox_to_anchor=(1, 1.4), ncol=4, markerfirst=False, columnspacing=1)



plt.subplot(1,2,2)

legends = ['Prop$_{GPEmu}$', 'Syn$_{GPEmu}$', 'Prop$_{Paper}$', 'Syn$_{Paper}$']
filenames = ['gpemu-prop.csv', 'gpemu-synergy.csv', 'paper-prop.csv', 'paper-synergy.csv']
colors = ['red', 'green', 'purple', 'blue']
linestyles = ['-', '-', ':', ':']
for i in range(4):
	df = pd.read_csv('srtf/'+filenames[i], delimiter=',', header=None, names=['jct', 'cdf'])
	jct = [float(x) for x in df['jct'].to_list()]
	cdf = [float(x) for x in df['cdf'].to_list()]
	jct.sort()
	cdf.sort()
	jct = [0] + jct
	cdf = [0] + cdf
	plt.plot(jct, cdf, linewidth=linewidth, linestyle=linestyles[i], label=legends[i], color=colors[i])
	

x_range = [0,12]
y_range = [0,1]
xticks = [0,4,8,12]
yticks = [0,0.2,0.4,0.6,0.8,1]
plt.ylim(y_range)
plt.xlim(x_range)
plt.xticks(xticks, xticks,fontsize=fontsize)
plt.yticks(yticks, yticks, fontsize=fontsize)

	

plt.xlabel('JCT (hours)', fontsize=fontsize)
# plt.ylabel('CDF of JCT', fontsize=fontsize)
plt.title('CDF of JCT for SRTF policy', fontsize=fontsize, y=1.05)

# plt.legend(loc="upper center", bbox_to_anchor=(1.15, 1.5), ncol=4, markerfirst=False, columnspacing=1)


fig.set_dpi(100)
plt.savefig('cdf.eps', bbox_inches='tight')
plt.savefig('cdf.png', bbox_inches='tight')
