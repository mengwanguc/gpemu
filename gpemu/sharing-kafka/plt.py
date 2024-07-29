import matplotlib.pyplot as plt
import pandas as pd

fontsize=40

# plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize

fig = plt.gcf()

columns = ['app_id', 'batch', 'time', 'throughput']

t0 = 1703993746.7113929

colors = ['orange', 'blue', 'green']
linestyles = ['--', '-.', ':']

for i in range(3):
    filename = 'app{}.csv'.format(i+1)
    df = pd.read_csv(filename, sep='\t', names=columns)
    times = df['time'].tolist()
    times = [float(t)-t0 for t in times]
    throughput = df['throughput'].tolist()
    plt.plot(times, throughput, marker='.', markersize=2, linestyle=linestyles[i], color=colors[i], lw=3, label='Job {}'.format(i+1))


plt.ylim(0,300)
plt.xlim(0,500)
# Adding labels and title
# xticks = [256,512,1024]
# yticks = [20,40,60,80]
# plt.xticks(xticks, fontsize=fontsize)
# plt.yticks(yticks, fontsize=fontsize)
plt.xlabel('Time (s)', fontsize=fontsize)
plt.ylabel('Images per sec', fontsize=fontsize)
plt.title('Throughput for 3 training jobs',fontsize=fontsize, pad=25)

fig.set_size_inches(8, 6)
fig.set_dpi(100)
plt.legend(fontsize=fontsize, loc='upper left', bbox_to_anchor=(1, 1), markerfirst=False)
plt.savefig('fig.png',  bbox_inches='tight')