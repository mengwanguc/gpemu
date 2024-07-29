import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fontsize=40

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = fontsize


fig = plt.gcf()

# sizes = []

sorted_data = []
cdf = []

# ssd, 24 workers, 8 v100
with open('filesizes.txt', 'r') as f:
	text = f.readlines()
	# sizes = [int(i)/1024 for i in text.split(',')]
	for line in text:
		values = line.split('\t')
		cdf.append(float(values[0]))
		sorted_data.append(int(values[1])/1000)

print(sorted_data)


# sorted_data = sorted(sizes)
# data_count = len(sorted_data)

# sorted_data = [0] + sorted_data
# cdf = [i/(data_count)*100 for i in range(data_count+1)]



plt.plot(sorted_data, cdf, linestyle='-', color='blue', lw=5)


plt.ylim(0,100)
plt.xlim(0,1000)
# Adding labels and title
xticks = range(0,1001,200)
yticks = range(0,101,20)
plt.xticks(xticks, xticks, fontsize=fontsize)
plt.yticks(yticks, yticks, fontsize=fontsize)
plt.xlabel('File size (KB)', fontsize=fontsize)
plt.ylabel('Size CDF (in %)', fontsize=fontsize)
plt.title('(a) CDF of ImageNet file size',fontsize=fontsize, pad=25)

fig.set_size_inches(8, 6)
fig.set_dpi(100)
# plt.legend(fontsize=fontsize)
plt.savefig('cdf.eps',  bbox_inches='tight')
# Displaying the chart
plt.show()