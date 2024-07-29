import json
import matplotlib.pyplot as plt
import numpy as np

train_top1, train_top5, val_top1, val_top5 = [],[],[],[]
train_top1_size2 = []
train_top1_size4 = []
train_top1_size8 = []
train_top1_size16 = []
train_top1_size32 = []
train_top1_size64 = []
epochs = 0
batch_size = 0
model = ""
batch_top1_top5 = []
batch_top1_top5_size2 = []
batch_top1_top5_size4 = []
batch_top1_top5_size8 = []
batch_top1_top5_size16 = []
batch_top1_top5_size32 = []
batch_top1_top5_size64 = []


short_batch = '128'
current_p = 'train_top5'
current_title = 'Average Training 5 Runs Grouping Method top 5 Accuracy'
process = '_trainingtop5_'

for seed_number in range(5):
    with open('diffclass-groupingresnet18_batch_128_gsize_64_epo_100_sampler-seed-'+str(seed_number)) as json_file:
        data = json.load(json_file)
        for k,v in data.items():
            if k == current_p:
                batch_top1_top5.append(data[k])
            elif k == 'model':
                model = data[k]           
            elif k == 'args.epochs':
                epochs = data[k]	
            elif k == 'args.batch_size':
                batch_size = data[k]
    with open("rand-perm_resnet18_batch_128_gsize_64_epo_100_sampler-seed-"+str(seed_number)) as json_file:
        data = json.load(json_file)
        for k,v in data.items():
            if k == current_p:
                batch_top1_top5_size2.append(data[k])
    with open('sequential-groupingresnet18_batch_128_gsize_64_epo_100_sampler-seed-'+str(seed_number)) as json_file:
        data = json.load(json_file)
        for k,v in data.items():
            if k == current_p:
                batch_top1_top5_size4.append(data[k])
    with open('resnet18_batch_128_epo_100_sampler-seed-'+str(seed_number)) as json_file:
        data = json.load(json_file)
        for k,v in data.items():
            if k == current_p:
                batch_top1_top5_size8.append(data[k])

def helper(batches):
    batch_range = []
    batch_top1, batch_top5 = [], []
    for k,v in batches.items():
        batch_range.append(int(k))
        batch_top1.append(v[0])
        batch_top5.append(v[1])
    return (batch_range, batch_top1, batch_top5)


# x1, top1s1, top5s1 = helper(batch_top1_top5)
# x2, top1s2, top5s2 = helper(batch_top1_top5_size2)
# x4, top1s4, top5s4 = helper(batch_top1_top5_size4)
# x8, top1s8, top5s8 = helper(batch_top1_top5_size8)
# x16, top1s16, top5s16 = helper(batch_top1_top5_size16)
# x32, top1s32, top5s32 = helper(batch_top1_top5_size32)
# x64, top1s64, top5s64 = helper(batch_top1_top5_size64)


epoch_range = range(1,epochs+1)

plt.plot(epoch_range, np.array([sum(x) for x in zip(*batch_top1_top5)])/5, '--',  label = 'diffclass_grouping_gsize64')
plt.plot(epoch_range, np.array([sum(x) for x in zip(*batch_top1_top5_size2)])/5, '-',label = 'rand-perm_grouping_gsize64')
plt.plot(epoch_range, np.array([sum(x) for x in zip(*batch_top1_top5_size4)])/5,  ':',label = 'sequential_grouping_gsize64')
plt.plot(epoch_range, np.array([sum(x) for x in zip(*batch_top1_top5_size8)])/5, '-.', label = 'no_grouping')
# plt.plot(epoch_range, np.array([sum(x) for x in zip(*batch_top1_top5_size16)])/5, label = 'groupsize_16')
# plt.plot(epoch_range, np.array([sum(x) for x in zip(*batch_top1_top5_size32)])/5, label = 'groupsize_32')
# plt.plot(epoch_range, np.array([sum(x) for x in zip(*batch_top1_top5_size64)])/5, label = 'groupsize_64')
plt.title(current_title)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,100])
plt.legend()
plt.show(block=True) 
ep = str(epochs)
bat = str(batch_size)
results_dir = ''
sample_file_name = model+"_average_"+current_p + "_batch_" + bat +"_epo_"+ ep +".png" 
plt.savefig(results_dir + sample_file_name)

"""
plt.plot(x1, top1s1, label = 'groupsize_1')
plt.plot(x1, top1s2, label = 'groupsize_2')
plt.plot(x1, top1s4, label = 'groupsize_4')
plt.plot(x1, top1s8, label = 'groupsize_8')
plt.plot(x1, top1s16, label = 'groupsize_16')
plt.plot(x1, top1s32, label = 'groupsize_32')
plt.plot(x1, top1s64, label = 'groupsize_64')
plt.title(current_title)
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.ylim([0,100])
plt.legend()
plt.show(block=True)
ep = str(epochs)
bat = str(batch_size)
results_dir = 'grouped_model_graph/'
sample_file_name = "batch_graph_"+ model +process+ "_batch_" + bat +"_epo_"+ ep +".png" 
plt.savefig(results_dir + sample_file_name)
"""