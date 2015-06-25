import numpy as np
import matplotlib.pyplot as plt
import os

# The setup is divided into three folder: Caffe, Code and Data
# This file should be in the Code folder
data_root = '../Data/'

for dir in next(os.walk(data_root))[1]: # could be used for all data
#for dir in ['n01530575']:    
    data = np.zeros( (2150,4096) )
    i = 0

    if dir == 'fooling_images':
        continue
    subdir = os.path.join(data_root, dir)
    print "Entering folder: " + subdir
    # Load all layer 6 activations into array
    for directory in next(os.walk(subdir))[1]:
        data[i] = np.load(os.path.join(subdir, directory) + '/layer6_activation.npy')
        i += 1

    # Select final data
    data = data[0:i]
    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    # Save data
    np.save(subdir + '/mean_layer6', mean)
    np.save(subdir + '/stdev_layer6', stdev)
    np.save('../Data/fooling_images/means/layer6/' + dir, mean)
    np.save('../Data/fooling_images/stdevs/layer6/' + dir, stdev)

    # Plot
    plt.figure(figsize=(80,10))
    plt.plot(mean, label="Mean activation")
    plt.plot(mean - stdev, label="Mean - Stdev")
    plt.plot(mean + stdev, label="Mean + Stdev")
    plt.legend()
    plt.xlabel('Neuron')
    plt.xlim(xmax=4100)
    plt.xticks(np.arange(0, 4101, 100))
    plt.ylabel('Activation')
    plt.title('Mean and stdev for layer 6 activation')
    plt.savefig(subdir + "/mean_stdev_layer6.jpg", bbox_inches='tight')
    plt.clf()
