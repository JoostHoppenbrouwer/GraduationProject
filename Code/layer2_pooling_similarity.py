import numpy as np
import scipy.spatial.distance as dst
import scipy.signal as sgl
import matplotlib.pyplot as plt
import os
import pickle

plt.rcParams['figure.figsize'] = (10, 10)

# Layer properties
NUM_KERNELS = 256
KERNEL_SIZE = 13

fooling_root = '../Data/fooling_images/'
data_root = '../Data/'

# Loop over fooling images layer 2 after pooling feat
for dir in next(os.walk(fooling_root))[1]:
#for dir in ['n01530575']:
    if (dir in ['crops', 'means', 'stdevs', 'n01530575']):
        continue

    subdir = os.path.join(fooling_root, dir)
    print "Entering folder: " + subdir

    layer2_fooling = np.load(subdir + '/layer2_pooling.npy')

    # Similarity plot
    plt.figure(figsize=(20,10))
    plt.xlim(0,2)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.ylabel('Similarity')
    plt.title('Layer 2 pooling similarity')

    # Create distance folder if not already existing
    folder = subdir + "/distance/"
    if not os.path.exists(os.path.dirname(folder)):
        os.makedirs(os.path.dirname(folder))

    # To store similarity
    final_sims = {}

    # Compare with all classes
    for classdir in next(os.walk(data_root))[1]:
        if classdir in ['fooling_images']:
            continue
        classfolder = os.path.join(data_root, classdir)
        print classdir

        i = 0
        MAX_COMPARE = 2150
        sims = np.zeros((MAX_COMPARE, NUM_KERNELS))
        # Compare with MAX_COMPARE images
        for directory in next(os.walk(classfolder))[1]:
            if i == MAX_COMPARE:
                break

            # Load and setup
            layer2_output = np.load(os.path.join(classfolder, directory) + '/layer2_pooling.npy')
            sim = np.zeros(NUM_KERNELS)

            # Compare patch by patch and store similarity
            for j in range(NUM_KERNELS):
                x = (j / 16) * 14
                y = (j % 16) * 14
                fooling_data = layer2_fooling[x : x + KERNEL_SIZE, y : y + KERNEL_SIZE]
                class_data = layer2_output[x: x + KERNEL_SIZE, y : y + KERNEL_SIZE]
                sim[j] = np.amax(sgl.convolve2d(fooling_data, class_data, mode='same', boundary='wrap'))

            sims[i] = sim

            i += 1

        sims = sims[0:i]
        class_sim = np.mean(sims, axis=0)

        # Plot
        if (classdir == dir):
            plt.plot(1, np.sum(class_sim) / NUM_KERNELS, 'go')
        else:
            plt.plot(1, np.sum(class_sim) / NUM_KERNELS, 'rx')

        # Store similarity
        final_sims[classdir] = np.sum(class_sim) / NUM_KERNELS

    plt.savefig(folder + "/layer2_pooling_similarity.jpg", bbox_inches='tight')
    pickle.dump( final_sims, open( folder + "layer2_pooling_sims.p", "wb" ) )
