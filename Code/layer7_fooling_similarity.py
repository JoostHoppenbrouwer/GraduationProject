import numpy as np
import scipy.spatial.distance as dst
import matplotlib.pyplot as plt
import os
import pickle

data_root = '../Data/fooling_images/'
means_root = '../Data/fooling_images/means/layer7/'
stdevs_root = '../Data/fooling_images/stdevs/layer7/'

# Loop over fooling images layer 7 feat
for dir in next(os.walk(data_root))[1]:
    if (dir in ['crops', 'means', 'stdevs']):
        continue

    subdir = os.path.join(data_root, dir)
    print "Entering folder: " + subdir

    layer7_output = np.load(subdir + '/layer7_activation.npy')

    plt.figure(figsize=(20,10))
    plt.figure(1)
    plt.title('Layer 7 accuracy for several thresholds (o = fooling class, x = other class)')
    plt.xlabel('Threshold')
    plt.xlim(0,6)
    plt.xticks( np.arange(6), ('', '0.0', '0.1', '0.25', '0.5', '1.0', '') )
    plt.ylabel('Accuracy')
    plt.ylim(0,1)

    plt.figure(figsize=(10,10))
    plt.figure(2)
    plt.title('Layer 7 cosine similarity (o = fooling class, x = other class)')
    plt.xlim(0,2)
    plt.xticks( np.arange(2) )
    plt.ylabel('Cosine similarity')
    plt.ylim(0,1)

    plt.figure(figsize=(10,10))
    plt.figure(3)
    plt.xlim(0,2)
    plt.xticks( np.arange(2) )
    plt.title('Layer 7 euclidean distance (o = fooling class, x = other class)')
    plt.ylabel('Euclidean distance')

    # Create distance folder if not already existing
    folder = subdir + "/distance/"
    if not os.path.exists(os.path.dirname(folder)):
        os.makedirs(os.path.dirname(folder))
    f = open(folder + "layer7_distance.txt", "w")

    # Loop through different thresholds
    threshold_index = -1
    for THRESHOLD in [0.0, 0.1, 0.25, 0.5, 1]:
        f.write("Using Threshold: " + str(THRESHOLD) + '\n')
        threshold_index += 1

        # To store cosine similarity with threshold 0.0
        if THRESHOLD == 0.0:
            cos_sims = {}

        # Loop over all class means and stdevs
        for file in next(os.walk(means_root))[2]:
            name = os.path.splitext(file)[0]
            f.write("Comparing with class: " + name + '\n')
            class_mean = np.load(os.path.join(means_root, file))
            class_stdev = np.load(os.path.join(stdevs_root, file))

            # Used for similarity percentage
            in_range = 0
            counter = 0

            # Check conditions
            index_array = []
            for i in range(0,4096):
                value = layer7_output[i]
                # Used for error distance
                if (value >= THRESHOLD):
                    low = class_mean[i] - class_stdev[i]
                    high = class_mean[i] + class_stdev[i]
                    if (low <= value <= high):
                        in_range += 1
                    counter += 1
                if class_mean[i] >= THRESHOLD:
                    index_array = index_array + [i]
            # Used for other distances
            layer7_output1 = layer7_output[index_array]
            class_mean1 = class_mean[index_array]

            # Accuracy
            plt.figure(1)
            error = float(in_range) / counter
            if (name == dir):
                plt.plot(threshold_index+1, error, 'go')
            else:
                plt.plot(threshold_index+1, error, 'rx')
            f.write("Accuracy: " + str(error) + ', ')

            if THRESHOLD == 0.0:

                # Cosine similarity
                plt.figure(2)
                cos_sim = 1-(dst.cosine(class_mean1, layer7_output1))
                # Store cosine similarity with threshold 0.0
                cos_sims[name] = cos_sim
                if (name == dir):
                    plt.plot(threshold_index+1, cos_sim, 'go')
                else:
                    plt.plot(threshold_index+1, cos_sim, 'rx')
                f.write("Cosine similarity: " + str(cos_sim) + ', ')

                # Euclidean distance
                plt.figure(3)
                euc_dst = dst.euclidean(class_mean1, layer7_output1)
                if (name == dir):
                    plt.plot(threshold_index+1, euc_dst, 'go')
                else:
                    plt.plot(threshold_index+1, euc_dst, 'rx')
                f.write("Euclidean distance: " + str(euc_dst) + '\n')

        if THRESHOLD == 0.0:
            pickle.dump( cos_sims, open( folder + "layer7_cos_sims.p", "wb" ) )

    f.close()
    plt.figure(1)
    plt.savefig(subdir + "/distance/layer7_accuracy.jpg", bbox_inches='tight')
    plt.close()
    plt.figure(2)
    plt.savefig(subdir + "/distance/layer7_cosine_similarity.jpg", bbox_inches='tight')
    plt.close()
    plt.figure(3)
    plt.savefig(subdir + "/distance/layer7_euclidean_distance.jpg", bbox_inches='tight')
    plt.close()
    plt.close("all")
