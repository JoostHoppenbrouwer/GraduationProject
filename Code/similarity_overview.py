import numpy as np
import scipy.spatial.distance as dst
import scipy.signal as sgl
import matplotlib.pyplot as plt
import os
import pickle

plt.rcParams['figure.figsize'] = (10, 10)

fooling_root = '../Data/fooling_images/'
data_root = '../Data/'

# Loop over fooling images
for dir in next(os.walk(fooling_root))[1]:
#for dir in ['n01530575', 'n02317335', 'n02799071']:
    if (dir in ['crops', 'means', 'stdevs']):
        continue

    subdir = os.path.join(fooling_root, dir)
    print "Entering folder: " + subdir

    # Load similarity measures
    layer1 = pickle.load( open( subdir + "/distance/layer1_sims.p", "rb" ) )
    layer1_pool = pickle.load( open( subdir + "/distance/layer1_pooling_sims.p", "rb" ) )
    layer1_pool_norm = pickle.load( open( subdir + "/distance/layer1_pooling_norm_sims.p", "rb" ) )
    layer2 = pickle.load( open( subdir + "/distance/layer2_sims.p", "rb" ) )
    layer2_pool = pickle.load( open( subdir + "/distance/layer2_pooling_sims.p", "rb" ) )
    layer2_pool_norm = pickle.load( open( subdir + "/distance/layer2_pooling_norm_sims.p", "rb" ) )
    layer3 = pickle.load( open( subdir + "/distance/layer3_sims.p", "rb" ) )
    layer4 = pickle.load( open( subdir + "/distance/layer4_sims.p", "rb" ) )
    layer5 = pickle.load( open( subdir + "/distance/layer5_sims.p", "rb" ) )
    layer5_pool = pickle.load( open( subdir + "/distance/layer5_pooling_sims.p", "rb" ) )
    layer6 = pickle.load( open( subdir + "/distance/layer6_cos_sims.p", "rb" ) )
    layer7 = pickle.load( open( subdir + "/distance/layer7_cos_sims.p", "rb" ) )

    # Similarity plot
    plt.figure(figsize=(20,10))
    plt.xlim(0,13)
    plt.xticks( np.arange(13), ('', 'Layer 1', 'Layer 1 Pooling', 'Layer 1 Pooling Norm',
        'Layer 2', 'Layer 2 Pooling', 'Layer 2 Pooling Norm', 'Layer 3', 'Layer 4',
        'Layer 5', 'Layer 5 Pooling', 'Layer 6', 'Layer 7', ''), rotation=20 )
    plt.ylim(0.0,1.1)
    plt.ylabel('Similarity')
    plt.title('Similarity overview')

    for synset in layer1:
        # Plot
        if (synset == dir):
            plt.plot(1, layer1[synset] / layer1[max(layer1, key=layer1.get)], 'go')
        else:
            plt.plot(1, layer1[synset] / layer1[max(layer1, key=layer1.get)], 'rx')

    for synset in layer1_pool:
        # Plot
        if (synset == dir):
            plt.plot(2, layer1_pool[synset] / layer1_pool[max(layer1_pool, key=layer1_pool.get)], 'go')
        else:
            plt.plot(2, layer1_pool[synset] / layer1_pool[max(layer1_pool, key=layer1_pool.get)], 'rx')

    for synset in layer1_pool_norm:
        # Plot
        if (synset == dir):
            plt.plot(3, layer1_pool_norm[synset] / layer1_pool_norm[max(layer1_pool_norm, key=layer1_pool_norm.get)], 'go')
        else:
            plt.plot(3, layer1_pool_norm[synset] / layer1_pool_norm[max(layer1_pool_norm, key=layer1_pool_norm.get)], 'rx')

    for synset in layer2:
        # Plot
        if (synset == dir):
            plt.plot(4, layer2[synset] / layer2[max(layer2, key=layer2.get)], 'go')
        else:
            plt.plot(4, layer2[synset] / layer2[max(layer2, key=layer2.get)], 'rx')

    for synset in layer2_pool:
        # Plot
        if (synset == dir):
            plt.plot(5, layer2_pool[synset] / layer2_pool[max(layer2_pool, key=layer2_pool.get)], 'go')
        else:
            plt.plot(5, layer2_pool[synset] / layer2_pool[max(layer2_pool, key=layer2_pool.get)], 'rx')

    for synset in layer2_pool_norm:
        # Plot
        if (synset == dir):
            plt.plot(6, layer2_pool_norm[synset] / layer2_pool_norm[max(layer2_pool_norm, key=layer2_pool_norm.get)], 'go')
        else:
            plt.plot(6, layer2_pool_norm[synset] / layer2_pool_norm[max(layer2_pool_norm, key=layer2_pool_norm.get)], 'rx')

    for synset in layer3:
        # Plot
        if (synset == dir):
            plt.plot(7, layer3[synset] / layer3[max(layer3, key=layer3.get)], 'go')
        else:
            plt.plot(7, layer3[synset] / layer3[max(layer3, key=layer3.get)], 'rx')

    for synset in layer4:
        # Plot
        if (synset == dir):
            plt.plot(8, layer4[synset] / layer4[max(layer4, key=layer4.get)], 'go')
        else:
            plt.plot(8, layer4[synset] / layer4[max(layer4, key=layer4.get)], 'rx')

    for synset in layer5:
        # Plot
        if (synset == dir):
            plt.plot(9, layer5[synset] / layer5[max(layer5, key=layer5.get)], 'go')
        else:
            plt.plot(9, layer5[synset] / layer5[max(layer5, key=layer5.get)], 'rx')

    for synset in layer5_pool:
        # Plot
        if (synset == dir):
            plt.plot(10, layer5_pool[synset] / layer5_pool[max(layer5_pool, key=layer5_pool.get)], 'go')
        else:
            plt.plot(10, layer5_pool[synset] / layer5_pool[max(layer5_pool, key=layer5_pool.get)], 'rx')

    for synset in layer6:
        # Plot
        if (synset == dir):
            plt.plot(11, layer6[synset] / layer6[max(layer6, key=layer6.get)], 'go')
        else:
            plt.plot(11, layer6[synset] / layer6[max(layer6, key=layer6.get)], 'rx')

    for synset in layer7:
        # Plot
        if (synset == dir):
            plt.plot(12, layer7[synset] / layer7[max(layer7, key=layer7.get)], 'go')
        else:
            plt.plot(12, layer7[synset] / layer7[max(layer7, key=layer7.get)], 'rx')

    plt.savefig(subdir + "/distance/similarity_overview.jpg", bbox_inches='tight')
