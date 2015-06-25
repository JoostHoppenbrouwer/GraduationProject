import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import scipy

# Take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # Force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # Tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    return data

# The setup is divided into three folder: Caffe, Code and Data
# This file should be in the Code folder
caffe_root = '../Caffe/'
data_root = '../Data/'

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Initialize Deep Neural Network
caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# Input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# Load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# Gather data
for dir in next(os.walk(data_root))[1]: # could be used for all data
#for dir in ['n01558993']: # could be used to run on specific classes
    if dir in ['fooling_images']: # fooling images + already classified
        continue
    subdir = os.path.join(data_root, dir)
    print "Entering folder: " + subdir
    # Used for error rates
    top_1_correct = 0;
    top_5_correct = 0;
    total = 0;
    # For all images
    total_images = len(next(os.walk(subdir))[2])
    progress = 0
    for file in next(os.walk(subdir))[2]:
        if (file.endswith(".JPEG")):
            # Input image
            IMAGE_FILE = os.path.join(subdir, file)
            print "Classifying: " + IMAGE_FILE
            input_image = caffe.io.load_image(IMAGE_FILE)

            # Create data folder if not already existing
            folder = subdir + "/" + os.path.splitext(file)[0] + "/"
            if not os.path.exists(os.path.dirname(folder)):
                os.makedirs(os.path.dirname(folder))

            # Prediction
            net.blobs['data'].reshape(1,3,227,227)
            net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(IMAGE_FILE))
            out = net.forward()
            feat = net.blobs['prob'].data[0]
            # Confidence Plot
            plt.figure(figsize=(20,10))
            plt.plot(feat.flat)
            plt.xlabel('Class')
            plt.ylabel('Confidence')
            plt.title('Confidence Plot')
            plt.savefig(folder + "/confidence_plot.jpg", bbox_inches='tight')
            plt.close()

            # Save input image
            fig = plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(folder + "/input_image.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Top 5 predictions
            # Sort top k predictions from softmax output
            top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
            top5 = labels[top_k]
            certainties = out['prob'][0][top_k]
            f = open(folder + "/top5.txt", "w")
            for label, certainty in zip(top5, certainties):
                f.write(label + '  ->  ')
                f.write(str(certainty))
                f.write('\n')
            f.close()

            # Update variables used for error rates
            if dir in top5[0]:
                top_1_correct += 1
            for label in top5:
                if dir in label:
                    top_5_correct += 1
                    break
            total += 1

            # Layer 1 output
            feat = net.blobs['conv1'].data[0]
            data = vis_square(feat, padval=1)
            np.save(folder + '/layer1_output', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 1 Output')
            plt.savefig(folder + "/layer1_output.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 1 after pooling
            feat = net.blobs['pool1'].data[0]
            data = vis_square(feat, padval=1)
            np.save(folder + '/layer1_pooling', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 1 after Pooling')
            plt.savefig(folder + "/layer1_pooling.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 1 after pooling and normalization
            feat = net.blobs['norm1'].data[0]
            data = vis_square(feat, padval=1)
            np.save(folder + '/layer1_pooling_norm', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 1 after Pooling and Normalization')
            plt.savefig(folder + "/layer1_pooling_norm.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 2 output
            feat = net.blobs['conv2'].data[0]
            data = vis_square(feat, padval=1)
            np.save(folder + '/layer2_output', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 2 Output')
            plt.savefig(folder + "/layer2_output.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 2 after pooling
            feat = net.blobs['pool2'].data[0]
            data = vis_square(feat, padval=1)
            np.save(folder + '/layer2_pooling', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 2 after Pooling')
            plt.savefig(folder + "/layer2_pooling.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 2 after pooling and normalization
            feat = net.blobs['norm2'].data[0]
            data = vis_square(feat, padval=1)
            np.save(folder + '/layer2_pooling_norm', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 2 after Pooling and Normalization')
            plt.savefig(folder + "/layer2_pooling_norm.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 3 output
            feat = net.blobs['conv3'].data[0]
            data = vis_square(feat, padval=0.5)
            np.save(folder + '/layer3_output', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 3 Output')
            plt.savefig(folder + "/layer3_output.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 4 output
            feat = net.blobs['conv4'].data[0]
            data = vis_square(feat, padval=0.5)
            np.save(folder + '/layer4_output', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 4 Output')
            plt.savefig(folder + "/layer4_output.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 5 output
            feat = net.blobs['conv5'].data[0]
            data = vis_square(feat, padval=0.5)
            np.save(folder + '/layer5_output', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 5 Output')
            plt.savefig(folder + "/layer5_output.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Layer 5 after pooling
            feat = net.blobs['pool5'].data[0]
            data = vis_square(feat, padval=1)
            np.save(folder + '/layer5_pooling', data)
            fig = plt.imshow(data)
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title('Layer 5 after Pooling')
            plt.savefig(folder + "/layer5_pooling.jpg", bbox_inches='tight', pad_inches=0.05)
            plt.close()

            # Fully Connected Layer 6 activation
            feat = net.blobs['fc6'].data[0]
            np.save(folder + '/layer6_activation', feat)
            plt.figure(figsize=(80,10))
            plt.plot(feat.flat)
            plt.xlabel('Neuron')
            plt.xlim(xmax=4100)
            plt.xticks(np.arange(0, 4101, 100))
            plt.ylabel('Activation')
            plt.title('Fully Connected Layer 6 Activation')
            plt.savefig(folder + "/layer6_activation.jpg", bbox_inches='tight')
            plt.close()

            # Fully Connected Layer 7 activation
            feat = net.blobs['fc7'].data[0]
            np.save(folder + '/layer7_activation', feat)
            plt.figure(figsize=(80,10))
            plt.plot(feat.flat)
            plt.xlabel('Neuron')
            plt.xlim(xmax=4100)
            plt.xticks(np.arange(0, 4101, 100))
            plt.ylabel('Activation')
            plt.title('Fully Connected Layer 7 Activation')
            plt.savefig(folder + "/layer7_activation.jpg", bbox_inches='tight')
            plt.close()

            # Fully Connected Layer 8 activation
            feat = net.blobs['fc8'].data[0]
            np.save(folder + '/layer8_activation', feat)
            plt.figure(figsize=(20,10))
            plt.plot(feat.flat)
            plt.xlabel('Neuron')
            plt.xticks(np.arange(0, 1001, 100))
            plt.ylabel('Activation')
            plt.title('Fully Connected Layer 8 Activation')
            plt.savefig(folder + "/layer8_activation.jpg", bbox_inches='tight')
            plt.close()

            # Print progress
            progress += 1
            percentage = (float(progress) / total_images) * 100
            print '\r[{0}{1}] {2}%'.format('#'*int(percentage), ' '*(100-int(percentage)), int(percentage))

    # Calculate and store error rates
    top_1_error = float(total - top_1_correct) / total
    top_5_error = float(total - top_5_correct) / total
    f = open(subdir + "/Precision.txt", "w")
    f.write('Top 1 error: ' + str(top_1_error) + '\n')
    f.write('Top 5 error: ' + str(top_5_error) + '\n')
    f.close()
