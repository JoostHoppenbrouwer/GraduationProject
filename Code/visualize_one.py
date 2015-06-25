import numpy as np
import matplotlib.pyplot as plt
import caffe
from Tkinter import Tk
from tkFileDialog import askopenfilename

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

# Ask the user for an image
print 'Choose image file:'
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
IMAGE_FILE = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(IMAGE_FILE)

net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(IMAGE_FILE))
out = net.forward()
print("Predicted class is #{}.\n".format(out['prob'].argmax()))

# Show input image
fig = plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
plt.axis('off')
plt.title('Input Image')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# The parameters are a list of [weights, biases]
# Layer 1 output
feat = net.blobs['conv1'].data[0]
data = vis_square(feat, padval=1)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 1 Output')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 1 after pooling
feat = net.blobs['pool1'].data[0]
data = vis_square(feat, padval=1)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 1 after Pooling')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 1 after pooling and normalization
feat = net.blobs['norm1'].data[0]
data = vis_square(feat, padval=1)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 1 after Pooling and Normalization')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 2 output
feat = net.blobs['conv2'].data[0]
data = vis_square(feat, padval=1)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 2 Output')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 2 after pooling
feat = net.blobs['pool2'].data[0]
data = vis_square(feat, padval=1)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 2 after Pooling')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 2 after pooling and normalization
feat = net.blobs['norm2'].data[0]
data = vis_square(feat, padval=1)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 2 after Pooling and Normalization')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 3 output
feat = net.blobs['conv3'].data[0]
data = vis_square(feat, padval=0.5)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 3 Output')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 4 output
feat = net.blobs['conv4'].data[0]
data = vis_square(feat, padval=0.5)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 4 Output')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 5 output
feat = net.blobs['conv5'].data[0]
data = vis_square(feat, padval=0.5)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 5 Output')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Layer 5 after pooling
feat = net.blobs['pool5'].data[0]
data = vis_square(feat, padval=1)
fig = plt.imshow(data)
plt.axis('off')
plt.title('Layer 5 after Pooling')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Fully Connected Layer 6 activation
feat = net.blobs['fc6'].data[0]
plt.plot(feat.flat)
plt.xlabel('Neuron')
plt.ylabel('Activation')
plt.title('Fully Connected Layer 6 Activation')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Fully Connected Layer 7 activation
feat = net.blobs['fc7'].data[0]
plt.plot(feat.flat)
plt.xlabel('Neuron')
plt.ylabel('Activation')
plt.title('Fully Connected Layer 7 Activation')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Fully Connected Layer 8 activation
feat = net.blobs['fc8'].data[0]
plt.plot(feat.flat)
plt.xlabel('Neuron')
plt.ylabel('Activation')
plt.title('Fully Connected Layer 8 Activation')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Confidence plot
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.xlabel('Classs')
plt.ylabel('Confidence score')
plt.title('Confidence Plot')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Load labels
imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

# Sort top 5 predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
top5 = labels[top_k]
confidences = out['prob'][0][top_k]
print "Top 5:\n"
for label, confidence in zip(top5, confidences):
    print label + ':'
    print "Confidence: " + str(confidence) + '\n'
