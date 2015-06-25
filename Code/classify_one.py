import numpy as np
import matplotlib.pyplot as plt
import caffe
from Tkinter import Tk
from tkFileDialog import askopenfilename

# The setup is divided into three folder: Caffe, Code and Data
# This file should be in the Code folder
caffe_root = '../Caffe/'
data_root = '../Data/'

# Set the right path to your model definition file and pretrained model weights
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

# Ask the user for an image
print 'Choose image file:'
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
IMAGE_FILE = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(IMAGE_FILE)

# Initialize the Deep Neural Network
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

# Prediction
input_image = caffe.io.load_image(IMAGE_FILE)
net.blobs['data'].reshape(1,3,227,227)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(IMAGE_FILE))
out = net.forward()
print("Predicted class is #{}.\n".format(out['prob'].argmax()))
feat = net.blobs['prob'].data[0]
plt.plot(feat.flat)
plt.xlabel('Class')
plt.ylabel('Confidence score')
plt.title('Confidence Plot')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# Sort top 5 predictions from softmax output
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
top5 = labels[top_k]
confidences = out['prob'][0][top_k]
print "Top 5:\n"
for label, confidence in zip(top5, confidences):
    print label + ':'
    print "Confidence: " + str(confidence) + '\n'
