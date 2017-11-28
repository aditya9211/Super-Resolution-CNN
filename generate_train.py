import numpy as np
import glob
import os
import scipy.misc as ms
import argparse

# Intilaizing the Input and Output Dim of Images
size_label = 21
size_input = 33
scale = 3	# UpScaling or Zoom factor 
stride = 20
padding = abs(size_input - size_label)/2

# Adjusting the images when the dimension is not a multiple of 33X33 
def modcrop(image, scale):
	sz = image.shape
	sz = sz - np.mod(sz, scale)
	imgs = image[0:sz[0], 0:sz[1]]
	return imgs

# Costruct a Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-train", "--train", required=True, help="path to train_images directory")
args = vars(ap.parse_args())
path = args["train"]
path = path + str("/*.bmp")


## Converting the Images in flatten numpy array format 
data  = np.zeros((1, size_input, size_input))
label = np.zeros((1, size_label, size_label))

print 'Converting The Images .... '
for filename in glob.glob(path):
	image = ms.imread(filename, mode='L')
	im_label = modcrop(image, scale)
	sz = im_label.shape
	im_input = ms.imresize(ms.imresize(im_label,(sz[0]/scale, sz[1]/scale),interp= 'bicubic'),sz,interp='bicubic')

	for x  in np.arange (0, sz[1]-size_input, stride):
		for y in np.arange(0, sz[0]-size_input, stride):
			subim_input = im_input[y : y+size_input, x : x+size_input]
			subim_label = im_label[y+padding : y+padding+size_label, x+padding : x+padding+size_label]
			data  = np.vstack([data,  [subim_input]])
           		label = np.vstack([label, [subim_label]])


print data.shape, label.shape
print
print 'Permuting the Array .....'
sample = np.random.choice(data.shape[0],data.shape[0])
data  = data[sample,:,:]
label = label[sample,:,:]


filename = "/tmp/SRCNN/"
if not os.path.exists(os.path.dirname(filename)):
	print 'Creating Dir ' + filename + '....'
	os.makedirs(os.path.dirname(filename))


print 'Saving to /tmp/SRCNN/ .......'
np.save(filename + str("/data.npy"), data)
np.save(filename + str("/label.npy"), label)
       
