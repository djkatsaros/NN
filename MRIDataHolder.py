#import sh
import matplotlib
import numpy as np
import os
import csv
import fnmatch
from skimage import  img_as_float
from skimage.io import imread
import string
import matplotlib.pyplot as plt
class MRIDataHolder(object):
    def __init__(self, path, img_size=(256,256,3),use_csv=True):
        self.data_paths = []
        self.diag_birad = []
        self.diag_ben_mal = []
        if use_csv:
            path_idx = 12
            birad_idx = 8
            ben_mal_idx = 9
            with open("mass_case_description_train_set.csv",'r') as csvfile:
                phenotypes = csv.reader(csvfile)
                for row in phenotypes:
                    self.data_paths.append("ROI_AND_CROPPED/DOI/" +
string.replace(row[path_idx],'\\','/')  + ".png")
                    #TODO:make path general
                    self.diag_birad.append(row[birad_idx])
                    self.diag_ben_mal.append((row[ben_mal_idx]))
            self.data_paths = self.data_paths[1:]
            self.diag_birad = self.diag_birad[1:]
            self.diag_ben_mal = self.diag_ben_mal[1:] #chop off column names 
            #remove masks
            for idx,p in enumerate(self.data_paths):
                im = imread(p)
                if np.any((im > 0) & (im < 1)):
                    self.data_paths.pop(idx)
                    self.diag_birad.pop(idx)
                    self.diag_ben_mal.pop(idx)

        #TODO:fix later for use when csv not available
        #for root, dir, file in os.walk("ROI_AND_CROPPED"):
         #   for f in [os.path.join(root,x) for x in file if fnmatch.fnmatch(x,'*.png')]:
          #      self.data_paths.append(f)
    def batchGenerator(self, batch_size):
        i = 0
        while batch_size*i < len(self.data_paths):
            images = []
            batch_end = batch_size*(i+1) if batch_size*(i+1) < len(self.data_paths) else len(self.data_paths)-1
            print batch_end
            for p in self.data_paths[i*batch_size:batch_end]:
                im = img_as_float(imread(p))
                if np.any(im > 1): #check if it is an ROI and not a mask
                    images.append(im)
            if not images:
                i+=1
                continue
            yield images
            i+=1



if __name__ == "__main__":
	test = MRIDataHolder("ROI_AND_CROPPED")
	batchGen = test.batchGenerator(1)
	print test.data_paths
	batch = batchGen.next()
        print np.sum(batch[0])
	#plt.imshow(batch[3])
	#plt.show()
