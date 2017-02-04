#import sh
import matplotlib
import numpy as np
import os
import shutil
import csv
from skimage import  img_as_float
from skimage.io import imread
import string
def gen_class_links(link_path,use_csv = True, birad = False):
    if birad == True:
        raise NameError("Sorry, birad sorting not yet implemented")
    subjects = {}
    if use_csv:
        path_idx = 12
        birad_idx = 8
        ben_mal_idx = 9
        with open("mass_case_description_train_set.csv",'r') as csvfile:
            phenotypes = csv.reader(csvfile)
            for row in phenotypes:
                next(phenotypes)
                diagnosis = row[birad_idx] if birad == True else row[ben_mal_idx]
                #fix slash weirdness
                path = string.replace(row[path_idx],'\\','/')
                png_path = "/home/spiro/ROI_AND_CROPPED/DOI/" + path + ".png"
                subjects.update({ png_path : str(diagnosis)})
                    #TODO:make path general
        #os.makedirs('./Benign','./Malignant')
        for p,d in subjects.iteritems():
            im = img_as_float(imread(p)) #float --> signif slowdown?
            if np.any((im > 0) & (im < 1)):
                filename = string.replace(p, '/', '_')
                if d.find("BENIGN"):
                    os.symlink(p,link_path + '/Benign/' + filename)
                else:
                    os.symlink(p, link_path + '/Malignant/' + filename)




if __name__ == "__main__":
    gen_class_links(link_path = "/home/spiro/link_path")
