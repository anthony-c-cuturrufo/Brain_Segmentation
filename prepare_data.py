import numpy as np
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt
import os 
import shutil
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter

'''
takes np array `x` and reverses the elements across axis 0 and saves to `new_filename`
'''
def add_flip(x, new_filename):
    x_flipped = Image.fromarray(np.flip(x))
    x_flipped.save(new_filename)
    
'''
takes np array `x` and rotates image by 45 degrees and saves to `new_filename`
'''
def add_rotation(x, new_filename):
    x_rotated = Image.fromarray(rotate(x, angle=45, order=0))
    x_rotated.save(new_filename)   

'''
takes np array `x`. Then adds noise if `x` is not a label. Then saves to `new_filename`.
'''
def add_noise(x, new_filename, is_label=False):
    if is_label:
        Image.fromarray(x).save(new_filename)
    else:
        x_noise = Image.fromarray(np.clip(x + np.random.normal(0,100,x.shape), 0, 255))
        x_noise.save(new_filename)
        
'''
takes np array `x`. Then adds blur if `x` is not a label. Then saves to `new_filename`.
'''
def add_blur(x, new_filename, is_label=False):
    if is_label:
        Image.fromarray(x).save(new_filename)
    else:
        x_blur = Image.fromarray(gaussian_filter(x,sigma=5))
        x_blur.save(new_filename)
                
'''
adds flip, rotation, noise, and blur augmentations to dataset
'''
def add_augmentations(path_to_imgs = "imgs/", path_to_labels = "labels/",flip=True, rotation=True, noise=True, blur=True):
    if flop or rotation or noise or blur:     
        for filename in os.listdir(path_to_imgs):
            if filename.endswith(".tif"): 
                image_name = filename[:-4]
                label_name = image_name+"_mask"
                with Image.open(os.path.join(path_to_imgs, filename)) as img:
                    x = np.array(img)
                    if flip: add_flip(x, os.path.join(path_to_imgs, image_name + "_flipped.tif"))
                    if rotation: add_rotation(x, os.path.join(path_to_imgs, image_name + "_rotated.tif"))
                    if noise: add_noise(x, os.path.join(path_to_imgs, image_name + "_noised.tif"))
                    if blur: add_blur(x, os.path.join(path_to_imgs, image_name + "_blurred.tif"))

                with Image.open(os.path.join(path_to_labels, label_name+".tif")) as label:
                    y = np.array(label)
                    if flip: add_flip(y, os.path.join(path_to_labels, label_name + "_flipped.tif"))
                    if rotation: add_rotation(y, os.path.join(path_to_labels, label_name + "_rotated.tif"))
                    if noise: add_noise(y, os.path.join(path_to_labels, label_name + "_noised.tif"), is_label = True)
                    if blur: add_blur(y, os.path.join(path_to_labels, label_name + "_blurred.tif"), is_label = True)


'''
Returns list of binary (0.0 to 1.0) numpy arrays from all .tif files in `path_to_train_data`. 
Uses PIL to open image and then normalizes image. 
'''
def read_train_data(path_to_train_data, path_to_clean_train_data, save_files=False):
    train_data = []
    for filename in os.listdir(path_to_train_data):
        if filename.endswith(".tif"): 
            im = Image.open(os.path.join(path_to_train_data, filename))
            for i, page in enumerate(ImageSequence.Iterator(im)):
                if save_files:
                    page.save(path_to_clean_train_data+filename[2:-4]+ "_" + str(i) + ".tif")
                train_data.append(np.array(page).astype(float)/255.)
    return train_data


'''
Returns list of binary (0.0 to 1.0) numpy arrays from all .tif files in `path_to_train_labels`. 
Uses PIL to open image and then normalizes image. Used to read the labels from set1
'''
def read_train_label_data1(path_to_train_labels, path_to_clean_train_labels, save_files=False):
    train_data = []
    for filename in os.listdir(path_to_train_labels):
        if filename.endswith(".tif"): 
            im = Image.open(os.path.join(path_to_train_labels, filename))
            for i, page in enumerate(ImageSequence.Iterator(im)):
                new_page = np.array(page).astype(float) / 255.0
                if save_files:
                    temp_img = Image.fromarray(new_page)
                    temp_img.save(path_to_clean_train_labels+filename[4:-6]+ "_" + str(i) + "_mask.tif")
                train_data.append(np.array(page).astype(float)/255.)
    return train_data

'''
Returns list of binary (1.0 or 0.0) numpy arrays from all .tif files in `path_to_train_labels`. 
Uses PIL to open image and then if the mode is:
    RGB -> converts binary by choosing most common color to be black and everything else to be white
    P -> inverts the image
Used to read the labels from set2
'''
def read_train_label_data2(path_to_train_labels, path_to_clean_train_labels, save_files = False):
    train_labels = []
    for filename in os.listdir(path_to_train_labels):
        if filename.endswith(".tif"): 
            im = Image.open(os.path.join(path_to_train_labels, filename))
            for i, page in enumerate(ImageSequence.Iterator(im)):
                dominant_color = page.getcolors()[0][1]
                new_page = np.array(page).astype(float)
                if page.mode == 'RGB':
                    temp = np.zeros((new_page.shape[0], new_page.shape[1]))
                    for ii in range(new_page.shape[0]):
                        for jj in range(new_page.shape[1]):
                            if np.array_equal(new_page[ii,jj],np.array(dominant_color)):
                                temp[ii,jj] = 0.0
                            else:
                                temp[ii,jj] = 1.0
                    new_page = temp
                    
                elif page.mode == 'P':
                    new_page = 1 - new_page
                else:
                    raise RuntimeError('cannot handle image mode:' + page.mode)
                if save_files:
                        temp_img = Image.fromarray(new_page)
                        temp_img.save(path_to_clean_train_labels+filename[2:-4]+ "_" + str(i) + "_mask.tif")
                train_labels.append(new_page)
        else:
            continue
    return train_labels

def create_dataset():
    #need to delete M-Khc002-1.tif from set1 since it is not labelled in set1_labelled 
    set1_path = "/Users/anthonycuturrufo/Documents/Research/Han_Lab/data/Images for ground truth/set1/"
    set2_path = "/Users/anthonycuturrufo/Documents/Research/Han_Lab/data/Images for ground truth/set2/"
    set1_labelled_path = "/Users/anthonycuturrufo/Documents/Research/Han_Lab/data/Images for ground truth/Set1_labels/"
    set2_labelled_path = "/Users/anthonycuturrufo/Documents/Research/Han_Lab/data/Images for ground truth/set2_labels/"
    path_to_imgs = "imgs/"
    path_to_labels = "labels/"
    
    if not os.path.exists(path_to_imgs): 
        os.makedirs(path_to_imgs)
        
    if not os.path.exists(path_to_labels): 
        os.makedirs(path_to_labels)
        
    print("preparing truth data in imgs/")
    read_train_data(set1_path, path_to_imgs, save_files = True)
    read_train_data(set2_path, path_to_imgs, save_files = True)
    print("preparing labelled data in labels/")
    read_train_label_data1(set1_labelled_path, path_to_labels, save_files=True)
    read_train_label_data2(set2_labelled_path, path_to_labels, save_files=True)
    
    print("adding augmentations")
    add_augmentations(flip=True, rotation=True, noise=True, blur=True)
    
if __name__ == "__main__":
    create_dataset()