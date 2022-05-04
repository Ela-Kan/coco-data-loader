
"""
MIT License

Copyright (c) [2022] [Ela Kanani]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
from pycocotools.coco import COCO
from SplitCOCO import splitCOCO # import class to split the coco.json file into test/train/split coco files
import cv2
import albumentations as A
from PIL import Image

class loadCOCO:

    def __init__(self,  file_name):
        # Split COCO json file into three separate files
        splitCOCO(file_name).run() 

        # Create COCO objects for each test, train, and validation
        self.total_coco = COCO("coco.json")
        self.train_coco = COCO("coco_train.json")
        self.test_coco = COCO("coco_test.json")
        self.val_coco = COCO("coco_val.json")

        self.category_ids = []

    def printInfo(self):
        """ Print info for user's interest, regarding the break down of test/split/vaildation
        Args:
            None. 
        """
        # Obtain Category IDs
        self.category_ids = self.total_coco.getCatIds()
        self.categories = self.total_coco.loadCats(self.category_ids)
        # Print how many of each category exists in each json file
        print("----------------- Training data breakdown -------------------")
        self.categoriesPerSplit(self.train_coco)
        print("----------------- Testing data breakdown --------------------")
        self.categoriesPerSplit(self.test_coco,)
        print("----------------- Validation data breakdown -----------------")
        self.categoriesPerSplit(self.val_coco)


    def categoriesPerSplit(self, coco):
        """Given a coco object, print how many of each category exists within.
        Args: 
            coco (COCO): coco json file to analyse
        """

        for cats in self.category_ids:
        # Get the ID of all the images containing the object of the category.
            image_ids = coco.getImgIds(catIds=[cats])
            print(f"Number of Images Containing {self.categories[cats-1]['name']}: {len(image_ids)}")


    def DataLoader(self, coco, is_training = False):
        """Given a coco object, data is loaded into images and masks arrays for each test, train and validation.
        Args:
            coco (COCO): coco json file to analyse
            is_training (bool): If true, augmentation should be applied and added to set. Default is False.
        Returns:
            imgs (np.ndarray): images resized as np arrays
            masks (np.ndarray): binary masks corresponding to each image
            """

        im_size = 512 # typical u-net image size
        ids = coco.getImgIds() # extract image IDs
        num_images = len(ids) # Find number of images
        imgs = []
        masks = []

        for data in range(0,num_images):
            # obtain image
            current_id = ids[data] # get image ID
            current_image_info = coco.loadImgs([current_id])[0]
            current_image_file_path =  current_image_info["coco_url"] # extract file path
            current_image = cv2.imread(current_image_file_path, cv2.IMREAD_COLOR)

            # obtain masks
            annotation_ids = coco.getAnnIds(imgIds=[current_id], catIds=self.category_ids, iscrowd=None) # load annotation IDs for current image
            annotations = coco.loadAnns(annotation_ids)
            current_mask = np.zeros(current_image.shape[:2])
            for ann in annotations:
                current_mask = np.maximum(current_mask,coco.annToMask(ann).dot(ann['category_id']))
                #current_mask = (current_mask / len(self.category_ids))*255 # uncomment this to toggle brightness of mask output. Please leave commented for training.
            
            # resize image and mask appropriately for U-Net
            imgs.append(cv2.resize(current_image, (im_size, im_size)))
            masks.append(cv2.resize(current_mask, (im_size, im_size)))
            
            # if training data, apply augmentations
            if is_training == True:
                transformed_image, transformed_mask = self.DataAugmenter(current_image, current_mask)
                imgs.append(cv2.resize(transformed_image, (im_size, im_size)))
                masks.append(cv2.resize(transformed_mask, (im_size, im_size)))

        return imgs, masks 

    def DataAugmenter(self, image, mask):
        """Given an image and mask, apply random transformations. Each has a probability of 0.5 of occuring.
        Args:
            image (np.ndarray): training image to augment 
            mask (np.ndarray): corresponding mask to input image
        Returns:
            transformed_image (np.ndarray): augmented image
            transformed_mask (np.ndarray): corresponding augmented mask
        """
        
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # Random flips in horizontal planes
            A.VerticalFlip(p=0.5),  # Random flips in vertical planes
            A.RandomRotate90(p=0.5),   # Randomly rotates by 0, 90, 180, 270 degrees
            A.CLAHE(p=0.5), # Random colour histogram equalisation
            A.Blur(p=0.5), # Random Blur
            A.Perspective(p=0.5), # Random Perspective Transformation
            A.ColorJitter(brightness=0, contrast=0, saturation=0.2, hue=0, always_apply=False, p=0.5) # Saturation Change
        ])

        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']  

        return transformed_image, transformed_mask 

    def convertArrayToImage(self, list_of_arrays):
        """Takes a list of images in np.array format and converts to PIL Image, as suitable for chosen training architecture.
        Args: list_of_arrays (list of np.ndarrays): list of images to convert
        Returns: list_of_images (list of PIL Images): list of converted images """

        list_of_images = []
        for array in range(0,len(list_of_arrays)):
            img = Image.fromarray(list_of_arrays[array].astype('uint8')).convert('RGB')
            list_of_images.append(img)
        return list_of_images



    def run(self):
        """Function to run all neccessary methods to obtain training, testing and validation masks and images. This includes data augmentations.
        Args:
            None.
        Returns:
             train_imgs (list of PIL images): list of training images
             train_msks (list of PIL images): list of training masks
             test_imgs (list of PIL images):  list of test images
             test_msks (list of PIL images):  list of test masks
             val_imgs (list of PIL images): list of val images
             val_msks (list of PIL images): list of val masks
        Note:
            Each index of the list corresponds to matching image/mask pairs. E.g. train_imgs[0] corresponds to train_msks[0]
            """
        self.printInfo()
        # obtain training images without augmentations
        train_imgs, train_msks = self.DataLoader(self.train_coco, is_training=True)
        test_imgs, test_msks = self.DataLoader(self.test_coco)
        val_imgs, val_msks = self.DataLoader(self.val_coco)

        # Randomly apply augmentations to training data to obtain 4x the amount of data
        original_train_size = len(train_imgs)
        for iter in range(0,2):
            for data in range(original_train_size): 
                transformed_image, transformed_mask = self.DataAugmenter(train_imgs[data], train_msks[data])
                train_imgs.append(transformed_image)
                train_msks.append(transformed_mask)
        
        # Return information
        print('----------------- After Augmentation -----------------')
        print('Number of training images: ' + str(len(train_msks)))

        # Convert np array data into PIL images for UNet Training
        train_imgs = self.convertArrayToImage(train_imgs)
        train_msks = self.convertArrayToImage(train_msks)
        test_imgs = self.convertArrayToImage(test_imgs)
        test_msks = self.convertArrayToImage(test_msks)
        val_imgs = self.convertArrayToImage(val_imgs)
        val_msks = self.convertArrayToImage(val_msks)
        
        # returns full dataset including augmentations
        return train_imgs, train_msks, test_imgs, test_msks, val_imgs, val_msks

    
if __name__ == "__main__":
    data = loadCOCO('coco.json')
    train_imgs, train_msks, test_imgs, test_msks, val_imgs, val_msks = data.run()

    # To see all training masks uncomment this
    """
    for i in range(0,len(train_msks)):
        cv2.imwrite(f"current_mask{i}.png", train_msks[i])
        cv2.imwrite(f"current_img{i}.png", train_imgs[i])
    """

    

    
