
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
from sklearn.model_selection import train_test_split


class splitCOCO:
    """ Class splits the .json file into test, train and validation .json files using a 70:20:10 (train : test : validation) split"""

    def __init__(self,  file_name):
        """Pre-allocate all neccessary variables."""
        self.file_name = file_name
        self.full_dataset = dict()
        self.train = dict()
        self.test = dict()
        self.val = dict()

    def loadCOCO(self):
        """Given a filename to the complete COCO containing all of the data, a dictionary is created from the json information and information about the file is printed
        Args:
            file_name (str): name of file where COCO data is stored
        Returns:
            full_dataset (dictionary): dictionary containing all data."""
        # Load in data from COCO file
        coco_file = open(self.file_name, encoding='utf-8')  # open the specific file
        self.full_dataset = json.load(coco_file)  # read in the whole json information

        # Relay information abput the dataset
        print('Number of Categories: ', len(self.full_dataset['categories']))
        print('Number of Images: ', len(self.full_dataset['images']))
        print('Number of Annotations: ', len(self.full_dataset['annotations']))

        return self.full_dataset


    def splitData(self):
        """Takes the full_dataset dictionary and splits it into train/test/validation dictionaries in ratio 70:20:10
        Args:
            full_dataset (dict): a complete json coco dataset
        Returns:
            train (dict): training data
            val (dict): validation data
            test (dict): testing data
        """
        # Split COCO into train/test/validation using 'train_test_split() from sklearn into 70:20:10 (train : test : validation) split
        train_images, val_images, = train_test_split(
            self.full_dataset['images'], test_size=0.3)
        val_images, test_images = train_test_split(
            val_images, test_size=(2/3))

        # Create dictionaries with relevant information
        self.train['info'] = self.full_dataset['info']
        self.train['categories'] = self.full_dataset['categories']
        self.train['images'] = train_images
        self.train['annotations'] = self.fetchAnnotations(train_images)  # find corresponding annotations

        self.val['info'] = self.full_dataset['info']
        self.val['categories'] = self.full_dataset['categories']
        self.val['images'] = val_images
        self.val['annotations'] = self.fetchAnnotations(val_images)  # find corresponding annotations

        self.test['info'] = self.full_dataset['info']
        self.test['categories'] = self.full_dataset['categories']
        self.test['images'] = test_images
        self.test['annotations'] = self.fetchAnnotations(test_images)  # find corresponding annotations

        # Print info
        print('Number of Training images (before augmentation):', len(train_images))
        print('Number of Validation images:', len(val_images))
        print('Number of Testing images:', len(test_images))
        return self.train, self.test, self.val


    def fetchAnnotations(self, images):
        """Given the full_dataset (the json), and a list of image information, find the corresponding annotations to the selected images
        Args:
            full_dataset (dict): a complete json coco dataset
            images (list): image information from desired split (e.g. test/train/val)
        Returns:
            annotations_list (list): annotations corresponding to the given images
        """
        # iterate through all images in image set and find corresponding annotations
        annotations_list = []
        for image in range(len(images)):  # search each image
            # iterate through all annotations
            for annotation in self.full_dataset['annotations']:
                # if annotation matches current image store it
                if annotation['image_id'] == images[image]['id']:
                    annotations_list.append(annotation)
        return annotations_list

    def saveTrainTestValidationCOCO(self):
        """Saves train, test and validation dictionaries into individual COCO .json files
        Args:
            train (dict): training data
            val (dict): validation data
            test (dict): testing data
            """
        # Create new json files with the information from the above split by creating dictionaries for test, train and validation and save these
        with open('coco_train.json', 'w', encoding='utf-8') as train_file:  # write training data to file
            json.dump(self.train, train_file, ensure_ascii=False)

        with open('coco_val.json', 'w', encoding='utf-8') as val_file:  # write training data to file
            json.dump(self.val, val_file, ensure_ascii=False)

        with open('coco_test.json', 'w', encoding='utf-8') as test_file:  # write training data to file
            json.dump(self.test, test_file, ensure_ascii=False)
    
    def run(self):
        self.loadCOCO()  # load COCO data
        self.train, self.test, self.val = self.splitData()  # create test/train/val split
        # save split COCO data into new .json files
        self.saveTrainTestValidationCOCO()


if __name__ == "__main__":
    splitCOCO('coco.json').run()
