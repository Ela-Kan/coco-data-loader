
# MS COCO Data Loader

This project contains two classes, a data splitter (train/test/validation) (*SplitCOCO.py*) and a data loader—including an augmentation pipeline (*LoadCOCO.py*). 
The purpose of the code is to load in [COCO](https://cocodataset.org/#format-data) formatted data from a .json file, and appropriately prepare it for use in deep learning applications.

## Acknowledgements

 - Example dataset taken from [GLENDA v1.5](http://ftp.itec.aau.at/datasets/GLENDA/) (*coco.json*) [1].
 The original use for this code was within a coursework project, seeking to achieve accurate multiclass segmentation of the above dataset—aiming to improve the 
 diagnosis of endometriosis.



## Author

[@Ela-Kan](https://github.com/Ela-Kan)


## Demo

Example image loaded following random augmentations. (a) Original mask [1], (b) Original image [1], (c) Augmented mask, (d) Augmented image.
![alt text](https://github.com/[Ela-Kan]/[coco-data-loader]/blob/[main]/Example_Augmentation.png?raw=true)


## Instructions for Usage

### Package installations
The following packages are required: 
- **numpy**
  `conda install numpy`
- **cv2**
  `conda install -c conda-forge opencv`
- **albumentations**
  `conda install -c conda-forge albumentations`
- **scikit-learn**
  `conda install scikit-learn`
- **pycocotools.coco**
  `conda install -c conda-forge pycocotools`

#### Note on pycocotools issues for Windows
If using Windows and you have installation issues, please use the alternate command:
`pip install pycocotools-windows`

Alternatively, if this also doesn't work, try [this](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/62381):
1) Install Visual C++ 2015 Build Tools (https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019)
2) Go to C:\Program Files (x86)\Microsoft Visual C++ Build Tools and run vcbuildtools_msbuild.bat
3) Then, in conda environment: `pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI`



## References
[1] A. Leibetseder, S. Kletz, K. Schoeffmann, S. Keckstein and J. Keckstein. 2020. GLENDA: Gynecologic Laparoscopy Endometriosis Dataset. In Proceedings of the 26th International Conference on Multimedia Modeling, MMM 2020. Springer, Cham.
## License

[MIT](https://choosealicense.com/licenses/mit/) License

Copyright (c) 2022, Ela Kanani

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