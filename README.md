## AI Face Mask Detector

> Team name:OB_13
>
> Team members:
>
> - Alexander Fulleringer(40005290)
> - Jun Huang(40168167)
> - Cheng Chen(40222770)

## Files Introduction

``` bash
├── data                                        # dataset files
│   ├── aug_1                                   # data that had been augmented for training and testing
│   │   ├── cloth_mask                          
│   │   ├── mask_worn_incorrectly
│   │   ├── n95_mask
│   │   ├── no_face_mask
│   │   ├── surgical_mask
│   ├── test_data                               # dataset for small testing
│   │   ├──   .
│   │   ├──   .
│   │   ├──   .
├── data_process                                # data process or loading module
│   ├── auto_crop.py                            # crop the image into 1:1 ratio
│   ├── DataAugmentation.py                     # image augmentation  
│   ├── DatasetHelper.py                        # image loader for pytorch
│   ├── ImageResize.py                          # resize the image into 256x256
├── Application.py                              # application that take one image and return prediction
├── Driver.py                                   # driver program for training, testing, running the model
├── Models.py                                   # contains all the model architectures
├── Training.py                                 # hosts the training process
├── Validation.py                               # validate the output during the training
├── evaluation.py                               # evaluate the model
├──     .
├──     .
└── .gitignore
```

## How To Use The Program
### 1. Preprare the Dataset and Env

Please make sure you have the `data/aug_1` located at the root folder of the project!

Install required packages:

``` bash
pip install -r requirements.txt
```

### 2. Play with the Project

If you want to playwith our trained model, please make sure you have the `Final_Model_Base_CNN`, `Final_Model_Less_Conv`, and `Final_Model_Less_Pooling` model files located at the root folder of the project!

For predicting one image, you can run `Driver.py` with flag `-p`:

``` bash
python Driver.py -p ./data/aug_1/mask_worn_incorrectly/4_00001_aug_2.jpeg
```

### 3. Train the Model Yourself

#### 3.1 Train with One Command

For training the `Models.Base_CNN`, you can run `Driver.py` with flag `-t`:

``` bash
python Driver.py -t
```

This will train the whole new `Base_CNN` model.
#### 3.2 Test the Base_CNN with One Command

For loading and testing the trained `Models.Base_CNN`, you can run `Driver.py` with flag `-t`:

``` bash
!python Driver.py -t
```

#### 3.3 More Details of the `Driver.py`

This will train the whole new `Base_CNN` model.

`Driver.py` is used to run the program itself and includes samples of all major function calls needed to train and characterize the nets.

Note that `Application.py` functions use images found in test_data by default. Training.py methods rely on the entire dataset usualy kept in `./data/aug_1`

To load a model and run it on 1 image use `Application.application_mode_single_image(model_type, saved_file_path,image_path)`.
This will load an image at image_path and run it through the saved model of type `model_type` found at `saved_file_path`

To load a model and run it on all images in a folder use `Application.application_mode)imageset(model_type, saved_file_path,folder_path)`
This loads the images in that folder and then tests the model on them before output the evaluation data.

To tune hyperparameters with 5-fold cross-validation run `Training.hyper_parameter_tuning(model_type, num_trials)`.
The model type should be a reference to a nn.module class. 
Our models are in `Models.py` so `Models.Base_CNN` is an appropriate input.
num_trials is simply how many configurations you'd like to try.

To tune a model manually wihtout using the hyperparam tuning, use `Training.train_net(model_type,  tuning=True)` This does k-fold cross validation and was used to test different model configurations.

To train and save a final model run `Training.train_final_model(model_type, filepath)`
where `model_type` is as above and the `filepath` is where you'd like to save it.
This trains on the entire trianing and validation set before evaluating on the withheld test data.

To load a model and run it (by default on the test set) use `Training.load_and_run(model_type, saved_file_path)`
where `modeil_type` as above and `saved_file_path` should be where the model file was previously saved.

### 4. Evaluation Visualization

#### 4.1 Evaluate the Trained Model and Its Variants 

Once you trained the models, you should have the following filse:

``` bash
├── Final_Model_Base_CNN                                 # trained model for Base_CNN
├── Final_Model_Less_Conv                                # 
├── Final_Model_Less_Pooling                             # 
├── Final_Model_Less_Pooling                             # 
├── Final_Test_MetricsBase_CNN.npy                       # scores files
├── Final_Test_MetricsLess_Conv.npy                      # 
├── Final_Test_MetricsLess_Pooling.npy                   # 
├──     .
├──     .
└── .gitignore
```

Then you can run the notebook file `model_evaluation_data.ipynb` to visualize the scores.

#### 4.2 Evaluate the Model with a Randomly Picked `test_data` Dataset

Then you can run the notebook file `model_evaluation_data.ipynb` to visualize the scores.


<!-- Dataset:
(Datasource)
https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset/metadata
author: Wobot Intelligence
license:CC0: Public Domain
all images in class cloth_mask, class no_mask, class surgical_mask, 000001.jpg~000096.jpg in class n95

https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection
author: Wvijay kumar
license:CC0: Public Domain
all images in class mask_worn_incorrectly

https://www.kaggle.com/datasets/coffee124/facemaskn95
author: coffee124
license:CC0: Public Domain
000097.jpg~000387.jpg

CNN Architecture:(By Alexander)

Evaluation:
1.Precision:
Recall:
F1-measure:
Accuracy:

2.Confusion matrix for the five classes
(All above are based on test data)

Reference:
PyTorch API:https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

Dataset:
Now using:(before)https://drive.google.com/drive/folders/1B9oVTjYrd7YsIzFzWNkCYO3srLJt0Z7Y
(after)https://drive.google.com/file/d/1-mhCw6ovURDwvl857LfBLlKB69KrK96M/view
(Datasource)
https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset/metadata
author: Wobot Intelligence
license:CC0: Public Domain
all images in class cloth_mask, class no_mask, class surgical_mask, 000001.jpg~000096.jpg in class n95

https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection
author: Wvijay kumar
license:CC0: Public Domain
all images in class mask_worn_incorrectly

https://www.kaggle.com/datasets/coffee124/facemaskn95
author: coffee124
license:CC0: Public Domain
000097.jpg~000387.jpg -->
