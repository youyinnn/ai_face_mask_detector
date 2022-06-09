## AI Face Mask Detector

> Team name:OB_13
>
> Team members:
>
> - Alexander Fulleringer(40005290)
> - Jun Huang(40168167)
> - Cheng Chen(40222770)

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

To tune hyperparameters with 5-fold cross-validation run `Training.hyper_parameter_tuning(model_type, num_trials)`.

The model type should be a reference to a nn.module class. Our models are in `Models.py` so `Models.Base_CNN` is an appropriate input.
num_trials is simply how many configurations you'd like to try.

To tune a model manually wihtout using the hyperparam tuning, use `Training.train_net(model_type, tuning=True)`.
This does k-fold cross validation and was used to test different model configurations.

To train and save a final model run `Training.train_final_model(model_type, filepath)`
where `model_type` is as above and the `filepath` is where you'd like to save it.
This trains on the entire trianing and validation set before evaluating on the withheld test data.

To load a model and run it (by default on the test set) use `Training.load_and_run(model_type, saved_file_path)` with `model_type` as above and saved_file_path should be where the model file was previously saved.

To load a model and run it on 1 image use `Application.application_mode(model_type, saved_file_path, image_path)`.
This will load an image at image_path and run it through the saved model of type model_type found at `saved_file_path`.

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
