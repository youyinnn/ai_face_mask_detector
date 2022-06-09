### AI Face Mask Detector
Latex Online Site:https://www.overleaf.com/project/629d23fca51a5722ec9d3557

Team name:OB_13

Team members:Alexander Fulleringer, Jun Huang, Cheng Chen

ID numbers: Alexander Fulleringer, Jun Huang, 40222770

How To Use The Program:

Driver.py is used to run the program itself and includes samples of all major function calls needed to train and characterize the nets.

Note that Application.py functions use images found in test_data by default. Training.py methods rely on the entire dataset usualy kept in aug_1

To load a model and run it on 1 image use Application.application_mode_single_image(model_type, saved_file_path,image_path)
This will load an image at image_path and run it through the saved model of type model_type found at saved_file_path

To load a model and run it on all images in a folder use Application.application_mode)imageset(model_type, saved_file_path,folder_path)
This loads the images in that folder and then tests the model on them before output the evaluation data.

To tune hyperparameters with 5-fold cross-validation run Training.hyper_parameter_tuning(model_type, num_trials).
The model type should be a reference to a nn.module class. Our models are in Models.py so Models.Base_CNN is an appropriate input.
num_trials is simply how many configurations you'd like to try.

To tune a model manually wihtout using the hyperparam tuning, use Training.train_net(model_type,  tuning=True) This does k-fold cross validation and was used to test different model configurations.

To train and save a final model run Training.train_final_model(model_type, filepath)
where model_type is as above and the filepath is where you'd like to save it.
This trains on the entire trianing and validation set before evaluating on the withheld test data.

To load a model and run it (by default on the test set) use Training.load_and_run(model_type, saved_file_path)
Modeil_type as above and saved_file_path should be where the model file was previously saved.



Dataset: 
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
000097.jpg~000387.jpg 



