import Training
import Models
import Application
import sys
# Tests randoms values for certain hyperparameters
# Performs 5-Fold cross validation on training/validation set


# Performs n_trials 5-fold CVs with random hyperparam values.
# Only tunes on train/validation set, does not see test set
#Training.hyper_parameter_tuning(Models.Base_CNN, n_trials=1)

# Perform a single 5-fold CV on a net. Used to test architecture.
#Training.train_net(Models.Base_CNN, tuning=True)

# Trains a model with hyperparams previously tuned with 5-fold CV
# Uses entire training and validation set to train, is tested on withheld test set
# Saves the model for later use.
#print("FINAL MODEL")
#Training.train_final_model(Models.Base_CNN, 'Final_Model',)

# Loads a saved model and runs it on the test set.
# Generates the test metrics.
# Training.load_and_run_model(Models.Base_CNN, 'Final_Model_Base_CNN')

# Runs a model in application mode. Loads a pretrained model from model_path to make a prediction on the image at img_path
# Application.application_mode(model_type=Models.Base_CNN,
#                              model_path='Final_Model_Base_CNN',
#                              img_path='./data/aug_1/cloth_mask/0_00001.jpeg')

#Training.train_final_model(Models.Less_Conv_CNN, 'Final_Model',)
#print("Loading Less Conv")
#Training.load_and_run_model(Models.Less_Conv_CNN, 'Final_Model_Less_Conv')

#print("Training Less Pool")
#Training.train_final_model(Models.Less_Pooling_CNN, 'Final_Model',)

#Training.load_and_run_model(Models.Less_Pooling_CNN, 'Final_Model_Less_Pooling')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        flag = sys.argv[1]
        if flag == '-r':
            print('load and test the Base_CNN model')
            Training.load_and_run_model(Models.Base_CNN, 'Final_Model_Base_CNN')
        elif flag == '-p':
            print('get prediction with Base_CNN model')
            image_path = sys.argv[2]
            Application.application_mode(model_type=Models.Base_CNN,
                             model_path='Final_Model_Base_CNN',
                             img_path=image_path)
        elif flag == '-t':
            print('train the Base_CNN model')
            Training.train_final_model(Models.Base_CNN, 'Final_Model',)
        
        elif flag == '-test_all':
            Application.application_mode_imageset(Models.Base_CNN, 'Final_Model_Base_CNN', './data/test_data/')
            