import Training
import Models

Training.train_final_model(Models.Base_CNN, 'Final_Model',)

Training.load_and_run_model(Models.Base_CNN, 'Final_Model_Base_CNN')

#Training.train_final_model(Models.Less_Conv_CNN, 'Final_Model',)

#Training.load_and_run_model(Models.Less_Conv_CNN, 'Final_Model_Less_Conv')

#Training.train_final_model(Models.Less_Pooling_CNN, 'Final_Model',)

#Training.load_and_run_model(Models.Less_Pooling_CNN, 'Final_Model_Less_Pooling')