# Mirage
This Repository is the practical part of a Bachelor Thesis. The project is mainly split into the data handling, model architecture, training and the main class combining everything. 
Using the method, one can create a model with his own set of hyperparameters fixed on top of the main Method. 

### data handling

The data handling consists of the dataCreation, which is responsible for the creation of the dataset. It mainly consists of the method create_data() with the following parameters:
- generated_blocks represents how many blocks from the ChaCha20 Stream you want to use for the input_sequence
- prediction_blocks represents how many blocks after the generated_blocks from the ChaCha20 Stream you want to use for the target_sequence
- bytes_per_block represents how many bytes are in one token
- train_size: amount of data used for training
- val_size: amount of data used for validation
- test_size: amount of data used for testing
- path: path used for saving the data 



-TODO
-


