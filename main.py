import dataCreation
import dataLoader
import model
import trainer
import torch
import torch.optim as optim
import torch.nn as nn

def main() :
    """
    This section contains the Variables you can choose to train the model with:
    :return:
    """
    train_size = 100000
    test_size = 10000
    val_size = 10000
    batch_size = 32
    generated_blocks = 2
    prediction_blocks = 1
    bytes_per_token = 1
    embedding_dim = 8  # embedding dimension
    max_input_length = 136  # Input Len
    max_t_length = 64  # target len
    num_encoder_layers = 2  # Number of encoder layers
    num_decoder_layers = 2
    output_dim = 8
    nhead = 8  # Number of attention heads
    num_decoder_layers = 2
    num_epochs = 10
    path_model = "model/mirage.pth "

    model_gpu = model.TransformerModel(embedding_dim, max_input_length, max_t_length, num_encoder_layers,
                                       num_decoder_layers, batch_size, output_dim, nhead)
    print("created model")
    loss_fn = nn.BCELoss()  # Use BCELoss for binary sequence prediction
    optimizer = optim.Adam(model_gpu.parameters(), lr=0.001)  # Adjust the learning rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    full_size = train_size+test_size+val_size
    dataCreation.create_data(generated_blocks=generated_blocks,prediction_blocks=prediction_blocks,bytes_per_token=bytes_per_token,train_size=train_size,test_size=test_size,val_size=val_size)
    print("Data Creation Complete")
    dataframe = dataLoader.load_csv(path="data/data_dump.csv")
    tensor_to_token, token_to_tensor = dataLoader.create_tensor_dict()
    train_data_loader,val_data_loader,test_data_loader = dataLoader.create_data_loader(dataframe=dataframe,train_ratio=train_size/full_size,val_ratio=val_size/full_size,token_to_tensor=token_to_tensor,b_size=batch_size)
    print("created dataset and dataloader")

    trainer.train(model_gpu,num_epochs,optimizer,loss_fn,train_data_loader,val_data_loader,test_data_loader,path_model)
if __name__ =="__main__":
    checkpoint = "google/flan-t5-small"

    main()

