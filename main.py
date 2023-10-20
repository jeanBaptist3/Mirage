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
    train_size = 10000
    test_size = 1000
    val_size = 1000
    batch_size = 32
    generated_blocks = 4
    prediction_blocks = 1
    bytes_per_token = 1
    embedding_dim = 256  #embedding dimension
    max_input_length = generated_blocks*64 + 8  # Input Len
    max_t_length = prediction_blocks*64  # target len
    num_encoder_layers = 4 # Number of encoder layers
    output_dim = 8
    nhead = 8  # Number of attention heads
    num_decoder_layers = 4
    num_epochs = 10
    model_name = "mirage" + "Big"
    path_model = f"model/dim{embedding_dim}/{model_name}{train_size}.pth "
    data_path = f"data/{train_size}data_dump.csv"
    new_data = "y"



    model_gpu = model.TransformerModel(embedding_dim, max_input_length, max_t_length, num_encoder_layers,num_decoder_layers, batch_size, output_dim, nhead)
    model_gpu =  model_gpu.cuda()
    print("created model")
    loss_fn = nn.BCELoss()  # Use BCELoss for binary sequence prediction
    optimizer = optim.Adam(model_gpu.parameters(), lr=0.0001)  # Adjust the learning rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    full_size = train_size+test_size+val_size
    if new_data == "y" :
        dataCreation.create_data(generated_blocks=generated_blocks,prediction_blocks=prediction_blocks,bytes_per_token=bytes_per_token,train_size=train_size,test_size=test_size,val_size=val_size,path = data_path)
        print("klappt")

    print("Data Creation Complete")
    dataframe = dataLoader.load_csv(path=data_path)
    token_to_tensor, tensor_to_token = dataLoader.create_tensor_dict()
    train_data_loader,val_data_loader,test_data_loader = dataLoader.create_data_loader(dataframe=dataframe,train_ratio=train_size/full_size,val_ratio=val_size/full_size,token_to_tensor=token_to_tensor,b_size=batch_size)
    print("created dataset and dataloader")

    trainer.train(model_gpu,num_epochs,optimizer,loss_fn,train_data_loader,val_data_loader,test_data_loader,path_model)
if __name__ =="__main__":
    main()

