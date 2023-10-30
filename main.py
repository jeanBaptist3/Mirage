import dataCreation
import dataLoader
import model
import trainer
import torch
import torch.optim as optim
import torch.nn as nn
import csv
from torch.utils.tensorboard import SummaryWriter


def main(iterations,new_data):
    """
    Constants
    """
    start_token = '<SOS>'
    end_token = '<EOS>'
    source_flag = 'source_sequence'
    target_flag = 'target_sequence'
    """
    This section contains the Variables you can choose to train the model with:
    """
    print(f'Iteration : {iterations}')
    
    train_size = 10240
    test_size = 1024
    val_size = 1024
    batch_size = 32
    encoder_blocks = 2
    decoder_blocks = 1
    bytes_per_token = 1
    embedding_dim = 256
    max_input_length = encoder_blocks * 64 + 8 + 1
    max_t_length = decoder_blocks * 64  + 1 #target
    num_encoder_layers = 4
    output_dim = 9
    nhead = 8  # Number of attention heads
    num_decoder_layers = 4
    num_epochs = 15
    optimizer = "ASGD"
    model_name = "mirage" + "longS" + optimizer + f'{encoder_blocks}_{iterations}'
    path_model = f"model/dim{embedding_dim}/{model_name}{train_size}.pth "
    data_path = f"data/{train_size}data_dump.csv"
    writer = SummaryWriter("model/logs")

    mask = torch.triu(torch.ones(max_t_length, max_t_length), diagonal=0)
    mask= torch.transpose(mask,0,1)
    mask = mask == 0
    mask = mask.cuda()

    #model initialization with hyperparameters
    model_gpu = model.TransformerModel(embedding_dim, max_input_length, max_t_length, num_encoder_layers,
                                       num_decoder_layers, batch_size, output_dim, nhead, mask)
    model_gpu = model_gpu.cuda()
    print("created model")
    loss_fn = nn.BCELoss()  # BCELoss for binary prediction
    optimizer = optim.ASGD(model_gpu.parameters(), lr=0.005)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    full_size = train_size + test_size + val_size
    #creating the examples with a new key or using the old ones
    if new_data:
        dataCreation.create_data(generated_blocks=encoder_blocks, prediction_blocks=decoder_blocks,
                                 bytes_per_token=bytes_per_token, train_size=train_size, test_size=test_size,
                                 val_size=val_size, path=data_path,start_token=start_token,end_token=end_token)
        print("Data Creation Complete")

 

    #loading the data from files
    dataframe = dataLoader.load_csv(path=data_path)
    token_to_tensor, tensor_to_token = dataLoader.create_tensor_dict(start_token=start_token,end_token=end_token)
    train_data_loader, val_data_loader, test_data_loader = dataLoader.create_data_loader(dataframe=dataframe,
                                                                                         train_ratio=train_size / full_size,
                                                                                         val_ratio=val_size / full_size,
                                                                                         token_to_tensor=token_to_tensor,
                                                                                         b_size=batch_size,
                                                                                         source_flag=source_flag,
                                                                                         encoder_flag=target_flag)
    print("created dataset and dataloader")

    model_trained = trainer.train(model_gpu, num_epochs, optimizer, loss_fn, train_data_loader, val_data_loader,
                                  test_data_loader, path_model,writer=writer,iteration=iterations)
    results = trainer.evaluate_model(model_gpu=model_trained, test_data_loader=test_data_loader,
                                     tensor_to_token=tensor_to_token,token_to_tensor=token_to_tensor, b_size=batch_size,writer=writer)
    header = ['batch index', 'accuracy in batch']

    with open(f'results/{model_name}_accuracy', 'w', encoding='UTF8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for text in results:
            writer.writerow(text)


if __name__ == "__main__":
    iterations = 5
    for i in range(iterations) :
        if(i > 0) :
            main(iterations=i,new_data=False)
        else:
            main(iterations=i,new_data=True)
