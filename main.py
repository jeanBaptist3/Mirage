import dataCreation
import dataLoader
import model
import trainer
import torch
import torch.optim as optim
import torch.nn as nn
import csv


def main(iterations,new_data):
    """
    This section contains the Variables you can choose to train the model with:
    """
    print(f'Iteration : {iterations}')
    train_size = 10000
    test_size = 1024
    val_size = 1024
    batch_size = 32
    generated_blocks = 64
    prediction_blocks = 1
    bytes_per_token = 1
    embedding_dim = 1024
    max_input_length = generated_blocks * 64 + 8
    max_t_length = prediction_blocks * 64  #target
    num_encoder_layers = 4
    output_dim = 8
    nhead = 8  # Number of attention heads
    num_decoder_layers = 4
    num_epochs = 10
    model_name = "mirage" + "longS" + f'{generated_blocks}_{iterations}'+'NOSOS'
    path_model = f"model/dim{embedding_dim}/{model_name}{train_size}.pth "
    data_path = f"data/{train_size}data_dump.csv"

    mask = torch.triu(torch.ones(max_t_length, max_t_length), diagonal=1)

    #model initialization with hyperparameters
    model_gpu = model.TransformerModel(embedding_dim, max_input_length, max_t_length, num_encoder_layers,
                                       num_decoder_layers, batch_size, output_dim, nhead, mask)
    model_gpu = model_gpu.cuda()
    print("created model")
    loss_fn = nn.BCELoss()  # BCELoss for binary prediction
    optimizer = optim.Adam(model_gpu.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    full_size = train_size + test_size + val_size
    #creating the examples with a new key or using the old ones
    if new_data:
        dataCreation.create_data(generated_blocks=generated_blocks, prediction_blocks=prediction_blocks,
                                 bytes_per_token=bytes_per_token, train_size=train_size, test_size=test_size,
                                 val_size=val_size, path=data_path)
        print("klappt")

    print("Data Creation Complete")

    #loading the data from files
    dataframe = dataLoader.load_csv(path=data_path)
    token_to_tensor, tensor_to_token = dataLoader.create_tensor_dict()
    train_data_loader, val_data_loader, test_data_loader = dataLoader.create_data_loader(dataframe=dataframe,
                                                                                         train_ratio=train_size / full_size,
                                                                                         val_ratio=val_size / full_size,
                                                                                         token_to_tensor=token_to_tensor,
                                                                                         b_size=batch_size)
    print("created dataset and dataloader")

    model_trained = trainer.train(model_gpu, num_epochs, optimizer, loss_fn, train_data_loader, val_data_loader,
                                  test_data_loader, path_model)
    results = trainer.evaluate_model(model_gpu=model_trained, test_data_loader=test_data_loader,
                                     tensor_to_token=tensor_to_token, b_size=batch_size)
    header = ['batch index', 'accuracy in batch']

    with open(f'results/{model_name}_accuracy', 'w', encoding='UTF8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for text in results:
            writer.writerow(text)


if __name__ == "__main__":
    iterations = 5
    for i in range(iterations) :
        print()
        if(iterations > 0) :
            main(iterations=iterations,new_data=False)
        else:
            main(iterations=iterations,new_data=True)
