import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# Initialize a list to store your processed data
processed_data = []


def load_csv(path):
    return pd.read_csv(path)


def hex_to_binary(hex_string):
    decimal_value = int(hex_string, 16)

    binary_string = bin(decimal_value)[2:]

    binary_string = binary_string.zfill(8)
    return binary_string


def create_tensor_dict(start_token,end_token):
    token_to_tensor = {}
    for i in range(256):
        hex_token = format(i, '02x')  # Convert the integer to a 2-character lowercase hex string
        binary_token = format(i, '08b')  # Convert the integer to an 8-character binary string
        binary_tensor = torch.tensor(
            [0.] + [float(bit) for bit in binary_token])  # Create a tensor with float values (0.0 or 1.0)
        token_to_tensor[hex_token] = binary_tensor


    tensor_to_token = {tuple(binary_tensor.tolist()): hex_to_binary(hex_token) for hex_token, binary_tensor in
                       token_to_tensor.items()}
    token_to_tensor[start_token] = torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.,0.])
    token_to_tensor[end_token] = torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,1.])

    tensor_to_token[[1.,1.,1.,1.,1.,1.,1.,1.,1.]] =  end_token
    tensor_to_token[[1.,0.,0.,0.,0.,0.,0.,0.,0.]] =  start_token
    return token_to_tensor, tensor_to_token


def encode_sequence(sequence, token_to_embedding):
    # Split the sequence into tokens
    tokens = sequence.split(",")

    """
    if is_encoder:
        embeddings = token_to_embedding['<SOSE>']
    else:
        embeddings = token_to_embedding['<SOSD>']
    """
    # Map tokens to embeddings
    embeddings = [token_to_embedding[token] for token in tokens]

    # Stack the embeddings to form the input tensor
    input_tensor = torch.stack(embeddings)

    return input_tensor


def create_data_loader(dataframe, train_ratio, val_ratio, token_to_tensor, b_size):
    X = dataframe[['block_and_nonce']]  # inputs
    y = dataframe[['next_block']]  # targets

    if train_ratio + val_ratio > 1:
        print("falsche Angaben der Verhältnisse")
        return
    # calculating index for training, validation and testing
    train_index = int(train_ratio * len(dataframe))
    val_index = int((train_ratio + val_ratio) * len(dataframe))

    # Split the dataset into training and validation sets
    X_train, y_train = X[:train_index], y[:train_index]
    X_val, y_val = X[train_index:val_index], y[train_index:val_index]
    X_test, y_test = X[val_index:], y[val_index:]

    train_input_embeddings = []
    train_target_embeddings = []

    for index, row in X_train.iterrows():
        input_sequence = row['block_and_nonce']
        target_sequence = y_train.loc[index, 'next_block']

        input_embedding = encode_sequence(input_sequence, token_to_tensor, True)
        target_embedding = encode_sequence( target_sequence, token_to_tensor, False)

        train_input_embeddings.append(input_embedding)
        train_target_embeddings.append(target_embedding)

    train_input_embeddings = torch.stack(train_input_embeddings)
    train_target_embeddings = torch.stack(train_target_embeddings)

    val_input_embeddings = []
    val_target_embeddings = []

    for index, row in X_val.iterrows():
        input_sequence = row['block_and_nonce']
        target_sequence = y_val.loc[index, 'next_block']

        input_embedding = encode_sequence( input_sequence, token_to_tensor, True)
        target_embedding = encode_sequence(target_sequence, token_to_tensor, False)

        val_input_embeddings.append(input_embedding)
        val_target_embeddings.append(target_embedding)

    val_input_embeddings = torch.stack(val_input_embeddings)
    val_target_embeddings = torch.stack(val_target_embeddings)

    test_input_embeddings = []
    test_target_embeddings = []

    for index, row in X_test.iterrows():
        input_sequence = row['block_and_nonce']
        target_sequence = y_test.loc[index, 'next_block']

        input_embedding = encode_sequence(  input_sequence, token_to_tensor, True)
        target_embedding = encode_sequence(  target_sequence, token_to_tensor, False)

        test_input_embeddings.append(input_embedding)
        test_target_embeddings.append(target_embedding)

    test_input_embeddings = torch.stack(test_input_embeddings)
    test_target_embeddings = torch.stack(test_target_embeddings)

    input_data = train_input_embeddings
    target_data = train_target_embeddings

    # Define batch size
    # Adjust as needed

    # Create TensorDataset
    train_dataset = TensorDataset(input_data, target_data)

    # Create DataLoader for training data
    train_data_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True, drop_last=True)

    input_data = val_input_embeddings
    target_data = val_target_embeddings

    # Create TensorDataset
    val_dataset = TensorDataset(input_data, target_data)

    # Create DataLoader for validation data
    val_data_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=True, drop_last=True)

    input_data = test_input_embeddings
    target_data = test_target_embeddings

    # Create TensorDataset
    test_dataset = TensorDataset(input_data, target_data)

    # Create DataLoader for validation data
    test_data_loader = DataLoader(test_dataset, batch_size=b_size, shuffle=True, drop_last=True)

    return train_data_loader, val_data_loader, test_data_loader
