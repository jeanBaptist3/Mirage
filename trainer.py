import torch
import time


def train(model_gpu, num_epochs, optimizer, loss_fn, train_data_loader, val_data_loader, test_data_loader, path):
    start_time = time.time()
    for epoch in range(num_epochs):
        model_gpu.train()  # training mode
        for input_batch, target_batch in train_data_loader:  # Iterate over mini-batches
            optimizer.zero_grad()  # Clear gradients
            # move both batches on the gpu
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()
            predictions = model_gpu(input_batch, target_batch)
            loss = loss_fn(predictions, target_batch)
            loss.backward()  # Backpropagation
            optimizer.step()  # Parameter updates
        print("step")

        model_gpu.eval()  # evaluation mode
        with torch.no_grad():
            for input_batch, target_batch in val_data_loader:
                # move both batches on the gpu
                input_batch = input_batch.cuda()
                target_batch = target_batch.cuda()
                predictions = model_gpu(input_batch, target_batch)
                val_loss = loss_fn(predictions, target_batch)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()} in %s seconds' % (
                        time.time() - start_time))


    torch.save(model_gpu, path)
    print("saved model")
    return model_gpu

"""
This method is the metric used for evaluating the bitwise accuracy of the model
"""
def string_similarity(s1, s2):
    same_characters = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
    similarity_percentage = (same_characters / len(s1)) * 100.0
    return similarity_percentage


def decode_sequence(sequence_tensor, embedding_to_token):
    dec_sequ = ""
    for i in range(sequence_tensor.size(0)):
        dec_sequ = dec_sequ + embedding_to_token[tuple(sequence_tensor[i].tolist())]
    return dec_sequ


def evaluate_model(model_gpu, test_data_loader, tensor_to_token, b_size):
    #evaluation mode
    model_gpu.eval()
    """
    This method is for calculating the bitwise accuracy for each batch, and also average them
    """
    batch_idx = 1
    evaluation_dataset = []
    overall_accuracy = 0.
    for input_batch, target_batch in test_data_loader:

        bits_per_batch = 0.
        input_data = input_batch.cuda()
        target_data = torch.zeros_like(input_data[:, :64, :])
        with torch.no_grad():  # Disable gradient tracking for inference
            predictions = torch.round(model_gpu(input_data, target_data))

        for i in range(b_size):
            target_dec = decode_sequence(target_batch[i], tensor_to_token)
            pred_dec = decode_sequence(predictions[i], tensor_to_token)
            bits_per_batch = bits_per_batch + string_similarity(target_dec, pred_dec)
        accuracy_per_batch = bits_per_batch / b_size
        overall_accuracy = overall_accuracy + accuracy_per_batch
        batch_idx = batch_idx + 1
        evaluation_dataset.append((batch_idx, accuracy_per_batch))

    evaluation_dataset.append(("all", overall_accuracy / (batch_idx - 1)))

    return evaluation_dataset
