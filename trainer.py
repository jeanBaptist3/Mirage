import torch

def train(model_gpu, num_epochs,optimizer, loss_fn, train_data_loader, val_data_loader, test_data_loader,path) :
    for epoch in range(num_epochs):
        model_gpu.train()  # Set the model to training mode
        for input_batch, target_batch in train_data_loader:  # Iterate over mini-batches
            optimizer.zero_grad()  # Clear gradients
            # move botch batches to GPU
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()
            model_gpu = model_gpu.cuda()
            predictions = model_gpu(input_batch, target_batch)
            loss = loss_fn(predictions, target_batch)  # BCELoss for binary sequences
            loss.backward()  # Backpropagation
            optimizer.step()  # Parameter updates
        print("step")

        model_gpu.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for input_batch, target_batch in val_data_loader:
                # Ensure that both input and target sequences are passed to the model
                input_batch = input_batch.cuda()
                target_batch = target_batch.cuda()
                predictions = model_gpu(input_batch, target_batch)
                val_loss = loss_fn(predictions, target_batch)  # BCELoss for binary sequences

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    model_gpu.eval()
    with torch.no_grad():
        for input_batch, target_batch in test_data_loader:
            # Ensure that both input and target sequences are passed to the model
            input_batch = input_batch.cuda()
            target_batch = target_batch.cuda()
            predictions = model_gpu(input_batch, target_batch)
            # Evaluate the model's performance on the test dataset

    torch.save(model_gpu, path)