
import torch


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training function
def train(model,loss_fn, optimizer, train_loader, test_loader, epochs=100, autoencoder=False, BCE=False):
    num_epochs = epochs 
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, targets in train_loader:

            optimizer.zero_grad()
            if autoencoder:
              out = model(data.to(device).view(-1,108).float())
            else:
              out = model(data.to(device))

            if autoencoder:
                if BCE:
                  loss = loss_fn(out.float(), data.to(device).float())
                else:
                  loss = loss_fn(out, data.to(device).view(-1,108).float())
                epoch_loss += loss.item()
            else:
                if BCE:
                  loss = loss_fn(out.float(), targets.to(device).view(-1,1).float())
                else:
                  loss = loss_fn(out, targets.to(device).long())

            loss.backward()
            optimizer.step()

        # Give status reports every 10 epochs
        if autoencoder:
            if epoch % 10 == 0:
                print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
                print(f" Loss: {epoch_loss/len(train_loader)}")
        else:
            if epoch % 10 == 0:
                print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
                print(f" Train accuracy: {evaluate(model,train_loader,BCE)}. Test accuracy: {evaluate(model,test_loader,BCE)}")

# Define evaluation function
def evaluate(model, evaluation_set, BCE=False):

    with torch.no_grad():
        correct = total = 0
        for data, targets in evaluation_set:
            out = model(data.to(device))
            if BCE:
              predictions = torch.flatten(torch.round(out))
            else:
              predictions = torch.argmax(out, 1)

            correctly_predicted = (predictions == targets.to(device)).sum()
            correct += correctly_predicted
            total += targets.to(device).size(0)
        
        accuracy = correct/total * 100

    return accuracy