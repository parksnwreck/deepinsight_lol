import torch
import phate

from data import MatchDataSet, MatchDataSetCleanedN1, MatchDataSetCleanedN2, MatchDataImageSetCleanedN1, MatchDataImageSetCleanedN2
from models import LogisticRegression, FeedForwardNetwork, MatchNeuralNet
from train import train, evaluate, device

# Initialize Torch datasets
datasets = [
    MatchDataSet('raw_data/MatchTimelinesFirst15.csv'),
    MatchDataSetCleanedN1('raw_data/MatchTimelinesFirst15.csv'),
    MatchDataSetCleanedN2('raw_data/MatchTimelinesFirst15.csv')
]

dataset_names = [
    'data_raw',
    'data_n1',
    'data_n2'
]

deepinsight_datasets = [
    MatchDataImageSetCleanedN1('raw_data/MatchTimelinesFirst15.csv'), 
    MatchDataImageSetCleanedN2('raw_data/MatchTimelinesFirst15.csv'), 
    MatchDataImageSetCleanedN2('raw_data/MatchTimelinesFirst15.csv', fe='pca'),
    MatchDataImageSetCleanedN2('raw_data/MatchTimelinesFirst15.csv', fe='kpca'),
    MatchDataImageSetCleanedN2('raw_data/MatchTimelinesFirst15.csv', fe=phate.PHATE(knn=7))
]

deepinsight_dataset_names = [
    'deepinsight_n1_tsne',
    'deepinsight_n2_tsne',
    'deepinsight_n2_pca',
    'deepinsight_n2_kpca',
    'deepinsight_n2_phate'
]

# Train LR models and feedfoward networks on non-image datasets
for dataset, dataset_name in zip(datasets, dataset_names):

    # Split dataset into training and test data
    train_dataset_len = round(len(dataset) * 0.8)
    test_dataset_len = len(dataset) - train_dataset_len
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_dataset_len, test_dataset_len))

    # Initialize loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    models = [
        LogisticRegression(),
        FeedForwardNetwork()
    ]

    model_names = [
        'LR',
        'FFN'
    ]

    for model, model_name in zip(models, model_names):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.CrossEntropyLoss()

        model.to(device)
        train(model, loss_fn, optimizer, train_loader, test_loader, epochs=300)

        # Print training and test set accuracies
        evaluate(model, train_loader)
        evaluate(model, test_loader)

        # Save model
        torch.save(model.state_dict(), model_name + '_' + dataset_name)

# Train CNNs on image datasets created using DeepInsight
for dataset, dataset_name in zip(deepinsight_datasets, deepinsight_dataset_names):

    # Split dataset into training and test data
    train_dataset_len = round(len(dataset) * 0.8)
    test_dataset_len = len(dataset) - train_dataset_len
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_dataset_len, test_dataset_len))

    # Initialize loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    models = [
        MatchNeuralNet(),
    ]

    model_names = [
        'CNN'
    ]

    for model, model_name in zip(models, model_names):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_fn = torch.nn.NLLLoss()

        model.to(device)
        model.double()
        train(model, loss_fn, optimizer, train_loader, test_loader, epochs=300)

        # Print training and test set accuracies
        evaluate(model, train_loader)
        evaluate(model, test_loader)

        # Save model
        torch.save(model.state_dict(), model_name + '_' + dataset_name)
