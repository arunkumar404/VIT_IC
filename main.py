import torch.nn as nn
import torch.optim as optim
from models.vision_transformer import VisionTransformer
from data.dataset import get_data_loaders
from utils.training import train, evaluate

def main():
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    model = VisionTransformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
