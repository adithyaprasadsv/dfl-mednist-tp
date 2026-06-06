from torch.utils.data import DataLoader, random_split
from medmnist import BloodMNIST

def load_datasets(num_clients=5, batch_size=64):
    train = BloodMNIST(split="train", download=True)
    test = BloodMNIST(split="test", download=True)

    length = len(train) // num_clients
    splits = [length] * num_clients
    splits[-1] = len(train) - sum(splits)

    client_datasets = random_split(train, splits)

    def get_loader(cid):
        return DataLoader(client_datasets[int(cid)], batch_size=batch_size, shuffle=True)

    return get_loader, DataLoader(test, batch_size=batch_size)
