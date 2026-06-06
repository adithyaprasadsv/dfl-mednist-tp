from torch.utils.data import DataLoader, random_split
from medmnist import BloodMNIST

def load_datasets(num_clients=5, batch_size=64):
    train = BloodMNIST(split="train", download=True)
    test = BloodMNIST(split="test", download=True)

    total = len(train)
    base = total // num_clients
    remainder = total % num_clients

    # Distribute remainder across first few clients
    splits = [base + (1 if i < remainder else 0) for i in range(num_clients)]

    client_datasets = random_split(train, splits)

    def get_loader(cid):
        return DataLoader(client_datasets[int(cid)], batch_size=batch_size, shuffle=True)

    return get_loader, DataLoader(test, batch_size=batch_size)

