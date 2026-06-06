import torch
from collections import OrderedDict
from flwr.client import NumPyClient
from model import BloodNet
from dataset import load_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_loader, _ = load_datasets()

class BloodClient(NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = BloodNet().to(DEVICE)
        self.train_loader = get_loader(cid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, params):
        state_dict = OrderedDict(
            (k, torch.tensor(v).to(DEVICE))
            for (k, v), v in zip(self.model.state_dict().items(), params)
        )
        self.model.load_state_dict(state_dict)

    def fit(self, params, config):
        self.set_parameters(params)
        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, params, config):
        return 0.0, 0, {}
