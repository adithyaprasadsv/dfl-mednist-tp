import flwr as fl
from client import BloodClient

def client_fn(cid):
    return BloodClient(cid)

fl.simulation.run_simulation(
    client_fn=client_fn,
    num_clients=5,
    num_rounds=3,
)
