from torch.utils.data import Dataset
import torch
import os
import pickle


import sys
sys.path.append("..")

from pdp.state_pdp import StatePDP
from utils.beam_search import beam_search
import random

class PDP(object):
    NAME = 'pdp'  # APDP


    @staticmethod
    def get_costs(dataset, pi):
        assert (pi[:, 0] == 0).all(), "not starting at depot"
        assert (torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) == pi.data.sort(1)[
            0]).all(), "not visiting all nodes"

        visited_time = torch.argsort(pi, 1)

        # Determine whether the index of the pick point is smaller than the index of its corresponding delivery point.
        assert (visited_time[:, 1:pi.size(1) // 2 + 1] < visited_time[:,
                                                         pi.size(1) // 2 + 1:]).all(), "deliverying without pick-up"

        batch_size = pi.shape[0]
        graph_size = pi.shape[1]
        edge_adjacency_matrixs = dataset['matrix']
        pi2 = torch.zeros(batch_size, graph_size)
        pi2[:, :-1] = pi[:, 1:]
        pi2[:, -1] = pi[:, 0]
        pi = pi.type(torch.long)
        pi2 = pi2.type(torch.long)
        costs = torch.zeros(batch_size)

        for i in range(batch_size):
            costs[i] = edge_adjacency_matrixs[i, pi[i, :], pi2[i, :]].sum()
        return costs, None


    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDPDataset(*args, **kwargs)


    @staticmethod
    def make_state(*args, **kwargs):
        return StatePDP.initialize(*args, **kwargs)


    @staticmethod
    def beam_search(input, beam_size, expand_size=None, compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        state = PDP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        def propose_expansions(beam):
            return model.propose_expansions(beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size)
        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    loc, depot, index, matrix, *args = args

    data = {
        'loc': torch.tensor(loc, dtype=torch.float32),
        'depot': torch.tensor(depot, dtype=torch.float32),
        'index': torch.tensor(index, dtype=torch.int64),
        'matrix': torch.tensor(matrix, dtype=torch.float64)
    }
    return data


'''
APDP dataset, generating APDP problem instance
'''
class PDPDataset(Dataset):

    def __init__(self, size=50, num_samples=1000000, filename=None, offset=0):
        super(PDPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        else:
            # Load data
            coordinates = torch.load("data/coordinates_500.pt")
            edge_adjacency_matrix = torch.load("data/distance_matrix_500.pt")

            self.data = []
            for i in range(num_samples):
                # Select the node including the depot
                index = torch.LongTensor(random.sample(range(0, 500), size+1))
                # Command index 0 is a depot.
                depot_index = index[0]
                loc_index = index[1:]

                self.data.append(
                    {
                        'loc': torch.index_select(coordinates, 0, loc_index),
                        'depot': torch.index_select(coordinates, 0, depot_index),
                        'index': index,
                        'matrix': edge_adjacency_matrix[index, :][:, index]
                    }
                )
        self.size = len(self.data)


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

