import torch
from typing import NamedTuple

import sys
sys.path.append("..")

from utils.boolmask import mask_long2bool, mask_long_scatter


class StatePDP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    to_delivery: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:

            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))


    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)


    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
                to_delivery=self.to_devliery[key],
            )
        return super(StatePDP, self).__getitem__(key)


    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        batch_size, n_loc= input['index'].size()

        depot_index = input['index'][:,0].reshape([batch_size, 1])
        depot_index = depot_index.to(torch.float32)
        depot_index = depot_index.cuda()
        depot_index = depot_index.to(input['index'].device)

        to_delivery=torch.cat([torch.ones(batch_size, 1, n_loc // 2 + 1, dtype=torch.uint8, device=input['index'].device),
            torch.zeros(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=input['index'].device)], dim=-1)

        return StatePDP(
            coords=input['index'],
            ids=torch.arange(batch_size, dtype=torch.int64, device=input['index'].device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=input['index'].device),
            visited_=(
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=input['index'].device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=input['index'].device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=input['index'].device),
            cur_coord=depot_index,
            i=torch.zeros(1, dtype=torch.int64, device=input['index'].device),  # Vector with length num_steps
            to_delivery=to_delivery,
        )



    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        n_loc = self.to_delivery.size(-1) - 1

        new_to_delivery = (selected + n_loc // 2) % (n_loc + 1)  # the pair node of selected node
        new_to_delivery = new_to_delivery[:, None]
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected].to(torch.float32)
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # The `self.lengths` actually doesn't serve much purpose

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            to_delivery = self.to_delivery.scatter(-1, new_to_delivery[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, to_delivery=to_delivery,
        )

    def all_finished(self):
        return self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a


    '''
    Gets a mask with the feasible actions 
    '''
    def get_mask(self):
        visited_loc = self.visited_

        n_loc = visited_loc.size(-1) - 1  # num of customers
        batch_size = visited_loc.size(0)

        mask_loc = visited_loc.to(self.to_delivery.device) | (1 - self.to_delivery)

        # Cannot visit the depot if just visited and still unserved nodes
        if self.i == 0:
            return torch.cat([torch.zeros(batch_size, 1, 1, dtype=torch.uint8, device=mask_loc.device),
                              torch.ones(batch_size, 1, n_loc, dtype=torch.uint8, device=mask_loc.device)], dim=-1) > 0

        return mask_loc > 0  # return true/false

#
    def construct_solutions(self, actions):
        return actions
