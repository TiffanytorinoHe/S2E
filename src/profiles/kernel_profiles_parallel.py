"""Cooperative CUDA blocks for both full candidate batches and selected traces."""
from kernel_profiles_block import Resident as SelectedResident
from kernel_profiles import Resident as BaseResident

class Resident(SelectedResident):
    def forward(self,loc,oi,record=False):
        self.lib.kp_forward=self.lib.kp_forward_block
        # Skip SelectedResident's small-batch dispatch, retain base validation.
        return BaseResident.forward(self,loc,oi,record)
