from pathlib import Path
from kernel_profiles import Resident as OriginalResident


class Resident(OriginalResident):
    def __init__(self,c,library=None):
        super().__init__(c,library or Path(__file__).with_name('libprofiles_block.so'))
        self._serial_forward=self.lib.kp_forward
        self.lib.kp_reverse=self.lib.kp_reverse_block
        self.lib.kp_reverse.restype=self._serial_forward.restype

    def forward(self,loc,oi,record=False):
        # Many candidates already saturate the lane-parallel forward engine.
        self.lib.kp_forward=self.lib.kp_forward_block if len(loc)<=128 else self._serial_forward
        return super().forward(loc,oi,record)
