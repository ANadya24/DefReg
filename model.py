import voxelmorph2d as vm2d
import voxelmorph3d as vm3d
import torch
from torch import nn


class VoxelMorph:
    """
    VoxelMorph Class is a higher level interface for both 2D and 3D
    Voxelmorph classes. It makes training easier and is scalable.
    """

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def __init__(self, input_dims, load_epoch=0, model_path=None, is_2d=False, use_gpu=False):
        self.dims = input_dims

        if is_2d:
            self.vm = vm2d
            self.voxelmorph = vm2d.VoxelMorph2d(input_dims[0] * 2, use_gpu)
        else:
            self.vm = vm3d
            self.voxelmorph = vm3d.VoxelMorph3d(input_dims[0] * 2, use_gpu)
            
        if load_epoch:
            self.voxelmorph.load_state_dict(model_path, f'vm_{load_epoch}')
        # else:
        #     self.voxelmorph.apply(self.weights_init)


        if use_gpu:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.voxelmorph = nn.DataParallel(self.voxelmorph)

            self.voxelmorph.to('cuda')

    def check_dims(self, x):
        try:
            if x.shape[1:] == self.dims:
                return
            else:
                raise TypeError
        except TypeError:
            print("Invalid Dimension Error. The supposed dimension is ",
                  self.dims, "But the dimension of the input is ", x.shape[1:])

    def forward(self, x):
        self.check_dims(x)
        return self.voxelmorph(x)

    def save_state_dict(self, save_dir, fname):
        torch.save(self.voxelmorph.state_dict(), save_dir + fname)
        print(f"Successfuly saved state_dict in {save_dir + fname}")

    def load_state_dict(self, save_dir, fname):
        self.voxelmorph.load_state_dict(save_dir + fname)
        print(f"Successfuly loaded state_dict from {save_dir + fname}")

