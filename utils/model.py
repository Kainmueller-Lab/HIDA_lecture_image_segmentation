import numpy as np
import torch

class ConvPass(torch.nn.Module):
    """Convolutional pass of a U-Net."""
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            activation
    ):
        """Constructor of a convolutional pass.
        The pass consists of a series of convolutions, each followed by an activation function.

        Args:
            in_channels:
                Number of input channels
            out_channels:
                Number of output channels
            kernel_sizes:
                List of kernel sizes
            activation:
                String name of the activation function, e.g. 'ReLU'
        """

        super(ConvPass, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)

        layers = []
        for kernel_size in kernel_sizes:
            self.dims = len(kernel_size)
            pad = tuple(np.array(kernel_size) // 2) # We want same padding, i.e. the output size is the same as the input size
            layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                          padding=pad))
            in_channels = out_channels

            if activation is not None:
                layers.append(activation())
        self.conv_pass = torch.nn.Sequential(*layers)


    def forward(self, x):
        """Forward pass of the convolutional pass.

        Args:
            x:
                Input tensor

        Returns:
            Output tensor after applying the convolutional pass

        """
        c = self.conv_pass(x)
        return c
    
def randomize_weights(module, scale=1.0):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight.data = torch.randn_like(m.weight) * scale
            if m.bias is not None:
                m.bias.data = torch.randn_like(m.bias) * scale


class Downsample(torch.nn.Module):
    """Downsampling layer of a U-Net."""

    def __init__(
            self,
            downsample_factor
    ):
        """Constructor of a downsampling layer.

        Args:
            downsample_factor:
                Factor by which to downsample the input tensor.

        """
        super(Downsample, self).__init__()
        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor
        self.down = torch.nn.MaxPool2d(downsample_factor, stride=downsample_factor)


    def forward(self, x):
        """Forward pass of the downsampling layer.

        Args:
            x:
                Input tensor

        Returns:
            Output tensor after applying the downsampling layer
        """
        self.assert_downsample_factor(x)
        d = self.down(x)
        return d


    def assert_downsample_factor(self, x):
        """Assert if the downsample factor is compatible with the input tensor.

        Args:
            x:
                Input tensor

        Returns:
            True if the downsample factor is compatible with the input tensor, raises RuntimeError otherwise
        """
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch in spatial dimension %d" % (
                        x.size(),
                        self.downsample_factor,
                        self.dims - d)
                )
        return True
    


class Upsample(torch.nn.Module):
    """Class that implements an upsampling layer."""

    def __init__(
            self,
            scale_factor,
            mode,
            in_channels,
            out_channels,
    ):
        """Constructor of an upsampling layer.

        Args:
            scale_factor:
                Factor by which to upsample the input tensor
            mode:
                Upsample mode, either 'nearest' or 'transpose_conv'
            in_channels:
                Number of input channels
            out_channels:
                Number of output channels

        """
        super(Upsample, self).__init__()

        self.dims = len(scale_factor)

        if mode == 'transpose_conv':
            self.up = torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=scale_factor,
                stride=scale_factor
            )
        elif mode == 'nearest':
            self.up = torch.nn.Upsample(
                scale_factor=tuple(scale_factor),
                mode=mode
            )
        else:
            raise RuntimeError("invalid mode")


    def center_crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape.

        Args:
            x:
                The input tensor.
            shape:
                The target shape.

        Returns:
            The center-cropped tensor.

        """

        x_target_size = x.size()[:-self.dims] + shape

        offset = tuple(
            (a - b) // 2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):
        """Forward pass.

        Args:
            f_left:
                Input tensor from the left branch
            g_out:
                Input tensor from the previous layer

        Returns:
            Concatenated cropped f_left with upsampled cropped g_out

        """
        g_up = self.up(g_out)

        f_cropped = self.center_crop(f_left, g_up.size()[-self.dims:])

        f_right = torch.cat([f_cropped, g_up], dim=1)
        
        return f_right
    

class UNet(torch.nn.Module):
    """UNet class."""

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factors,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            activation,
            num_fmaps_out,
            constant_upsample
    ):
        """Unet constructor.

        Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 2D tensors shaped like::

            ``(batch=1, channels, height, width)``.

        Args:

            in_channels:
                The number of input channels.
            num_fmaps:
                The number of feature maps in the first layer. This is also the
                number of output feature maps. Stored in the ``channels``
                dimension.
            fmap_inc_factors:
                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.
            downsample_factors:
                List of tuples ``(y, x)`` to use to down- and up-sample the
                feature maps between layers.
            kernel_size_down:
                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer.
            kernel_size_up:
                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer.
            activation:
                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).
            constant_upsample:
                If set to true, perform a constant (i.e. nearest) upsampling instead of a
                transposed convolution in the upsampling layers.

        """
        super(UNet, self).__init__()

        self.num_levels = len(downsample_factors) + 1
        self.in_channels = in_channels
        self.out_channels = num_fmaps_out if num_fmaps_out else num_fmaps
        self.mode = 'nearest' if constant_upsample else 'transpose_conv'
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up = kernel_size_up
        self.downsample_factors = downsample_factors

        # left convolutional passes
        self.l_conv = torch.nn.ModuleList([
            ConvPass(
                in_channels
                if level == 0
                else num_fmaps * fmap_inc_factors ** (level - 1),
                num_fmaps * fmap_inc_factors ** level,
                kernel_size_down[level],
                activation=activation)
            for level in range(self.num_levels)
        ])
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = torch.nn.ModuleList([
            Downsample(downsample_factors[level])
            for level in range(self.num_levels - 1)
        ])

        # right up/crop/concatenate layers
        self.r_up = torch.nn.ModuleList([
            Upsample(
                downsample_factors[level],
                mode=self.mode,
                in_channels=num_fmaps * fmap_inc_factors ** (level + 1),
                out_channels=num_fmaps * fmap_inc_factors ** (level + 1),
            )
            for level in range(self.num_levels - 1)
        ])

        # right convolutional passes
        self.r_conv = torch.nn.ModuleList([
            ConvPass(
                num_fmaps * fmap_inc_factors ** level +
                num_fmaps * fmap_inc_factors ** (level + 1),
                num_fmaps * fmap_inc_factors ** level
                if num_fmaps_out is None or level != 0
                else num_fmaps_out,
                kernel_size_up[level],
                activation=activation)
            for level in range(self.num_levels - 1)
        ])
        self.weights_init()

    def weights_init(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)

        if isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.01)


    def _rec_forward(self, level, f_in):
        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:
            fs_out = f_left
        else:
            # down
            g_in = self.l_down[i](f_left)
            # nested levels
            gs_out = self._rec_forward(level - 1, g_in)
            # up, concat, and crop
            fs_right = self.r_up[i](f_left, gs_out)

            # convolve
            fs_out = self.r_conv[i](fs_right)

        return fs_out
    

    def forward(self, x):
        """Forward pass through the network."

        Args:
            x:
                Input tensor of shape (b, c, y, x).

        Returns:
            The feature.

        """

        y = self._rec_forward(self.num_levels - 1, x)
        return y
    

# class TestDataset(Dataset):
#     """(Subset of the) Kaggle Data Science Bowl 2018 dataset."""

#     def __init__(
#         self,
#         root_dir,
#         mode,
#         prediction_type,
#         padding_size=None
#     ):
#         self.mode = mode
#         self.files = glob.glob(os.path.join(root_dir, mode, "*.zarr"))
#         self.prediction_type = prediction_type
#         self.padding_size = padding_size
#         self.define_augmentation()
#         self.define_padding()

#     def __len__(self):
#         return len(self.files)

#     def define_augmentation(self):
#         """Define the augmentation pipeline using Albumentations."""
#         if self.mode == "train":
#             self.aug_transform = A.Compose(augmentation_list())
#         else:
#             self.aug_transform = A.Compose([])

#     def define_padding(self):
#         """Define the initial padding of the images."""
#         if self.padding_size is not None:
#             pad = self.padding_size
#             def pad_image(image, **kwargs):
#                 return np.pad(
#                     image,
#                     ((pad, pad), (pad, pad), (0, 0)),
#                     mode='constant',
#                     constant_values=0
#                 )

#             self.pad_transform = A.Compose([
#                 A.Lambda(image=pad_image, mask=pad_image)
#             ])
#         else:
#             self.pad_transform = A.Compose([])

#     def get_filename(self, idx):
#         """Get the filename of the idx-th sample."""
#         return self.files[idx]

#     def __getitem__(self, idx):
#         fn = self.get_filename(idx)
#         raw, label = self.load_sample(fn)
#         raw = self.normalize(raw)

#         # Padding
#         if self.padding_size is not None:
#             raw = np.transpose(raw, [1, 2, 0])  # CHW -> HWC
#             label = np.transpose(label, [1, 2, 0])  # CHW -> HWC
#             padded = self.pad_transform(image=raw, mask=label)
#             raw = padded["image"]
#             label = padded["mask"]
#             raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
#             label = np.transpose(label, [2, 0, 1])  # HWC -> CHW

#         # Augmentation (only during training)
#         if self.mode == "train":
#             raw = np.transpose(raw, [1, 2, 0])  # CHW -> HWC
#             label = np.transpose(label, [1, 2, 0])  # CHW -> HWC
#             raw, label = self.augment_sample(raw, label)
#             raw = np.transpose(raw, [2, 0, 1])  # HWC -> CHW
#             label = np.transpose(label, [2, 0, 1])  # HWC -> CHW

#         raw, label = torch.tensor(raw.copy(), dtype=torch.float32), torch.tensor(label.copy(), dtype=torch.float32)
#         return raw, label

#     def augment_sample(self, raw, label):
#         """Apply Albumentations augmentations."""
#         augmented = self.aug_transform(image=raw, mask=label)
#         raw_aug = augmented["image"]
#         label_aug = augmented["mask"]
#         return raw_aug, label_aug

#     @staticmethod
#     def normalize(raw):
#         """Normalize the raw image to zero mean and unit variance."""
#         raw -= np.mean(raw)
#         raw /= np.std(raw)
#         return raw

#     def load_sample(self, filename):
#         """Load a sample from a Zarr file."""
#         data = zarr.open(filename)
#         raw = np.array(data['volumes/raw'])

#         if self.prediction_type == PredictionType.TWO_CLASS:
#             label = np.array(data['volumes/gt_fgbg'])
#         elif self.prediction_type == PredictionType.THREE_CLASS:
#             label = np.array(data['volumes/gt_threeclass'])
#         else:
#             raise NotImplementedError

#         label = label.astype(np.float32)
#         return raw, label
