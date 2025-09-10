import numpy as np
from typing import Tuple

# TODO: 
# - understand He init
# - use im2col? 


# ---- HELPERS ----
def he_init(shape):
    """He initialization for weights."""
    fan_in = np.prod(shape[1:]) if len(shape) > 1 else shape[0]
    return np.random.randn(*shape) * np.sqrt(2. / fan_in)

def softmax_logits(logits):
    """Compute the softmax probabilities from logits.

    Args:
        logits: _np.ndarray of shape (N, C) where N is the number of samples and C is the number of classes.

    Returns:
        _type_: _np.ndarray of shape (N, C) representing the softmax probabilities.
    """
    z = logits - logits.max(axis=1, keepdims=True) # for numerical stability
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_from_logits(logits, y_idx):
    """Compute the cross-entropy loss from logits and true labels.

    Args:
        logits (np.ndarray): Logits of shape (N, C) where N is the number of samples and C is the number of classes.
        y_idx (np.ndarray): True labels of shape (N,) with integer class indices.

    Returns:
        float: The average cross-entropy loss.
    """
    N = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(z).sum(axis=1, keepdims=True))
    log_probs = z - log_sum_exp
    loss = -log_probs[np.arange(N), y_idx].mean()

    # Gradient
    probs = np.exp(log_probs)
    dlogits = probs
    dlogits[np.arange(N), y_idx] -= 1
    dlogits /= N
    return loss, dlogits


def accuracy_from_logits(logits, y_idx):
    """Compute the accuracy from logits and true labels.

    Args:
        logits (np.ndarray): Logits of shape (N, C) where N is the number of samples and C is the number of classes.
        y_idx (np.ndarray): True labels of shape (N,) with integer class indices.

    Returns:
        float: The accuracy as a fraction between 0 and 1.
    """
    y_pred = np.argmax(logits, axis=1)
    return (y_pred == y_idx).mean()

# ---- LAYERS ----
class Conv2D:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=3,
                 stride: int=1,
                 padding: int=0,
                 bias: bool=True):
        """Initialize the Conv2D layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding added to both sides of the input. Defaults to 0.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.

        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_width = self.kernel_height = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.W = he_init((out_channels, in_channels, self.kernel_height, self.kernel_width))
        self.b = np.zeros((out_channels, 1)) if bias else None

        # Grad
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

        # Cache
        self.cache = None


    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the Conv2D layer.

        Args:
            x (np.ndarray): Input data of shape (N, C_in, H_in, W_in).

        Returns:
            np.ndarray: Output data of shape (N, C_out, H_out, W_out).

        """
        self.x_shape = x.shape
       
        cols, out_h, out_w = 
        # Initialize output
        out = np.zeros((N, self.out_channels, H_out, W_out))

        # Pad input
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        # Perform convolution
        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_height
                        w_end = w_start + self.kernel_width

                        out[n, c_out, h, w] = np.sum(
                            x_padded[n, :, h_start:h_end, w_start:w_end] * self.W[c_out]
                        )
                        if self.b is not None:
                            out[n, c_out, h, w] += self.b[c_out]

        # Cache values for backward pass
        self.cache = (x, x_padded)

        return out
