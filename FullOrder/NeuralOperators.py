import torch
import torch.nn as nn
import torch.fft
import math
# -------------------------
# DCT/IDCT (DCT-II / DCT-III) 1D and separable 2D implementations, this aims to use fft in chebyshev expansion
# thus we applied the 1st-type Chebyshev polynomials, Tk(x) = cos(k arccos x),
# and it can be writtin in the Discrete Cosine Transform(DCT).
# -------------------------
def dct_1d(x):
    # x: (..., N), real
    N = x.shape[-1]
    v = torch.cat([x, x.flip(-1)], dim=-1)  # (..., 2N)s
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    exp_factor = torch.exp(-1j * math.pi * k / (2 * N))
    X = (V[..., :N] * exp_factor).real
    X[..., 0] *= 0.5
    return X


def idct_1d(X):
    # inverse of dct_1d (DCT-III), X: (..., N)
    N = X.shape[-1]
    c = X.clone()
    c[..., 0] = c[..., 0] * 2.0
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    exp_factor = torch.exp(1j * math.pi * k / (2 * N))
    V = torch.zeros(X.shape[:-1] + (2 * N,), dtype=torch.cfloat, device=X.device)
    V[..., :N] = (c * exp_factor)
    if N > 1:
        V[..., N + 1:] = torch.conj(V[..., 1:N].flip(-1))
    V[..., N] = torch.tensor(0.0 + 0.0j)
    v = torch.fft.ifft(V, dim=-1)
    x = v[..., :N].real
    return x


def dct_2d(x):
    # x: (..., H, W)
    # apply dct along last dim then along -2
    orig_shape = x.shape
    # last dim
    x_resh = x.reshape(-1, orig_shape[-1])
    y = dct_1d(x_resh).reshape(*orig_shape)
    # swap last two and apply again
    y_perm = y.permute(*range(y.dim() - 2), y.dim() - 1, y.dim() - 2)
    shp = y_perm.shape
    y2 = dct_1d(y_perm.reshape(-1, shp[-1])).reshape(shp)
    return y2.permute(*range(y2.dim() - 2), y2.dim() - 1, y2.dim() - 2)


def idct_2d(X):
    # inverse 2D: apply idct along -2 then -1 (reverse order)
    X_perm = X.permute(*range(X.dim() - 2), X.dim() - 1, X.dim() - 2)
    shp = X_perm.shape
    y = idct_1d(X_perm.reshape(-1, shp[-1])).reshape(shp)
    y = y.permute(*range(y.dim() - 2), y.dim() - 1, y.dim() - 2)
    z = idct_1d(y.reshape(-1, y.shape[-1])).reshape(y.shape)
    return z


# -------------------------
# Chebyshev / Cosine spectral conv (real coefficients)
# -------------------------
class ChebSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_h, modes_w):
        """
        in_channels, out_channels: channels
        modes_h, modes_w: number of retained modes in each dim (use <= H, W)
        The weight shape: (in_channels, out_channels, modes_h, modes_w) real
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m_h = modes_h
        self.m_w = modes_w
        # real coefficients for Chebyshev/DCT space
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_h, modes_w) * (1.0 / (in_channels * out_channels) ** 0.5))

    def forward(self, x):
        # x: [B, C, H, W] real
        B, C, H, W = x.shape
        # compute DCT2 on each channel
        # reshape to (..., H, W) to operate with dct_2d
        x_dct = dct_2d(x)  # shape [B, C, H, W]
        # crop modes (take top-left modes_h x modes_w)
        x_modes = x_dct[:, :, :self.m_h, :self.m_w]  # [B, C, m_h, m_w]
        # multiply by real weights: einsum over in_channel
        # out_modes[b, o, i, j] = sum_c x_modes[b, c, i, j] * weight[c, o, i, j]
        out_modes = torch.einsum("b c i j, c o i j -> b o i j", x_modes, self.weight)
        # create full spectral tensor with zeros then place modes back
        out_dct = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        out_dct[:, :, :self.m_h, :self.m_w] = out_modes
        # inverse DCT2
        out = idct_2d(out_dct)
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        in_channels, out_channels: number of channels
        modes: modes that reserved, Assume that H, W >= modes!!!!!
        weights are in complex，symmetric can be recovered by conjugate mirror
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # einsum over in_channel
        # input: [B, in, H, W], weights: [in, out, mh, mw]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        x: [B, C, H, W] (实数)
        """
        B, C, H, W = x.shape
        # 2D FFT (use complex)
        x_ft = torch.fft.rfft2(x, norm="forward")  # [B, C, H, W//2+1]

        # Output a frequency tensor
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        # Low frequency modes × modes
        mh, mw = self.modes, self.modes
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(x_ft[:, :, :mh, :mw], self.weights)

        # IFFT
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="forward")
        return x_out


# -------------------------
# CFNO block: combine Fourier spectral conv and Chebyshev spectral conv per layer
# -------------------------
class CFNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, cheb_modes, alpha_init=0.5):
        # alpha\in[0,1], 0.5 is the default for initialization and self-adaptive fitting
        super().__init__()
        self.fourier = SpectralConv2d(in_channels, out_channels, modes)
        mh, mw = cheb_modes
        self.cheb = ChebSpectralConv2d(in_channels, out_channels, mh, mw)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y_f = self.fourier(x)
        y_c = self.cheb(x)
        a = torch.sigmoid(self.alpha)
        y_blend = a * y_f + (1.0 - a) * y_c
        y_cat = torch.cat([y_f, y_c], dim=1)
        y_fused = self.fuse(y_cat)
        return y_blend + y_fused


# -------------------------
# CFNO network (example stack)
# -------------------------
class CFNO2d(nn.Module):
    def __init__(self, modes=12, cheb_modes=(12, 12), width=32, depth=4):
        super().__init__()
        self.width = width
        self.depth = depth
        # input lifting (like your FNO fc0)
        self.fc0 = nn.Linear(2, width)
        # create layer stacks of CFNOBlock with 1x1 conv residuals (similar to FNO architecture)
        self.blocks = nn.ModuleList()
        self.w_convs = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(CFNOBlock(width, width, modes, cheb_modes))
            self.w_convs.append(nn.Conv2d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x: [B, 2, H, W]
        B, C, H, W = x.shape
        # lift
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 2]
        x = self.fc0(x)  # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]
        # stack
        for block, w_conv in zip(self.blocks, self.w_convs):
            y = block(x)
            x = y + w_conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # [B, H, W, 2]
        x = x.permute(0, 3, 1, 2)  # [B, 2, H, W]
        return x


# -------------------- Networks: FNO, CNO, CFNO --------------------
class FNO2d_small(nn.Module):
    def __init__(self, modes=8, width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(depth)])     # fourier transform
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])    # weights
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):  # x: [B,1,H,W] source f
        # B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,H,W,1]
        x = self.fc0(x)  # [B,H,W,width]
        x = x.permute(0, 3, 1, 2)  # [B,width,H,W]
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# CNO model: use ChebSpectralConv2d blocks instead of Fourier
class CNO2d_small(nn.Module):
    def __init__(self, cheb_modes=(8, 8), width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList(
            [ChebSpectralConv2d(width, width, cheb_modes[0], cheb_modes[1]) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# CFNO combining both
class CFNO2d_small(nn.Module):
    def __init__(self, modes=8, cheb_modes=(8, 8), width=16, depth=3, alpha_init=0.5, input_features=3, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([CFNOBlock(width, width, modes, cheb_modes, alpha_init=alpha_init) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
