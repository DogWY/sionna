#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and layers for the MIMO package."""

from typing import Optional
import numpy as np
import torch

from sionna.phy.block import Block
from sionna.phy.config import config, dtypes, Precision
from sionna.phy.utils import expand_to_rank, insert_dims, inv_cholesky

__all__ = [
    "complex2real_vector",
    "real2complex_vector",
    "complex2real_matrix",
    "real2complex_matrix",
    "complex2real_covariance",
    "real2complex_covariance",
    "complex2real_channel",
    "real2complex_channel",
    "whiten_channel",
    "List2LLR",
    "List2LLRSimple",
]


def complex2real_vector(z: torch.Tensor) -> torch.Tensor:
    r"""Transforms a complex-valued vector into its real-valued equivalent.

    Transforms the last dimension of a complex-valued tensor into
    its real-valued equivalent by stacking the real and imaginary
    parts on top of each other.

    For a vector :math:`\mathbf{z}\in \mathbb{C}^M` with real and imaginary
    parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively, this function returns
    the vector :math:`\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`.

    :param z: [..., M], `torch.complex`. Complex-valued input vector.

    :output z_real: [..., 2M], `torch.float`. Real-valued equivalent vector.

    .. rubric:: Examples

    .. code-block:: python

        z = torch.complex(torch.tensor([1., 2.]), torch.tensor([3., 4.]))
        zr = complex2real_vector(z)
        # zr = tensor([1., 2., 3., 4.])
    """
    x = z.real
    y = z.imag
    return torch.cat([x, y], dim=-1)


def real2complex_vector(z: torch.Tensor) -> torch.Tensor:
    r"""Transforms a real-valued vector into its complex-valued equivalent.

    Transforms the last dimension of a real-valued tensor into
    its complex-valued equivalent by interpreting the first half
    as the real and the second half as the imaginary part.

    For a vector :math:`\mathbf{z}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in \mathbb{R}^{2M}`
    with :math:`\mathbf{x}\in \mathbb{R}^M` and :math:`\mathbf{y}\in \mathbb{R}^M`,
    this function returns
    the vector :math:`\mathbf{x}+j\mathbf{y}\in\mathbb{C}^M`.

    :param z: [..., 2M], `torch.float`. Real-valued input vector.

    :output z_complex: [..., M], `torch.complex`. Complex-valued equivalent vector.

    .. rubric:: Examples

    .. code-block:: python

        z = torch.tensor([1., 2., 3., 4.])
        zc = real2complex_vector(z)
        # zc = tensor([1.+3.j, 2.+4.j])
    """
    x, y = z.chunk(2, dim=-1)
    return torch.complex(x, y)


def complex2real_matrix(z: torch.Tensor) -> torch.Tensor:
    r"""Transforms a complex-valued matrix into its real-valued equivalent.

    Transforms the last two dimensions of a complex-valued tensor into
    their real-valued matrix equivalent representation.

    For a matrix :math:`\mathbf{Z}\in \mathbb{C}^{M\times K}` with real and imaginary
    parts :math:`\mathbf{X}\in \mathbb{R}^{M\times K}` and
    :math:`\mathbf{Y}\in \mathbb{R}^{M\times K}`, respectively, this function returns
    the matrix :math:`\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}`, given as

    .. math::

        \tilde{\mathbf{Z}} = \begin{pmatrix}
                                \mathbf{X} & -\mathbf{Y}\\
                                \mathbf{Y} & \mathbf{X}
                             \end{pmatrix}.

    :param z: [..., M, K], `torch.complex`. Complex-valued input matrix.

    :output z_real: [..., 2M, 2K], `torch.float`. Real-valued equivalent matrix.

    .. rubric:: Examples

    .. code-block:: python

        z = torch.complex(torch.ones(2, 3), torch.ones(2, 3) * 2)
        zr = complex2real_matrix(z)
        # zr.shape = torch.Size([4, 6])
    """
    x = z.real
    y = z.imag
    row1 = torch.cat([x, -y], dim=-1)
    row2 = torch.cat([y, x], dim=-1)
    return torch.cat([row1, row2], dim=-2)


def real2complex_matrix(z: torch.Tensor) -> torch.Tensor:
    r"""Transforms a real-valued matrix into its complex-valued equivalent.

    Transforms the last two dimensions of a real-valued tensor into
    their complex-valued matrix equivalent representation.

    For a matrix :math:`\tilde{\mathbf{Z}}\in \mathbb{R}^{2M\times 2K}`,
    satisfying

    .. math::

        \tilde{\mathbf{Z}} = \begin{pmatrix}
                                \mathbf{X} & -\mathbf{Y}\\
                                \mathbf{Y} & \mathbf{X}
                             \end{pmatrix}

    with :math:`\mathbf{X}\in \mathbb{R}^{M\times K}` and
    :math:`\mathbf{Y}\in \mathbb{R}^{M\times K}`, this function returns
    the matrix :math:`\mathbf{Z}=\mathbf{X}+j\mathbf{Y}\in\mathbb{C}^{M\times K}`.

    :param z: [..., 2M, 2K], `torch.float`. Real-valued input matrix.

    :output z_complex: [..., M, K], `torch.complex`. Complex-valued equivalent matrix.

    .. rubric:: Examples

    .. code-block:: python

        zr = torch.tensor([[1., 1., -2., -2.],
                           [1., 1., -2., -2.],
                           [2., 2., 1., 1.],
                           [2., 2., 1., 1.]])
        zc = real2complex_matrix(zr)
        # zc.shape = torch.Size([2, 2])
    """
    m = z.shape[-2] // 2
    k = z.shape[-1] // 2
    x = z[..., :m, :k]
    y = z[..., m:, :k]
    return torch.complex(x, y)


def complex2real_covariance(r: torch.Tensor) -> torch.Tensor:
    r"""Transforms a complex-valued covariance matrix to its real-valued equivalent.

    Assume a proper complex random variable :math:`\mathbf{z}\in\mathbb{C}^M` :cite:p:`ProperRV`
    with covariance matrix :math:`\mathbf{R}= \in\mathbb{C}^{M\times M}`
    and real and imaginary parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively.
    This function transforms the given :math:`\mathbf{R}` into the covariance matrix of the real-valued equivalent
    vector :math:`\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`, which
    is computed as :cite:p:`CovProperRV`

    .. math::

        \mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
        \begin{pmatrix}
            \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
            \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
        \end{pmatrix}.

    :param r: [..., M, M], `torch.complex`. Complex-valued covariance matrix.

    :output q: [..., 2M, 2M], `torch.float`. Real-valued equivalent covariance matrix.

    .. rubric:: Examples

    .. code-block:: python

        r = torch.complex(torch.eye(2), torch.zeros(2, 2))
        rr = complex2real_covariance(r)
        # rr.shape = torch.Size([4, 4])
    """
    q = complex2real_matrix(r)
    return q / 2.0


def real2complex_covariance(q: torch.Tensor) -> torch.Tensor:
    r"""Transforms a real-valued covariance matrix to its complex-valued equivalent.

    Assume a proper complex random variable :math:`\mathbf{z}\in\mathbb{C}^M` :cite:p:`ProperRV`
    with covariance matrix :math:`\mathbf{R}= \in\mathbb{C}^{M\times M}`
    and real and imaginary parts :math:`\mathbf{x}\in \mathbb{R}^M` and
    :math:`\mathbf{y}\in \mathbb{R}^M`, respectively.
    This function transforms the given covariance matrix of the real-valued equivalent
    vector :math:`\tilde{\mathbf{z}}=\left[\mathbf{x}^{\mathsf{T}}, \mathbf{y}^{\mathsf{T}} \right ]^{\mathsf{T}}\in\mathbb{R}^{2M}`, which
    is given as :cite:p:`CovProperRV`

    .. math::

        \mathbb{E}\left[\tilde{\mathbf{z}}\tilde{\mathbf{z}}^{\mathsf{H}} \right] =
        \begin{pmatrix}
            \frac12\Re\{\mathbf{R}\} & -\frac12\Im\{\mathbf{R}\}\\
            \frac12\Im\{\mathbf{R}\} & \frac12\Re\{\mathbf{R}\}
        \end{pmatrix},

    into is complex-valued equivalent :math:`\mathbf{R}`.

    :param q: [..., 2M, 2M], `torch.float`. Real-valued covariance matrix.

    :output r: [..., M, M], `torch.complex`. Complex-valued equivalent covariance matrix.

    .. rubric:: Examples

    .. code-block:: python

        q = torch.eye(4) * 0.5
        r = real2complex_covariance(q)
        # r.shape = torch.Size([2, 2])
    """
    r = real2complex_matrix(q)
    return r * 2.0


def complex2real_channel(
    y: torch.Tensor, h: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Transforms a complex-valued MIMO channel into its real-valued equivalent.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}`.

    This function returns the real-valued equivalent representations of
    :math:`\mathbf{y}`, :math:`\mathbf{H}`, and :math:`\mathbf{S}`,
    which are used by a wide variety of MIMO detection algorithms (Section VII) :cite:p:`YH2015`.
    These are obtained by applying :meth:`~sionna.phy.mimo.complex2real_vector` to :math:`\mathbf{y}`,
    :meth:`~sionna.phy.mimo.complex2real_matrix` to :math:`\mathbf{H}`,
    and :meth:`~sionna.phy.mimo.complex2real_covariance` to :math:`\mathbf{S}`.

    :param y: [..., M], `torch.complex`. Complex-valued received signals.
    :param h: [..., M, K], `torch.complex`. Complex-valued channel matrices.
    :param s: [..., M, M], `torch.complex`. Complex-valued noise covariance matrices.

    :output yr: [..., 2M], `torch.float`. Real-valued equivalent received signals.
    :output hr: [..., 2M, 2K], `torch.float`. Real-valued equivalent channel matrices.
    :output sr: [..., 2M, 2M], `torch.float`. Real-valued equivalent noise covariance matrices.

    .. rubric:: Examples

    .. code-block:: python

        y = torch.complex(torch.randn(4), torch.randn(4))
        h = torch.complex(torch.randn(4, 2), torch.randn(4, 2))
        s = torch.eye(4, dtype=torch.complex64)
        yr, hr, sr = complex2real_channel(y, h, s)
    """
    yr = complex2real_vector(y)
    hr = complex2real_matrix(h)
    sr = complex2real_covariance(s)
    return yr, hr, sr


def real2complex_channel(
    y: torch.Tensor, h: torch.Tensor, s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Transforms a real-valued MIMO channel into its complex-valued equivalent.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}`.

    This function transforms the real-valued equivalent representations of
    :math:`\mathbf{y}`, :math:`\mathbf{H}`, and :math:`\mathbf{S}`, as, e.g.,
    obtained with the function :meth:`~sionna.phy.mimo.complex2real_channel`,
    back to their complex-valued equivalents (Section VII) :cite:p:`YH2015`.

    :param y: [..., 2M], `torch.float`. Real-valued received signals.
    :param h: [..., 2M, 2K], `torch.float`. Real-valued channel matrices.
    :param s: [..., 2M, 2M], `torch.float`. Real-valued noise covariance matrices.

    :output yc: [..., M], `torch.complex`. Complex-valued equivalent received signals.
    :output hc: [..., M, K], `torch.complex`. Complex-valued equivalent channel matrices.
    :output sc: [..., M, M], `torch.complex`. Complex-valued equivalent noise covariance matrices.

    .. rubric:: Examples

    .. code-block:: python

        yr = torch.randn(8)
        hr = torch.randn(8, 4)
        sr = torch.eye(8)
        yc, hc, sc = real2complex_channel(yr, hr, sr)
    """
    yc = real2complex_vector(y)
    hc = real2complex_matrix(h)
    sc = real2complex_covariance(s)
    return yc, hc, sc


def whiten_channel(
    y: torch.Tensor,
    h: torch.Tensor,
    s: torch.Tensor,
    return_s: bool = True,
) -> tuple[torch.Tensor, ...]:
    r"""Whitens a canonical MIMO channel.

    Assume the canonical MIMO channel model

    .. math::

        \mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^M(\mathbb{R}^M)` is the received signal vector,
    :math:`\mathbf{x}\in\mathbb{C}^K(\mathbb{R}^K)` is the vector of transmitted symbols,
    :math:`\mathbf{H}\in\mathbb{C}^{M\times K}(\mathbb{R}^{M\times K})` is the known channel matrix,
    and :math:`\mathbf{n}\in\mathbb{C}^M(\mathbb{R}^M)` is a noise vector with covariance
    matrix :math:`\mathbf{S}\in\mathbb{C}^{M\times M}(\mathbb{R}^{M\times M})`.

    This function whitens this channel by multiplying :math:`\mathbf{y}` and
    :math:`\mathbf{H}` from the left by
    :math:`\mathbf{S}^{-\frac{1}{2}}=\mathbf{L}^{-1}`, where
    :math:`\mathbf{L}\in \mathbb{C}^{M\times M}` is the Cholesky
    decomposition of :math:`\mathbf{S}`.
    Optionally, the whitened noise covariance matrix :math:`\mathbf{I}_M`
    can be returned.

    :param y: [..., M], `torch.float` or `torch.complex`. Received signals.
    :param h: [..., M, K], `torch.float` or `torch.complex`. Channel matrices.
    :param s: [..., M, M], `torch.float` or `torch.complex`. Noise covariance matrices.
    :param return_s: `bool`, (default `True`).
        If `True`, the whitened covariance matrix is returned.

    :output yw: [..., M], `torch.float` or `torch.complex`. Whitened received signals.
    :output hw: [..., M, K], `torch.float` or `torch.complex`. Whitened channel matrices.
    :output sw: [..., M, M], `torch.float` or `torch.complex`.
        Whitened noise covariance matrices.
        Only returned if ``return_s`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        y = torch.complex(torch.randn(4), torch.randn(4))
        h = torch.complex(torch.randn(4, 2), torch.randn(4, 2))
        s = torch.eye(4, dtype=torch.complex64) * 0.5
        yw, hw, sw = whiten_channel(y, h, s)
    """
    # Compute whitening matrix
    l_inv = inv_cholesky(s)

    # Whiten observation and channel matrix
    yw = (l_inv @ y.unsqueeze(-1)).squeeze(-1)
    hw = l_inv @ h

    if return_s:
        # Ideal interference covariance matrix after whitening
        m = s.shape[-2]
        sw = torch.eye(m, dtype=s.dtype, device=s.device)
        # Expand to batch shape
        batch_shape = s.shape[:-2]
        if batch_shape:
            sw = sw.expand(*batch_shape, m, m)
        return yw, hw, sw
    else:
        return yw, hw


class List2LLR(Block):
    r"""
    Abstract class defining a callable to compute LLRs from a list of
    candidate vectors (or paths) provided by a MIMO detector.

    The following channel model is assumed

    .. math::
        \bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}

    where :math:`\bar{\mathbf{y}}\in\mathbb{C}^S` are the channel outputs,
    :math:`\mathbf{R}\in\mathbb{C}^{S\times S}` is an upper-triangular matrix,
    :math:`\bar{\mathbf{x}}\in\mathbb{C}^S` is the transmitted vector whose entries
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    and :math:`\bar{\mathbf{n}}\in\mathbb{C}^S` is white noise
    with :math:`\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}`.

    It is assumed that a MIMO detector such as :class:`~sionna.phy.mimo.KBestDetector`
    produces :math:`K` candidate solutions :math:`\bar{\mathbf{x}}_k\in\mathcal{C}^S`
    and their associated distance metrics :math:`d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2`
    for :math:`k=1,\dots,K`. This layer can also be used with the real-valued representation of the channel.

    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [..., M], `torch.complex` or `torch.float`. Channel outputs of the whitened channel.
    :input r: [..., num_streams, num_streams], same dtype as ``y``.
        Upper triangular channel matrix of the whitened channel.
    :input dists: [..., num_paths], `torch.float`. Distance metric for each path (or candidate).
    :input path_inds: [..., num_paths, num_streams], `torch.int32`.
        Symbol indices for every stream of every path (or candidate).
    :input path_syms: [..., num_paths, num_streams], same dtype as ``y``.
        Constellation symbol for every stream of every path (or candidate).

    :output llr: [..., num_streams, num_bits_per_symbol], `torch.float`.
        LLRs for all bits of every stream.

    .. rubric:: Notes

    An implementation of this class does not need to make use of all of
    the provided inputs which enable various different implementations.
    """

    def __init__(
        self,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)


class List2LLRSimple(List2LLR):
    r"""
    Computes LLRs from a list of candidate vectors (or paths) provided by a MIMO detector.

    The following channel model is assumed:

    .. math::
        \bar{\mathbf{y}} = \mathbf{R}\bar{\mathbf{x}} + \bar{\mathbf{n}}

    where :math:`\bar{\mathbf{y}}\in\mathbb{C}^S` are the channel outputs,
    :math:`\mathbf{R}\in\mathbb{C}^{S\times S}` is an upper-triangular matrix,
    :math:`\bar{\mathbf{x}}\in\mathbb{C}^S` is the transmitted vector whose entries
    are uniformly and independently drawn from the constellation :math:`\mathcal{C}`,
    and :math:`\bar{\mathbf{n}}\in\mathbb{C}^S` is white noise
    with :math:`\mathbb{E}\left[\bar{\mathbf{n}}\right]=\mathbf{0}` and
    :math:`\mathbb{E}\left[\bar{\mathbf{n}}\bar{\mathbf{n}}^{\mathsf{H}}\right]=\mathbf{I}`.

    It is assumed that a MIMO detector such as :class:`~sionna.phy.mimo.KBestDetector`
    produces :math:`K` candidate solutions :math:`\bar{\mathbf{x}}_k\in\mathcal{C}^S`
    and their associated distance metrics :math:`d_k=\lVert \bar{\mathbf{y}} - \mathbf{R}\bar{\mathbf{x}}_k \rVert^2`
    for :math:`k=1,\dots,K`. This layer can also be used with the real-valued representation of the channel.

    The LLR for the :math:`i\text{th}` bit of the :math:`k\text{th}` stream is computed as

    .. math::
        \begin{aligned}
            LLR(k,i) &= \log\left(\frac{\Pr(b_{k,i}=1|\bar{\mathbf{y}},\mathbf{R})}{\Pr(b_{k,i}=0|\bar{\mathbf{y}},\mathbf{R})}\right)\\
                &\approx \min_{j \in  \mathcal{C}_{k,i,0}}d_j - \min_{j \in  \mathcal{C}_{k,i,1}}d_j
        \end{aligned}

    where :math:`\mathcal{C}_{k,i,1}` and :math:`\mathcal{C}_{k,i,0}` are the set of indices
    in the list of candidates for which the :math:`i\text{th}` bit of the :math:`k\text{th}`
    stream is equal to 1 and 0, respectively. The LLRs are clipped to :math:`\pm LLR_\text{clip}`
    which can be configured through the parameter ``llr_clip_val``.

    If :math:`\mathcal{C}_{k,i,0}` is empty, :math:`LLR(k,i)=LLR_\text{clip}`;
    if :math:`\mathcal{C}_{k,i,1}` is empty, :math:`LLR(k,i)=-LLR_\text{clip}`.

    :param num_bits_per_symbol: Number of bits per constellation symbol
    :param llr_clip_val: The absolute values of LLRs are clipped to this value.
        Defaults to 20.0.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [..., M], `torch.complex` or `torch.float`. Channel outputs of the whitened channel.
    :input r: [..., num_streams, num_streams], same dtype as ``y``.
        Upper triangular channel matrix of the whitened channel.
    :input dists: [..., num_paths], `torch.float`. Distance metric for each path (or candidate).
    :input path_inds: [..., num_paths, num_streams], `torch.int32`.
        Symbol indices for every stream of every path (or candidate).
    :input path_syms: [..., num_paths, num_streams], same dtype as ``y``.
        Constellation symbol for every stream of every path (or candidate).

    :output llr: [..., num_streams, num_bits_per_symbol], `torch.float`.
        LLRs for all bits of every stream.

    .. rubric:: Examples

    .. code-block:: python

        list2llr = List2LLRSimple(num_bits_per_symbol=4)
        # Use with KBestDetector outputs
    """

    def __init__(
        self,
        num_bits_per_symbol: int,
        llr_clip_val: float = 20.0,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        # Array composed of binary representations of all symbol indices
        num_points = 2**num_bits_per_symbol
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(num_points):
            a[i, :] = np.array(
                list(np.binary_repr(i, num_bits_per_symbol)), dtype=np.int32
            )

        # Compute symbol indices for which the bits are 0 or 1, e.g.:
        # The ith column of c0 provides all symbol indices for which
        # the ith bit is 0.
        c0 = np.zeros([num_points // 2, num_bits_per_symbol])
        c1 = np.zeros([num_points // 2, num_bits_per_symbol])
        for i in range(num_bits_per_symbol):
            c0[:, i] = np.where(a[:, i] == 0)[0]
            c1[:, i] = np.where(a[:, i] == 1)[0]

        # Convert to tensor and add dummy dimensions needed for broadcasting
        # Shape: [1, 1, 1, num_points/2, num_bits_per_symbol]
        c0_tensor = torch.tensor(c0, dtype=torch.int32, device=self.device)
        self.register_buffer(
            "_c0",
            expand_to_rank(c0_tensor, 5, 0),
            persistent=False,
        )
        c1_tensor = torch.tensor(c1, dtype=torch.int32, device=self.device)
        self.register_buffer(
            "_c1",
            expand_to_rank(c1_tensor, 5, 0),
            persistent=False,
        )

        # Assign this absolute value to all LLRs without counter-hypothesis
        self._llr_clip_val = llr_clip_val

    @property
    def llr_clip_val(self) -> float:
        """The value to which the absolute values of LLRs are clipped."""
        return self._llr_clip_val

    @llr_clip_val.setter
    def llr_clip_val(self, value: float) -> None:
        self._llr_clip_val = value

    @staticmethod
    @torch.compile(fullgraph=True)
    def _fused_equal_any(path_inds: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Fused equal + any operation to avoid materializing large intermediate tensor.

        When compiled, this fuses the comparison and reduction into a single kernel.

        :param path_inds: Symbol indices with shape [..., num_paths, num_streams, 1]
        :param c: Reference indices with shape [1, 1, 1, num_symbols/2, num_bits]

        :output result: [..., num_paths, num_streams, num_bits], `torch.bool`.
            Boolean tensor indicating bit matches.
        """
        return (path_inds == c).any(dim=-2)

    def call(
        self,
        y: torch.Tensor,
        r: torch.Tensor,
        dists: torch.Tensor,
        path_inds: torch.Tensor,
        path_syms: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LLRs from candidate paths."""
        # dists:     [batch_size, num_paths]
        # path_inds: [batch_size, num_paths, num_streams]

        # Scaled by 0.5 to account for the reduced noise power in each complex
        # dimension if real channel representation is used.
        if not y.is_complex():
            dists = dists / 2.0

        # Compute for every symbol in every path which bits are 0 or 1
        # b0/b1: [batch_size, num_paths, num_streams, num_bits_per_symbol]
        # Use compiled function to fuse equal+any, avoiding large intermediate tensor
        path_inds = insert_dims(path_inds, 2, axis=-1)
        b0 = self._fused_equal_any(path_inds, self._c0)
        b1 = self._fused_equal_any(path_inds, self._c1)

        # Compute distances for all bits in all paths, set distance to inf
        # if the bit does not have the correct value
        dists = expand_to_rank(dists, b0.dim(), axis=-1)
        d0 = torch.where(b0, dists, torch.tensor(float("inf"), dtype=dists.dtype, device=dists.device))
        d1 = torch.where(b1, dists, torch.tensor(float("inf"), dtype=dists.dtype, device=dists.device))

        # Compute minimum distance for each bit in each stream
        # l0/l1: [batch_size, num_streams, num_bits_per_symbol]
        l0, _ = d0.min(dim=1)
        l1, _ = d1.min(dim=1)

        # Compute LLRs
        llr = l0 - l1

        # Clip LLRs
        llr = llr.clamp(-self._llr_clip_val, self._llr_clip_val)

        return llr
