#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Classes and functions related to OFDM channel estimation."""

import itertools
import json
from abc import abstractmethod
from importlib_resources import files
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.special import jv

from sionna.phy import Block, PI, SPEED_OF_LIGHT
from sionna.phy.channel.tr38901 import models
from sionna.phy.config import config, Config, Precision
from sionna.phy.object import Object
from sionna.phy.ofdm import RemoveNulledSubcarriers, ResourceGrid
from sionna.phy.utils import expand_to_rank, flatten_last_dims
from sionna.phy.utils.linalg import matrix_pinv

__all__ = [
    "BaseChannelEstimator",
    "LSChannelEstimator",
    "BaseChannelInterpolator",
    "NearestNeighborInterpolator",
    "LinearInterpolator",
    "LMMSEInterpolator1D",
    "SpatialChannelFilter",
    "LMMSEInterpolator",
    "tdl_freq_cov_mat",
    "tdl_time_cov_mat",
]


class BaseChannelEstimator(Block):
    r"""Abstract block for implementing an OFDM channel estimator.

    Any block that implements an OFDM channel estimator must implement this
    class and its
    :meth:`~sionna.phy.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    abstract method.

    This class extracts the pilots from the received resource grid ``y``, calls
    the :meth:`~sionna.phy.ofdm.BaseChannelEstimator.estimate_at_pilot_locations`
    method to estimate the channel for the pilot-carrying resource elements,
    and then interpolates the channel to compute channel estimates for the
    data-carrying resource elements using the interpolation method specified by
    ``interpolation_type`` or the ``interpolator`` object.

    :param resource_grid: Resource grid
    :param interpolation_type: The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are
        :class:`~sionna.phy.ofdm.NearestNeighborInterpolator` (``"nn"``),
        :class:`~sionna.phy.ofdm.LinearInterpolator` without (``"lin"``) or
        with averaging across OFDM symbols (``"lin_time_avg"``).
    :param interpolator: An instance of
        :class:`~sionna.phy.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.phy.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specified
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Observed resource grid.
    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.ofdm import ResourceGrid, LSChannelEstimator

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=2,
                          num_streams_per_tx=2,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

        estimator = LSChannelEstimator(rg, interpolation_type="lin")

        batch_size = 16
        y = torch.randn(batch_size, 1, 4, 14, 64, dtype=torch.complex64)
        no = torch.ones(1) * 0.1

        h_hat, err_var = estimator(y, no)
        print(h_hat.shape)
        # torch.Size([16, 1, 4, 2, 2, 14, 60])
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        interpolation_type: str = "nn",
        interpolator: Optional["BaseChannelInterpolator"] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        assert isinstance(
            resource_grid, ResourceGrid
        ), "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern
        self._remove_nulled_scs = RemoveNulledSubcarriers(
            resource_grid, precision=self.precision, device=self.device
        )

        assert interpolation_type in [
            "nn",
            "lin",
            "lin_time_avg",
            None,
        ], "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type

        if interpolator is not None:
            assert isinstance(
                interpolator, BaseChannelInterpolator
            ), "`interpolator` must implement the BaseChannelInterpolator interface"
            self._interpol = interpolator
        elif self._interpolation_type == "nn":
            self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin":
            self._interpol = LinearInterpolator(self._pilot_pattern)
        elif self._interpolation_type == "lin_time_avg":
            self._interpol = LinearInterpolator(self._pilot_pattern, time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols
        mask = flatten_last_dims(self._pilot_pattern.mask)
        # Use stable=True to ensure consistent ordering across CPU/GPU
        # (otherwise ties among pilot positions may be ordered differently)
        pilot_ind = torch.argsort(mask.float(), dim=-1, descending=True, stable=True)
        self.register_buffer(
            "_pilot_ind",
            pilot_ind[..., :num_pilot_symbols].to(device=self.device),
            persistent=False,
        )

    @abstractmethod
    def estimate_at_pilot_locations(
        self, y_pilots: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the channel for the pilot-carrying resource elements.

        This is an abstract method that must be implemented by a concrete
        OFDM channel estimator that implements this class.

        :param y_pilots: [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `torch.complex`.
            Observed signals for the pilot-carrying resource elements.
        :param no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
            Variance of the AWGN.

        :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `torch.complex`.
            Channel estimates for the pilot-carrying resource elements.
        :output err_var: Same shape as ``h_hat``, `torch.float`.
            Channel estimation error variance for the pilot-carrying
            resource elements.
        """
        pass

    def call(
        self, y: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Remove nulled subcarriers (guards, dc)
        y_eff = self._remove_nulled_scs(y)

        # Flatten the resource grid for pilot extraction
        # New shape: [..., num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)

        # Gather pilots along the last dimension
        # Expand pilot_ind to match y_eff_flat dimensions
        pilot_ind = self._pilot_ind.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pilot_ind = pilot_ind.expand(
            y_eff_flat.shape[0], y_eff_flat.shape[1], y_eff_flat.shape[2], -1, -1, -1
        )
        y_pilots = torch.gather(
            y_eff_flat.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, pilot_ind.shape[3], pilot_ind.shape[4], -1),
            -1,
            pilot_ind,
        )

        # Compute channel estimates at pilot locations
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)

        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_hat, err_var = self._interpol(h_hat, err_var)
            err_var = torch.maximum(err_var, torch.zeros_like(err_var))

        return h_hat, err_var


class LSChannelEstimator(BaseChannelEstimator):
    r"""Least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated across the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized
    observation, where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated across
    the entire resource grid.

    :param resource_grid: Resource grid
    :param interpolation_type: The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are
        :class:`~sionna.phy.ofdm.NearestNeighborInterpolator` (``"nn"``),
        :class:`~sionna.phy.ofdm.LinearInterpolator` without (``"lin"``) or
        with averaging across OFDM symbols (``"lin_time_avg"``).
    :param interpolator: An instance of
        :class:`~sionna.phy.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.phy.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specified
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], `torch.complex`.
        Observed resource grid.
    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
        Variance of the AWGN.

    :output h_ls: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_ls``, `torch.float`.
        Channel estimation error variance across the entire resource grid
        for all transmitters and streams.
    """

    def estimate_at_pilot_locations(
        self, y_pilots: torch.Tensor, no: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], torch.complex
        #     The observed signals for the pilot-carrying resource elements.
        #
        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   torch.float
        #     The variance of the AWGN.

        # Get pilots tensor
        pilots = self._pilot_pattern.pilots.to(y_pilots.device)

        # Compute LS channel estimates
        # Safe division to handle zero pilots
        h_ls = torch.where(
            pilots.abs() > 0,
            y_pilots / pilots,
            torch.zeros_like(y_pilots),
        )

        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        no = expand_to_rank(no, h_ls.dim(), -1)

        # Expand rank of pilots for broadcasting
        pilots_expanded = expand_to_rank(pilots, h_ls.dim(), 0)

        # Compute error variance, broadcastable to the shape of h_ls
        err_var = torch.where(
            pilots_expanded.abs() > 0,
            no / pilots_expanded.abs().square(),
            torch.zeros_like(no),
        )

        # Broadcast err_var to match h_ls shape
        err_var = err_var.expand(h_ls.shape).clone()

        return h_ls, err_var


class BaseChannelInterpolator(Object):
    r"""Abstract class for implementing an OFDM channel interpolator.

    Any class that implements an OFDM channel interpolator must implement this
    callable class.

    A channel interpolator is used by an OFDM channel estimator
    (:class:`~sionna.phy.ofdm.BaseChannelEstimator`) to compute channel estimates
    for the data-carrying resource elements from the channel estimates for the
    pilot-carrying resource elements.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates for the pilot-carrying resource elements.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Channel estimation error variances for the pilot-carrying resource
        elements.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.
    """

    @abstractmethod
    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class NearestNeighborInterpolator(BaseChannelInterpolator):
    r"""Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided channel estimates and error
    variances according to the nearest neighbor method. It is assumed
    that the measurements were taken at the nonzero positions of a
    :class:`~sionna.phy.ofdm.PilotPattern`.

    The figure below shows how four channel estimates are interpolated
    across a resource grid. Grey fields indicate measurement positions
    while the colored regions show which resource elements are assigned
    to the same measurement value.

    .. image:: ../figures/nearest_neighbor_interpolation.png

    :param pilot_pattern: Used pilot pattern.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates for the pilot-carrying resource elements.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Channel estimation error variances for the pilot-carrying resource
        elements.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.
    """

    def __init__(self, pilot_pattern) -> None:
        super().__init__()

        assert pilot_pattern.num_pilot_symbols > 0, "The pilot pattern cannot be empty"

        # Reshape mask to shape [-1, num_ofdm_symbols, num_effective_subcarriers]
        mask = pilot_pattern.mask.cpu().numpy()
        mask_shape = mask.shape  # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots.cpu().numpy()
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots) == 0, -1))
        assert (
            max_num_zero_pilots < pilots.shape[-1]
        ), "Each pilot sequence must have at least one nonzero entry"

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int64)
        for a in range(gather_ind.shape[0]):  # For each pilot pattern...
            i_p, j_p = np.where(mask[a])  # ...determine the pilot indices

            for i in range(mask_shape[-2]):  # Iterate over...
                for j in range(mask_shape[-1]):  # ... all resource elements
                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i - i_p) + np.abs(j - j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a]) == 0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance
                    ind = np.argmin(d)

                    # Store it in the index tensor
                    gather_ind[a, i, j] = ind

        # Reshape to the original shape of the mask.
        self.register_buffer(
            "_gather_ind",
            torch.tensor(
                np.reshape(gather_ind, mask_shape),
                dtype=torch.int64,
                device=self.device,
            ),
            persistent=False,
        )

    def _interpolate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Interpolate using nearest neighbor method."""
        # inputs: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #          num_pilots]
        device = inputs.device

        # Move batch dimensions to end
        # [num_tx, num_streams_per_tx, num_pilots, batch_size, num_rx, num_rx_ant]
        inputs = inputs.permute(3, 4, 5, 0, 1, 2)

        # Gather using indices
        # gather_ind: [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #              num_effective_subcarriers]
        # Move to input device to avoid device mismatch
        gather_ind = self._gather_ind.to(device)

        # Expand gather_ind to match batch dimensions
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  num_effective_subcarriers, batch_size, num_rx, num_rx_ant]
        gather_ind = gather_ind.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gather_ind = gather_ind.expand(
            -1, -1, -1, -1, inputs.shape[3], inputs.shape[4], inputs.shape[5]
        )

        # inputs: [num_tx, num_streams_per_tx, num_pilots, batch_size, num_rx,
        #          num_rx_ant]
        # Expand inputs to match output shape
        inputs = (
            inputs.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, gather_ind.shape[2], gather_ind.shape[3], -1, -1, -1, -1)
        )

        # Gather along the pilots dimension
        outputs = torch.gather(inputs, 4, gather_ind.unsqueeze(4)).squeeze(4)

        # Move batch dimensions back to front
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  num_ofdm_symbols, num_effective_subcarriers]
        outputs = outputs.permute(4, 5, 6, 0, 1, 2, 3)

        return outputs

    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var


class LinearInterpolator(BaseChannelInterpolator):
    r"""Linear channel estimate interpolation on a resource grid.

    This class computes for each element of an OFDM resource grid
    a channel estimate based on ``num_pilots`` provided channel estimates and
    error variances through linear interpolation.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.phy.ofdm.PilotPattern`.

    The interpolation is done first across sub-carriers and then
    across OFDM symbols.

    :param pilot_pattern: Used pilot pattern
    :param time_avg: If enabled, measurements will be averaged across OFDM
        symbols (i.e., time). This is useful for channels that do not vary
        substantially over the duration of an OFDM frame.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates for the pilot-carrying resource elements.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Channel estimation error variances for the pilot-carrying resource
        elements.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.
    """

    def __init__(self, pilot_pattern, time_avg: bool = False) -> None:
        super().__init__()

        assert pilot_pattern.num_pilot_symbols > 0, "The pilot pattern cannot be empty"

        self._time_avg = time_avg

        # Reshape mask to shape [-1, num_ofdm_symbols, num_effective_subcarriers]
        mask = pilot_pattern.mask.cpu().numpy()
        mask_shape = mask.shape  # Store to reconstruct the original shape
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots.cpu().numpy()
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots) == 0, -1))
        assert (
            max_num_zero_pilots < pilots.shape[-1]
        ), "Each pilot sequence must have at least one nonzero entry"

        # Create actual pilot patterns for each stream over the resource grid
        z = np.zeros_like(mask, dtype=pilots.dtype)
        for a in range(z.shape[0]):
            z[a][np.where(mask[a])] = pilots[a]

        ##
        # Frequency-domain interpolation
        ##
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_x_freq",
            torch.arange(0, mask.shape[-1], dtype=torch.float32, device=config.device),
        )

        x_0_freq = np.zeros_like(mask, np.int64)
        x_1_freq = np.zeros_like(mask, np.int64)

        # Set REs of OFDM symbols without any pilot equal to -1 (dummy value)
        x_0_freq[np.sum(np.abs(z), axis=-1) == 0] = -1
        x_1_freq[np.sum(np.abs(z), axis=-1) == 0] = -1

        y_0_freq_ind = np.copy(x_0_freq)  # Indices used to gather estimates
        y_1_freq_ind = np.copy(x_1_freq)  # Indices used to gather estimates

        # For each stream
        for a in range(z.shape[0]):
            pilot_count = 0  # Counts the number of non-zero pilots

            # Indices of non-zero pilots within the pilots vector
            pilot_ind = np.where(np.abs(pilots[a]))[0]

            # Go through all OFDM symbols
            for i in range(x_0_freq.shape[1]):
                # Indices of non-zero pilots within the OFDM symbol
                pilot_ind_ofdm = np.where(np.abs(z[a][i]))[0]

                # If OFDM symbol contains only one non-zero pilot
                if len(pilot_ind_ofdm) == 1:
                    x_0_freq[a][i] = pilot_ind_ofdm[0]
                    x_1_freq[a][i] = pilot_ind_ofdm[0]
                    y_0_freq_ind[a, i] = pilot_ind[pilot_count]
                    y_1_freq_ind[a, i] = pilot_ind[pilot_count]

                # If OFDM symbol contains two or more pilots
                elif len(pilot_ind_ofdm) >= 2:
                    x0 = 0
                    x1 = 1

                    for j in range(x_0_freq.shape[2]):
                        x_0_freq[a, i, j] = pilot_ind_ofdm[x0]
                        x_1_freq[a, i, j] = pilot_ind_ofdm[x1]
                        y_0_freq_ind[a, i, j] = pilot_ind[pilot_count + x0]
                        y_1_freq_ind[a, i, j] = pilot_ind[pilot_count + x1]
                        if j == pilot_ind_ofdm[x1] and x1 < len(pilot_ind_ofdm) - 1:
                            x0 = x1
                            x1 += 1

                pilot_count += len(pilot_ind_ofdm)

        x_0_freq = np.reshape(x_0_freq, mask_shape)
        x_1_freq = np.reshape(x_1_freq, mask_shape)
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_x_0_freq",
            torch.tensor(x_0_freq, dtype=torch.float32, device=config.device),
        )
        self.register_buffer(
            "_x_1_freq",
            torch.tensor(x_1_freq, dtype=torch.float32, device=config.device),
        )

        # We add +1 to shift all indices as the input will be padded
        # at the beginning with 0
        self.register_buffer(
            "_y_0_freq_ind",
            torch.tensor(
                np.reshape(y_0_freq_ind, mask_shape) + 1,
                dtype=torch.int64,
                device=config.device,
            ),
        )
        self.register_buffer(
            "_y_1_freq_ind",
            torch.tensor(
                np.reshape(y_1_freq_ind, mask_shape) + 1,
                dtype=torch.int64,
                device=config.device,
            ),
        )

        ##
        # Time-domain interpolation
        ##
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer(
            "_x_time",
            torch.arange(
                0, mask.shape[-2], dtype=torch.float32, device=config.device
            ).unsqueeze(-1),
        )

        y_0_time_ind = np.zeros(z.shape[:2], np.int64)  # Gather indices
        y_1_time_ind = np.zeros(z.shape[:2], np.int64)  # Gather indices

        # For each stream
        for a in range(z.shape[0]):
            # Indices of OFDM symbols for which channel estimates were computed
            ofdm_ind = np.where(np.sum(np.abs(z[a]), axis=-1))[0]

            # Only one OFDM symbol with pilots
            if len(ofdm_ind) == 1:
                y_0_time_ind[a] = ofdm_ind[0]
                y_1_time_ind[a] = ofdm_ind[0]

            # Two or more OFDM symbols with pilots
            elif len(ofdm_ind) >= 2:
                x0 = 0
                x1 = 1
                for i in range(z.shape[1]):
                    y_0_time_ind[a, i] = ofdm_ind[x0]
                    y_1_time_ind[a, i] = ofdm_ind[x1]
                    if i == ofdm_ind[x1] and x1 < len(ofdm_ind) - 1:
                        x0 = x1
                        x1 += 1

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer(
            "_y_0_time_ind",
            torch.tensor(
                np.reshape(y_0_time_ind, mask_shape[:-1]),
                dtype=torch.int64,
                device=config.device,
            ),
        )
        self.register_buffer(
            "_y_1_time_ind",
            torch.tensor(
                np.reshape(y_1_time_ind, mask_shape[:-1]),
                dtype=torch.int64,
                device=config.device,
            ),
        )

        self.register_buffer(
            "_x_0_time",
            self._y_0_time_ind.unsqueeze(-1).float(),
        )
        self.register_buffer(
            "_x_1_time",
            self._y_1_time_ind.unsqueeze(-1).float(),
        )

        # Number of OFDM symbols carrying at least one pilot
        n = np.sum(np.abs(np.reshape(z, mask_shape)), axis=-1, keepdims=True)
        n = np.sum(n > 0, axis=-2, keepdims=True)
        self.register_buffer(
            "_num_pilot_ofdm_symbols",
            torch.tensor(n, dtype=torch.float32, device=config.device),
        )

    def _interpolate_1d(
        self,
        inputs: torch.Tensor,
        x: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        y0_ind: torch.Tensor,
        y1_ind: torch.Tensor,
    ) -> torch.Tensor:
        """Perform 1D linear interpolation."""
        # inputs: [num_tx, num_streams_per_tx, 1+num_pilots, batch_size, num_rx,
        #          num_rx_ant]

        # Expand indices to match batch dimensions
        batch_dims = inputs.shape[3:]

        # y0_ind, y1_ind: [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #                  num_effective_subcarriers]
        y0_ind_expanded = y0_ind.to(inputs.device)
        y1_ind_expanded = y1_ind.to(inputs.device)

        # Add batch dimensions
        for _ in batch_dims:
            y0_ind_expanded = y0_ind_expanded.unsqueeze(-1)
            y1_ind_expanded = y1_ind_expanded.unsqueeze(-1)

        y0_ind_expanded = y0_ind_expanded.expand(*y0_ind.shape, *batch_dims)
        y1_ind_expanded = y1_ind_expanded.expand(*y1_ind.shape, *batch_dims)

        # Expand inputs for gathering
        # inputs needs to match shape for gather
        inputs_expanded = (
            inputs.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, y0_ind.shape[2], y0_ind.shape[3], -1, -1, -1, -1)
        )

        # Gather y0 and y1
        y0 = torch.gather(inputs_expanded, 4, y0_ind_expanded.unsqueeze(4)).squeeze(4)
        y1 = torch.gather(inputs_expanded, 4, y1_ind_expanded.unsqueeze(4)).squeeze(4)

        # Move batch dimensions back
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  num_ofdm_symbols, num_effective_subcarriers]
        y0 = y0.permute(4, 5, 6, 0, 1, 2, 3).contiguous()
        y1 = y1.permute(4, 5, 6, 0, 1, 2, 3).contiguous()

        # Compute linear interpolation
        # Expand x, x0, x1 to match output shape
        x = x.to(inputs.device)
        x0 = x0.to(inputs.device).to(y0.dtype)
        x1 = x1.to(inputs.device).to(y0.dtype)

        x = expand_to_rank(x, y0.dim(), 0)
        x0 = expand_to_rank(x0, y0.dim(), 0)
        x1 = expand_to_rank(x1, y0.dim(), 0)

        slope = torch.where(
            x1 != x0,
            (y1 - y0) / (x1 - x0),
            torch.zeros_like(y0),
        )

        return ((x - x0) * slope + y0).contiguous()

    def _interpolate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Interpolate channel estimates across the resource grid."""
        # Pad the inputs with a leading 0
        pad = (1, 0)  # Pad last dimension
        inputs = torch.nn.functional.pad(inputs, pad)

        # Move batch dimensions to end
        # [num_tx, num_streams_per_tx, 1+num_pilots, batch_size, num_rx, num_rx_ant]
        inputs = inputs.permute(3, 4, 5, 0, 1, 2).contiguous()

        # Frequency-domain interpolation
        h_hat_freq = self._interpolate_1d(
            inputs,
            self._x_freq,
            self._x_0_freq,
            self._x_1_freq,
            self._y_0_freq_ind,
            self._y_1_freq_ind,
        )

        # Time-domain averaging (optional)
        if self._time_avg:
            num_ofdm_symbols = h_hat_freq.shape[-2]
            h_hat_freq = h_hat_freq.sum(dim=-2, keepdim=True)
            n = self._num_pilot_ofdm_symbols.to(h_hat_freq.device).to(h_hat_freq.dtype)
            n = expand_to_rank(n, h_hat_freq.dim(), 0)
            h_hat_freq = h_hat_freq / n
            h_hat_freq = h_hat_freq.repeat(1, 1, 1, 1, 1, num_ofdm_symbols, 1)

        # Time-domain interpolation
        # Transpose: [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #             num_effective_subcarriers, batch_size, num_rx, num_rx_ant]
        h_hat_time = h_hat_freq.permute(3, 4, 5, 6, 0, 1, 2).contiguous()

        # Expand for time interpolation gathering
        y_0_time_ind = self._y_0_time_ind.to(h_hat_time.device)
        y_1_time_ind = self._y_1_time_ind.to(h_hat_time.device)

        # y_0_time_ind shape: [num_tx, num_streams_per_tx, num_ofdm_symbols]
        # We need to expand to match h_hat_time shape on dims 3-6
        trailing_dims = h_hat_time.shape[3:]  # [num_eff_subcarriers, batch, rx, rx_ant]

        y_0_time_expanded = y_0_time_ind
        y_1_time_expanded = y_1_time_ind

        # Add trailing dimensions
        for _ in trailing_dims:
            y_0_time_expanded = y_0_time_expanded.unsqueeze(-1)
            y_1_time_expanded = y_1_time_expanded.unsqueeze(-1)

        # Expand to match h_hat_time shape
        y_0_time_expanded = y_0_time_expanded.expand(
            *y_0_time_ind.shape, *trailing_dims
        )
        y_1_time_expanded = y_1_time_expanded.expand(
            *y_1_time_ind.shape, *trailing_dims
        )

        # Gather y0 and y1 for time interpolation
        y0 = torch.gather(h_hat_time, 2, y_0_time_expanded)
        y1 = torch.gather(h_hat_time, 2, y_1_time_expanded)

        # Move back to standard order for output
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  num_ofdm_symbols, num_effective_subcarriers]
        y0 = y0.permute(4, 5, 6, 0, 1, 2, 3).contiguous()
        y1 = y1.permute(4, 5, 6, 0, 1, 2, 3).contiguous()

        # Linear interpolation in time
        x = self._x_time.to(h_hat_time.device).to(y0.dtype)
        x0 = self._x_0_time.to(h_hat_time.device).to(y0.dtype)
        x1 = self._x_1_time.to(h_hat_time.device).to(y0.dtype)

        x = expand_to_rank(x, y0.dim(), 0)
        x0 = expand_to_rank(x0, y0.dim(), 0)
        x1 = expand_to_rank(x1, y0.dim(), 0)

        slope = torch.where(
            x1 != x0,
            (y1 - y0) / (x1 - x0),
            torch.zeros_like(y0),
        )

        h_hat_time = ((x - x0) * slope + y0).contiguous()

        return h_hat_time

    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_hat = self._interpolate(h_hat)

        # The interpolator requires complex-valued inputs
        err_var_complex = err_var.to(torch.complex64)
        err_var = self._interpolate(err_var_complex)
        err_var = err_var.real

        return h_hat, err_var


class LMMSEInterpolator1D(Object):
    r"""LMMSE interpolation across the inner dimension of the input.

    The two inner dimensions of the input ``h_hat`` form a matrix
    :math:`\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}`.
    LMMSE interpolation is performed across the inner dimension as follows:

    .. math::
        \tilde{\mathbf{h}}_n = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` and :math:`\hat{\mathbf{h}}_n` is
    the :math:`n^{\text{th}}` (transposed) row of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` interpolation LMMSE matrix:

    .. math::
        \mathbf{A}_n = \mathbf{R} \mathbf{\Pi}_n \left( \mathbf{\Pi}_n^\intercal \mathbf{R} \mathbf{\Pi}_n + \tilde{\mathbf{\Sigma}}_n \right)^{-1} \mathbf{\Pi}_n^\intercal.

    where :math:`\mathbf{R}` is the :math:`M \times M` covariance matrix across
    the inner dimension of the quantity which is estimated,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads
    :math:`K_n` values to a vector of size :math:`M` according to the
    ``pilot_mask`` for the :math:`n^{\text{th}}` row,
    and :math:`\tilde{\mathbf{\Sigma}}_n \in \mathbb{R}^{K_n \times K_n}` is
    the regularized channel estimation error covariance.
    The :math:`i^{\text{th}}` diagonal element of
    :math:`\tilde{\mathbf{\Sigma}}_n` is such that:

    .. math::

        \left[ \tilde{\mathbf{\Sigma}}_n \right]_{i,i} = \max \left\{ \left[ \mathbf{\Sigma}_n \right]_{i,i},\; 0 \right\}

    built from ``err_var`` and assumed to be diagonal.

    The returned channel estimates are

    .. math::
        \begin{bmatrix}
            {\tilde{\mathbf{h}}_1}^\intercal\\
            \vdots\\
            {\tilde{\mathbf{h}}_N}^\intercal
        \end{bmatrix}.

    The returned channel estimation error variances are the diagonal
    coefficients of

    .. math::
        \text{diag} \left( \mathbf{R} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R} \right), 1 \leq n \leq N

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size
    :math:`M \times M` that zeros the columns corresponding to rows not
    carrying any pilots.
    Note that interpolation is not performed for rows not carrying any pilots.

    **Remark**: The interpolation matrix differs across rows as different
    rows may carry pilots on different elements and/or have different
    estimation error variances.

    :param pilot_mask: Mask indicating the allocation of resource elements.
        0: Data, 1: Pilot, 2: Not used.
    :param cov_mat: Covariance matrix of the channel across the inner
        dimension
    :param last_step: Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], `torch.complex`.
        Channel estimates.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, :math:`N`, :math:`M`], `torch.float`.
        Channel estimation error variances.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, :math:`N`, :math:`M`], `torch.complex`.
        Channel estimates interpolated across the inner dimension.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        The channel estimation error variances of the interpolated channel
        estimates.
    """

    def __init__(
        self,
        pilot_mask: np.ndarray,
        cov_mat: torch.Tensor,
        last_step: bool,
    ) -> None:
        if cov_mat.dtype == torch.complex64:
            precision = "single"
        elif cov_mat.dtype == torch.complex128:
            precision = "double"
        else:
            raise TypeError("`cov_mat` dtype must be complex64 or complex128")
        super().__init__(precision=precision)

        # Register as buffer for CUDAGraph compatibility (on config.device to avoid DeviceCopy)
        self.register_buffer(
            "_rzero", torch.tensor(0.0, dtype=self.dtype, device=config.device)
        )

        # Size of inner and outer dimensions
        inner_dim_size = pilot_mask.shape[-1]
        outer_dim_size = pilot_mask.shape[-2]
        self._inner_dim_size = inner_dim_size
        self._outer_dim_size = outer_dim_size

        # Register cov_mat as buffer for CUDAGraph compatibility
        self.register_buffer("_cov_mat", cov_mat.to(config.device))
        self._last_step = last_step

        # Extract pilot locations
        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]

        # List of indices of pilots in the inner dimension
        pilot_indices = []
        max_num_pil = 0
        add_err_var_indices = np.zeros(
            [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, 5], int
        )
        # Pre-compute list of valid pilot positions for compile-friendly iteration
        # Each entry is (tx, st, oi, ii, pil_idx) as Python ints
        valid_pilot_positions = []

        for tx in range(num_tx):
            pilot_indices.append([])
            for st in range(num_streams_per_tx):
                pilot_indices[-1].append([])
                for oi in range(outer_dim_size):
                    pilot_indices[-1][-1].append([])
                    num_pil = 0
                    for ii in range(inner_dim_size):
                        if pilot_mask[tx, st, oi, ii] == 0:
                            continue
                        if pilot_mask[tx, st, oi, ii] == 1:
                            pilot_indices[tx][st][oi].append(ii)
                            indices = [tx, st, oi, num_pil, num_pil]
                            add_err_var_indices[tx, st, oi, ii] = indices
                            # Store valid position as Python tuple
                            valid_pilot_positions.append((tx, st, oi, ii, num_pil))
                            num_pil += 1
                    max_num_pil = max(max_num_pil, num_pil)

        # Store as Python list (not tensor) for compile-friendly iteration
        self._valid_pilot_positions = valid_pilot_positions

        # Register as buffer for CUDAGraph compatibility (on config.device)
        self.register_buffer(
            "_add_err_var_indices",
            torch.tensor(add_err_var_indices, dtype=torch.int64, device=config.device),
        )

        # Build pilot covariance matrix
        cov_mat_np = cov_mat.cpu().numpy()
        pil_cov_mat = np.zeros(
            [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil],
            complex,
        )
        for tx, st, oi in itertools.product(
            range(num_tx), range(num_streams_per_tx), range(outer_dim_size)
        ):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            if num_pil > 0:
                tmp = np.take(cov_mat_np, pil_ind, axis=0)
                pil_cov_mat_ = np.take(tmp, pil_ind, axis=1)
                pil_cov_mat[tx, st, oi, :num_pil, :num_pil] = pil_cov_mat_
        # Register as buffer for CUDAGraph compatibility (on config.device)
        self.register_buffer(
            "_pil_cov_mat",
            torch.tensor(pil_cov_mat, dtype=self.cdtype, device=config.device),
        )

        # Pre-compute B matrix
        b_mat = np.zeros(
            [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, inner_dim_size],
            complex,
        )
        for tx, st, oi in itertools.product(
            range(num_tx), range(num_streams_per_tx), range(outer_dim_size)
        ):
            pil_ind = pilot_indices[tx][st][oi]
            num_pil = len(pil_ind)
            if num_pil > 0:
                b_mat_ = np.take(cov_mat_np, pil_ind, axis=0)
                b_mat[tx, st, oi, :num_pil, :] = b_mat_
        # Register as buffer for CUDAGraph compatibility (on config.device)
        self.register_buffer(
            "_b_mat", torch.tensor(b_mat, dtype=self.cdtype, device=config.device)
        )

        # Indices for scatter
        pil_loc = np.zeros(
            [
                num_tx,
                num_streams_per_tx,
                outer_dim_size,
                inner_dim_size,
                max_num_pil,
                5,
            ],
            dtype=int,
        )
        for tx, st, oi, p, ii in itertools.product(
            range(num_tx),
            range(num_streams_per_tx),
            range(outer_dim_size),
            range(max_num_pil),
            range(inner_dim_size),
        ):
            if p >= len(pilot_indices[tx][st][oi]):
                pil_loc[tx, st, oi, ii, p] = [
                    tx,
                    st,
                    oi,
                    inner_dim_size,
                    inner_dim_size,
                ]
            else:
                pil_loc[tx, st, oi, ii, p] = [
                    tx,
                    st,
                    oi,
                    ii,
                    pilot_indices[tx][st][oi][p],
                ]
        # Register as buffer for CUDAGraph compatibility (on config.device)
        # Extract only the row/col indices (indices 3 and 4) for compile-friendly scatter
        # Shape: [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, max_num_pil]
        self.register_buffer(
            "_pil_loc_row",
            torch.tensor(pil_loc[..., 3], dtype=torch.int64, device=config.device),
        )
        self.register_buffer(
            "_pil_loc_col",
            torch.tensor(pil_loc[..., 4], dtype=torch.int64, device=config.device),
        )

        # Error variance matrix
        err_var_mat = np.zeros(
            [
                num_tx,
                num_streams_per_tx,
                outer_dim_size,
                inner_dim_size,
                inner_dim_size,
            ],
            complex,
        )
        for tx, st, oi in itertools.product(
            range(num_tx), range(num_streams_per_tx), range(outer_dim_size)
        ):
            pil_ind = pilot_indices[tx][st][oi]
            mask = np.zeros([inner_dim_size], complex)
            mask[pil_ind] = 1.0
            mask = np.expand_dims(mask, axis=1)
            err_var_mat[tx, st, oi] = cov_mat_np * mask
        # Register as buffer for CUDAGraph compatibility (on config.device)
        self.register_buffer(
            "_err_var_mat",
            torch.tensor(err_var_mat, dtype=self.cdtype, device=config.device),
        )

    @torch.compiler.disable  # Complex linear algebra (Cholesky, solve_triangular) is slower when compiled
    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = h_hat.shape[0]
        num_rx = h_hat.shape[1]
        num_rx_ant = h_hat.shape[2]
        num_tx = h_hat.shape[3]
        num_tx_stream = h_hat.shape[4]
        outer_dim_size = self._outer_dim_size
        inner_dim_size = self._inner_dim_size

        device = h_hat.device

        # Move internal buffers to input device to avoid device mismatch errors
        pil_loc_row = self._pil_loc_row.to(device)
        pil_loc_col = self._pil_loc_col.to(device)
        rzero = self._rzero.to(device)

        # Keep track of old error variance for later use
        err_var_old = err_var

        #####################################
        # Compute the interpolation matrix
        #####################################

        # Compute A matrix (covariance + error variance)
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = self._pil_cov_mat.to(device)
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        pil_cov_mat = expand_to_rank(pil_cov_mat, 8, 0)
        pil_cov_mat = pil_cov_mat.expand(
            batch_size, num_rx, num_rx_ant, -1, -1, -1, -1, -1
        )

        # Add error variance to diagonal using scatter
        # Transpose for scatter operation
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, max_num_pil,
        #  batch_size, num_rx, num_rx_ant]
        pil_cov_mat_ = pil_cov_mat.permute(3, 4, 5, 6, 7, 0, 1, 2).clone()
        err_var_c = err_var.to(self.cdtype)
        # [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size,
        #  batch_size, num_rx, num_rx_ant]
        err_var_ = err_var_c.permute(3, 4, 5, 6, 0, 1, 2)

        # Add error variance to diagonal using pre-computed valid positions
        # (avoids data-dependent branching for torch.compile compatibility)
        for tx, st, oi, ii, pil_idx in self._valid_pilot_positions:
            pil_cov_mat_[tx, st, oi, pil_idx, pil_idx] += err_var_[tx, st, oi, ii]

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, max_num_pil]
        a_mat = pil_cov_mat_.permute(5, 6, 7, 0, 1, 2, 3, 4)

        # Compute B matrix
        # [num_tx, num_streams_per_tx, outer_dim_size, max_num_pil, inner_dim_size]
        b_mat = self._b_mat.to(device)
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, inner_dim_size]
        b_mat = expand_to_rank(b_mat, 8, 0)
        b_mat = b_mat.expand(batch_size, num_rx, num_rx_ant, -1, -1, -1, -1, -1)

        # Solve least squares: a_mat @ ext_mat = b_mat
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, max_num_pil, inner_dim_size]
        # Use direct Cholesky solve for better numerical stability (avoids squaring
        # condition number like matrix_pinv does with Gram matrix A^H @ A)
        
        # Add precision-dependent regularization for numerical stability at high SNR
        # (when err_var becomes very small, the matrix can become ill-conditioned)
        eps = torch.finfo(self.dtype).eps
        rcond = eps * a_mat.shape[-1]
        diag_mean = torch.diagonal(a_mat, dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).unsqueeze(-1)
        reg = rcond * diag_mean * torch.eye(a_mat.shape[-1], dtype=a_mat.dtype, device=device)
        a_mat_reg = a_mat + reg
        
        # Cholesky solve: a_mat @ X = b_mat
        chol, _ = torch.linalg.cholesky_ex(a_mat_reg, check_errors=False)
        y = torch.linalg.solve_triangular(chol, b_mat, upper=False)
        ext_mat = torch.linalg.solve_triangular(chol.mH, y, upper=True)

        # Conjugate transpose
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, max_num_pil]
        ext_mat = ext_mat.mH

        # Scatter to expand columns from max_num_pil to inner_dim_size
        # Using the pil_loc indices to place values correctly
        # [num_tx, num_streams_per_tx, outer_dim_size, inner_dim_size, max_num_pil,
        #  batch_size, num_rx, num_rx_ant]
        ext_mat_t = ext_mat.permute(3, 4, 5, 6, 7, 0, 1, 2)

        # Create output tensor with extra padding row/column
        ext_mat_full = torch.zeros(
            num_tx,
            num_tx_stream,
            outer_dim_size,
            inner_dim_size + 1,
            inner_dim_size + 1,
            batch_size,
            num_rx,
            num_rx_ant,
            dtype=self.cdtype,
            device=device,
        )

        # Scatter values according to pilot locations using fully vectorized advanced indexing
        # This avoids Python loops which cause slow compilation and execution
        # pil_loc_row and pil_loc_col were moved to device at the beginning of this method

        # Create broadcast-compatible index tensors for the first 3 dimensions
        tx_idx = torch.arange(num_tx, device=device)[:, None, None, None, None]
        st_idx = torch.arange(num_tx_stream, device=device)[None, :, None, None, None]
        oi_idx = torch.arange(outer_dim_size, device=device)[None, None, :, None, None]

        # Single vectorized scatter using advanced indexing
        # All indices broadcast to [num_tx, num_tx_stream, outer, inner, max_pil]
        # The trailing dims [batch, num_rx, num_rx_ant] are preserved automatically
        ext_mat_full[tx_idx, st_idx, oi_idx, pil_loc_row, pil_loc_col] = ext_mat_t

        # Remove padding and transpose back
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size, inner_dim_size]
        ext_mat = ext_mat_full[:, :, :, :inner_dim_size, :inner_dim_size, :, :, :]
        ext_mat = ext_mat.permute(5, 6, 7, 0, 1, 2, 3, 4)

        ################################################
        # Apply interpolation over the inner dimension
        ################################################

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        h_hat_out = (ext_mat @ h_hat.unsqueeze(-1)).squeeze(-1)

        ##############################
        # Compute the error variances
        ##############################

        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #  outer_dim_size, inner_dim_size]
        cov_mat = self._cov_mat.to(device)
        cov_mat = expand_to_rank(cov_mat, 8, 0)
        err_var_out = torch.diagonal(cov_mat, dim1=-2, dim2=-1)
        err_var_mat = self._err_var_mat.to(device)
        err_var_mat = expand_to_rank(err_var_mat, 8, 0)
        # Transpose (NOT conjugate transpose) to swap last two dimensions
        err_var_mat_t = err_var_mat.transpose(-1, -2)
        err_var_out = err_var_out - (ext_mat * err_var_mat_t).sum(dim=-1)
        err_var_out = err_var_out.real
        err_var_out = torch.maximum(err_var_out, rzero)

        #####################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        #####################################
        if not self._last_step:
            # Conjugate transpose of LMMSE matrix
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            ext_mat_h = ext_mat.transpose(-1, -2).conj()

            # First part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size, inner_dim_size]
            h_hat_var_1 = cov_mat @ ext_mat_h
            h_hat_var_1 = h_hat_var_1.transpose(-1, -2)
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var_1 = (ext_mat * h_hat_var_1).sum(dim=-1)

            # Second part of the estimate covariance
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_old_c = err_var_old.to(self.cdtype).unsqueeze(-1)
            h_hat_var_2 = err_var_old_c * ext_mat_h
            h_hat_var_2 = h_hat_var_2.transpose(-1, -2)
            h_hat_var_2 = (ext_mat * h_hat_var_2).sum(dim=-1)

            # Variance of h_hat
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_var = h_hat_var_1 + h_hat_var_2

            # Scaling factor
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_c = err_var_out.to(self.cdtype)
            h_var = torch.diagonal(cov_mat, dim1=-2, dim2=-1)
            denom = h_hat_var + h_var - err_var_c
            # Use divide_no_nan equivalent
            s = torch.where(
                denom.abs() > 1e-12, 2.0 * h_var / denom, torch.zeros_like(denom)
            )

            # Apply scaling to estimate
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            h_hat_out = s * h_hat_out

            # Updated variance (using complex arithmetic, then take real part)
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #  outer_dim_size, inner_dim_size]
            err_var_out_c = (
                s * (s - 1.0) * h_hat_var + (1.0 - s) * h_var + s * err_var_c
            )
            err_var_out = err_var_out_c.real
            err_var_out = torch.maximum(err_var_out, rzero)

        return h_hat_out, err_var_out


class SpatialChannelFilter(Object):
    r"""Implements linear minimum mean square error (LMMSE) smoothing.

    We consider the following model:

    .. math::

        \mathbf{y} = \mathbf{h} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated
    with covariance matrix
    :math:`\mathbb{E}\left[ \mathbf{h} \mathbf{h}^{\mathsf{H}} \right] = \mathbf{R}`,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`.

    The channel estimate :math:`\hat{\mathbf{h}}` is computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{A} \mathbf{y}

    where

    .. math::

        \mathbf{A} = \mathbf{R} \left( \mathbf{R} + N_0 \mathbf{I}_M \right)^{-1}

    where :math:`\mathbf{I}_M` is the :math:`M \times M` identity matrix.
    The estimation error is:

    .. math::

        \tilde{h} = \mathbf{h} - \hat{\mathbf{h}}

    The error variances

    .. math::

             \sigma^2_i = \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right], 0 \leq i \leq M-1

    are the diagonal elements of

    .. math::

        \mathbb{E}\left[\mathbf{\tilde{h}} \mathbf{\tilde{h}}^{\mathsf{H}} \right] = \mathbf{R} - \mathbf{A}\mathbf{R}.

    :param cov_mat: Spatial covariance matrix of the channel
    :param last_step: Set to `True` if this is the last interpolation step.
        Otherwise, set to `False`.
        If `True`, the output is scaled to ensure its variance is as expected
        by the following interpolation step.

    :input h_hat: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.complex`.
        Channel estimates.
    :input err_var: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.float`.
        Channel estimation error variances.

    :output h_hat: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.complex`.
        Channel estimates smoothed across the spatial dimension.
    :output err_var: [batch_size, num_rx, num_tx, num_streams_per_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant], `torch.float`.
        The channel estimation error variances of the smoothed channel
        estimates.
    """

    def __init__(self, cov_mat: torch.Tensor, last_step: bool) -> None:
        if cov_mat.dtype == torch.complex64:
            precision = "single"
        elif cov_mat.dtype == torch.complex128:
            precision = "double"
        else:
            raise TypeError("`cov_mat` dtype must be complex64 or complex128")
        super().__init__(precision=precision)

        # Register as buffers for CUDAGraph compatibility (on config.device to avoid DeviceCopy)
        self.register_buffer(
            "_rzero", torch.zeros((), dtype=self.dtype, device=config.device)
        )
        self.register_buffer("_cov_mat", cov_mat.to(config.device))
        self._last_step = last_step

        # Indices for adding to diagonal
        num_rx_ant = cov_mat.shape[0]
        add_diag_indices = [[rxa, rxa] for rxa in range(num_rx_ant)]
        self.register_buffer(
            "_add_diag_indices",
            torch.tensor(add_diag_indices, dtype=torch.int64, device=config.device),
        )

    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = h_hat.device

        # Move internal buffers to input device to avoid device mismatch errors
        cov_mat = self._cov_mat.to(device)
        rzero = self._rzero.to(device)

        # [..., num_rx_ant]
        err_var = err_var.to(self.cdtype)
        # Keep track of the previous estimation error variances for later use
        err_var_old = err_var

        # [num_rx_ant, num_rx_ant]
        cov_mat_t = cov_mat.T

        ##########################################
        # Compute LMMSE matrix
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat_expanded = expand_to_rank(cov_mat, h_hat.dim() + 1, 0)

        # Adding the error variances to the diagonal
        # [..., num_rx_ant, num_rx_ant]
        lmmse_mat = cov_mat_expanded + torch.diag_embed(err_var)

        # Add precision-dependent regularization for numerical stability at high SNR
        # (when err_var becomes very small, the matrix can become ill-conditioned)
        eps = torch.finfo(self.dtype).eps
        rcond = eps * lmmse_mat.shape[-1]
        diag_mean = torch.diagonal(lmmse_mat, dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).unsqueeze(-1)
        reg = rcond * diag_mean * torch.eye(lmmse_mat.shape[-1], dtype=lmmse_mat.dtype, device=device)
        lmmse_mat = lmmse_mat + reg

        # [..., num_rx_ant, num_rx_ant]
        # Use cholesky_ex with check_errors=False and solve_triangular for better
        # CUDA graph compatibility (avoids synchronization in cholesky_solve)
        l, info = torch.linalg.cholesky_ex(lmmse_mat, check_errors=False)
        # Solve L L^H X = B via two triangular solves:
        # 1) L Y = B (lower triangular)
        # 2) L^H X = Y (upper triangular on conjugate transpose)
        y = torch.linalg.solve_triangular(l, cov_mat_expanded, upper=False, left=True)
        lmmse_mat = torch.linalg.solve_triangular(l.mH, y, upper=True, left=True)
        lmmse_mat = lmmse_mat.transpose(-1, -2).conj()

        ##########################################
        # Apply smoothing
        ##########################################

        # [..., num_rx_ant]
        h_hat = (lmmse_mat @ h_hat.unsqueeze(-1)).squeeze(-1)

        ##########################################
        # Compute the estimation error variances
        ##########################################

        # [..., num_rx_ant, num_rx_ant]
        cov_mat_t_expanded = expand_to_rank(cov_mat_t, lmmse_mat.dim(), 0)
        # [..., num_rx_ant]
        err_var_out = (cov_mat_t_expanded * lmmse_mat).sum(dim=-1)
        # [..., num_rx_ant]
        err_var_out = torch.diagonal(cov_mat_expanded, dim1=-2, dim2=-1) - err_var_out
        err_var_out = err_var_out.real
        err_var_out = torch.maximum(err_var_out, rzero)

        ##########################################
        # If this is *not* the last
        # interpolation step, scales the
        # input `h_hat` to ensure
        # it has the variance expected by the
        # next interpolation step.
        #
        # The error variance also `err_var`
        # is updated accordingly.
        ##########################################
        if not self._last_step:
            # Conjugate transpose of the LMMSE matrix
            # [..., num_rx_ant, num_rx_ant]
            lmmse_mat_h = lmmse_mat.transpose(-1, -2).conj()

            # First part of the estimate covariance
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_1 = cov_mat_expanded @ lmmse_mat_h
            h_hat_var_1 = h_hat_var_1.transpose(-1, -2)
            # [..., num_rx_ant]
            h_hat_var_1 = (lmmse_mat * h_hat_var_1).sum(dim=-1)

            # Second part of the estimate covariance
            # [..., num_rx_ant, 1]
            err_var_old_expanded = err_var_old.unsqueeze(-1)
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = err_var_old_expanded * lmmse_mat_h
            # [..., num_rx_ant, num_rx_ant]
            h_hat_var_2 = h_hat_var_2.transpose(-1, -2)
            # [..., num_rx_ant]
            h_hat_var_2 = (lmmse_mat * h_hat_var_2).sum(dim=-1)

            # Variance of h_hat
            # [..., num_rx_ant]
            h_hat_var = h_hat_var_1 + h_hat_var_2

            # Scaling factor
            # [..., num_rx_ant]
            err_var_c = err_var_out.to(self.cdtype)
            h_var = torch.diagonal(cov_mat_expanded, dim1=-2, dim2=-1)
            denom = h_hat_var + h_var - err_var_c
            s = torch.where(
                denom.abs() > 1e-12, 2.0 * h_var / denom, torch.zeros_like(denom)
            )

            # Apply scaling to estimate
            # [..., num_rx_ant]
            h_hat = s * h_hat

            # Updated variance (using complex arithmetic, then take real part)
            # [..., num_rx_ant]
            err_var_out_c = (
                s * (s - 1.0) * h_hat_var + (1.0 - s) * h_var + s * err_var_c
            )
            err_var_out = err_var_out_c.real
            err_var_out = torch.maximum(err_var_out, rzero)

        return h_hat, err_var_out


class LMMSEInterpolator(BaseChannelInterpolator):
    r"""LMMSE interpolation on a resource grid with optional spatial smoothing.

    This class computes for each element of an OFDM resource grid
    a channel estimate and error variance
    through linear minimum mean square error (LMMSE) interpolation/smoothing.
    It is assumed that the measurements were taken at the nonzero positions
    of a :class:`~sionna.phy.ofdm.PilotPattern`.

    Depending on the value of ``order``, the interpolation is carried out
    across time (t), i.e., OFDM symbols, frequency (f), i.e., subcarriers,
    and optionally space (s), i.e., receive antennas, in any desired order.

    For simplicity, we describe the underlying algorithm assuming that
    interpolation across the sub-carriers is performed first, followed by
    interpolation across OFDM symbols, and finally by spatial smoothing across
    receive antennas.
    The algorithm is similar if interpolation and/or smoothing are performed in
    a different order.
    For clarity, antenna indices are omitted when describing frequency and time
    interpolation, as the same process is applied to all the antennas.

    The input ``h_hat`` is first reshaped to a resource grid
    :math:`\hat{\mathbf{H}} \in \mathbb{C}^{N \times M}`, by scattering the
    channel estimates at pilot locations according to the ``pilot_pattern``.
    :math:`N` denotes the number of OFDM symbols and :math:`M` the number of
    sub-carriers.

    The first pass consists in interpolating across the sub-carriers:

    .. math::
        \hat{\mathbf{h}}_n^{(1)} = \mathbf{A}_n \hat{\mathbf{h}}_n

    where :math:`1 \leq n \leq N` is the OFDM symbol index and
    :math:`\hat{\mathbf{h}}_n` is the :math:`n^{\text{th}}` (transposed) row
    of :math:`\hat{\mathbf{H}}`.
    :math:`\mathbf{A}_n` is the :math:`M \times M` matrix such that:

    .. math::
        \mathbf{A}_n = \bar{\mathbf{A}}_n \mathbf{\Pi}_n^\intercal

    where

    .. math::
        \bar{\mathbf{A}}_n = \underset{\mathbf{Z} \in \mathbb{C}^{M \times K_n}}{\text{argmin}} \left\lVert \mathbf{Z}\left( \mathbf{\Pi}_n^\intercal \mathbf{R^{(f)}} \mathbf{\Pi}_n + \mathbf{\Sigma}_n \right) - \mathbf{R^{(f)}} \mathbf{\Pi}_n \right\rVert_{\text{F}}^2

    and :math:`\mathbf{R^{(f)}}` is the :math:`M \times M` channel frequency
    covariance matrix,
    :math:`\mathbf{\Pi}_n` the :math:`M \times K_n` matrix that spreads
    :math:`K_n` values to a vector of size :math:`M` according to the
    ``pilot_pattern`` for the :math:`n^{\text{th}}` OFDM symbol,
    and :math:`\mathbf{\Sigma}_n \in \mathbb{R}^{K_n \times K_n}` is the
    channel estimation error covariance built from ``err_var`` and assumed to
    be diagonal.
    Computation of :math:`\bar{\mathbf{A}}_n` is done using an algorithm based
    on complete orthogonal decomposition.
    This is done to avoid matrix inversion for badly conditioned covariance
    matrices.

    The channel estimation error variances after the first interpolation pass
    are computed as

    .. math::
        \mathbf{\Sigma}^{(1)}_n = \text{diag} \left( \mathbf{R^{(f)}} - \mathbf{A}_n \mathbf{\Xi}_n \mathbf{R^{(f)}} \right)

    where :math:`\mathbf{\Xi}_n` is the diagonal matrix of size
    :math:`M \times M` that zeros the columns corresponding to sub-carriers
    not carrying any pilots.
    Note that interpolation is not performed for OFDM symbols which do not
    carry pilots.

    **Remark**: The interpolation matrix differs across OFDM symbols as
    different OFDM symbols may carry pilots on different sub-carriers and/or
    have different estimation error variances.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next interpolation step, and the
    error variances are updated accordingly:

    .. math::
        \begin{aligned}
            \left[\hat{\mathbf{h}}_n^{(2)}\right]_m &= s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m\\
            \left[\mathbf{\Sigma}^{(2)}_n\right]_{m,m}  &= s_{n,m}\left( s_{n,m}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m} + \left( 1 - s_{n,m} \right) \left[\mathbf{R^{(f)}}\right]_{m,m} + s_{n,m} \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m}
        \end{aligned}

    where the scaling factor :math:`s_{n,m}` is such that:

    .. math::
        \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}_n^{(1)}\right]_m \right\rvert^2 \right\} = \left[\mathbf{R^{(f)}}\right]_{m,m} +  \mathbb{E} \left\{ \left\lvert s_{n,m} \left[\hat{\mathbf{h}}^{(1)}_n\right]_m - \left[\mathbf{h}_n\right]_m \right\rvert^2 \right\}

    which leads to:

    .. math::
        \begin{aligned}
            s_{n,m} &= \frac{2 \left[\mathbf{R^{(f)}}\right]_{m,m}}{\left[\mathbf{R^{(f)}}\right]_{m,m} - \left[\mathbf{\Sigma}^{(1)}_n\right]_{m,m} + \left[\hat{\mathbf{\Sigma}}^{(1)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(1)}_n &= \mathbf{A}_n \mathbf{R^{(f)}} \mathbf{A}_n^{\mathrm{H}}.
        \end{aligned}

    The second pass consists in interpolating across the OFDM symbols:

    .. math::
        \hat{\mathbf{h}}_m^{(3)} = \mathbf{B}_m \tilde{\mathbf{h}}^{(2)}_m

    where :math:`1 \leq m \leq M` is the sub-carrier index and
    :math:`\tilde{\mathbf{h}}^{(2)}_m` is the :math:`m^{\text{th}}` column of

    .. math::
        \hat{\mathbf{H}}^{(2)} = \begin{bmatrix}
                                    {\hat{\mathbf{h}}_1^{(2)}}^\intercal\\
                                    \vdots\\
                                    {\hat{\mathbf{h}}_N^{(2)}}^\intercal
                                 \end{bmatrix}

    and :math:`\mathbf{B}_m` is the :math:`N \times N` interpolation LMMSE
    matrix:

    .. math::
        \mathbf{B}_m = \bar{\mathbf{B}}_m \tilde{\mathbf{\Pi}}_m^\intercal

    where

    .. math::
        \bar{\mathbf{B}}_m = \underset{\mathbf{Z} \in \mathbb{C}^{N \times L_m}}{\text{argmin}} \left\lVert \mathbf{Z} \left( \tilde{\mathbf{\Pi}}_m^\intercal \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m + \tilde{\mathbf{\Sigma}}^{(2)}_m \right) -  \mathbf{R^{(t)}}\tilde{\mathbf{\Pi}}_m \right\rVert_{\text{F}}^2

    where :math:`\mathbf{R^{(t)}}` is the :math:`N \times N` channel time
    covariance matrix,
    :math:`\tilde{\mathbf{\Pi}}_m` the :math:`N \times L_m` matrix that
    spreads :math:`L_m` values to a vector of size :math:`N` according to the
    ``pilot_pattern`` for the :math:`m^{\text{th}}` sub-carrier,
    and :math:`\tilde{\mathbf{\Sigma}}^{(2)}_m \in \mathbb{R}^{L_m \times L_m}`
    is the diagonal matrix of channel estimation error variances
    built by gathering the error variances from
    (:math:`\mathbf{\Sigma}^{(2)}_1,\dots,\mathbf{\Sigma}^{(2)}_N`)
    corresponding to resource elements carried by the :math:`m^{\text{th}}`
    sub-carrier.
    Computation of :math:`\bar{\mathbf{B}}_m` is done using an algorithm based
    on complete orthogonal decomposition.
    This is done to avoid matrix inversion for badly conditioned covariance
    matrices.

    The resulting channel estimate for the resource grid is

    .. math::
        \hat{\mathbf{H}}^{(3)} = \left[ \hat{\mathbf{h}}_1^{(3)} \dots \hat{\mathbf{h}}_M^{(3)} \right]

    The resulting channel estimation error variances are the diagonal
    coefficients of the matrices

    .. math::
        \mathbf{\Sigma}^{(3)}_m = \mathbf{R^{(t)}} - \mathbf{B}_m \tilde{\mathbf{\Xi}}_m \mathbf{R^{(t)}}, 1 \leq m \leq M

    where :math:`\tilde{\mathbf{\Xi}}_m` is the diagonal matrix of size
    :math:`N \times N` that zeros the columns corresponding to OFDM symbols
    not carrying any pilots.

    **Remark**: The interpolation matrix differs across sub-carriers as
    different sub-carriers may have different estimation error variances
    computed by the first pass.
    However, all sub-carriers carry at least one channel estimate as a result
    of the first pass, ensuring that a channel estimate is computed for all the
    resource elements after the second pass.

    **Remark:** LMMSE interpolation requires knowledge of the time and
    frequency covariance matrices of the channel.
    The functions :func:`~sionna.phy.ofdm.tdl_time_cov_mat`
    and :func:`~sionna.phy.ofdm.tdl_freq_cov_mat` compute the expected time
    and frequency covariance matrices, respectively, for the
    :class:`~sionna.phy.channel.tr38901.TDL` channel models.

    Scaling of the estimates is then performed to ensure that their
    variances match the ones expected by the next smoothing step, and the
    error variances are updated accordingly:

    .. math::
        \begin{aligned}
            \left[\hat{\mathbf{h}}_m^{(4)}\right]_n &= \gamma_{m,n} \left[\hat{\mathbf{h}}_m^{(3)}\right]_n\\
            \left[\mathbf{\Sigma}^{(4)}_m\right]_{n,n}  &= \gamma_{m,n}\left( \gamma_{m,n}-1 \right) \left[\hat{\mathbf{\Sigma}}^{(3)}_m\right]_{n,n} + \left( 1 - \gamma_{m,n} \right) \left[\mathbf{R^{(t)}}\right]_{n,n} + \gamma_{m,n} \left[\mathbf{\Sigma}^{(3)}_n\right]_{m,m}
        \end{aligned}

    where:

    .. math::
        \begin{aligned}
            \gamma_{m,n} &= \frac{2 \left[\mathbf{R^{(t)}}\right]_{n,n}}{\left[\mathbf{R^{(t)}}\right]_{n,n} - \left[\mathbf{\Sigma}^{(3)}_m\right]_{n,n} + \left[\hat{\mathbf{\Sigma}}^{(3)}_n\right]_{m,m}}\\
            \hat{\mathbf{\Sigma}}^{(3)}_m &= \mathbf{B}_m \mathbf{R^{(t)}} \mathbf{B}_m^{\mathrm{H}}
        \end{aligned}

    Finally, a spatial smoothing step is applied to every resource element
    carrying a channel estimate.
    For clarity, we drop the resource element indexing :math:`(n,m)`.
    We denote by :math:`L` the number of receive antennas, and by
    :math:`\mathbf{R^{(s)}}\in\mathbb{C}^{L \times L}` the spatial covariance
    matrix.

    LMMSE spatial smoothing consists in the following computations:

    .. math::
        \hat{\mathbf{h}}^{(5)} = \mathbf{C} \hat{\mathbf{h}}^{(4)}

    where

    .. math::
        \mathbf{C} = \mathbf{R^{(s)}} \left( \mathbf{R^{(s)}} + \mathbf{\Sigma}^{(4)} \right)^{-1}.

    The estimation error variances are the diagonal coefficients of

    .. math::
        \mathbf{\Sigma}^{(5)} = \mathbf{R^{(s)}} - \mathbf{C}\mathbf{R^{(s)}}

    The smoothed channel estimate :math:`\hat{\mathbf{h}}^{(5)}` and
    corresponding error variances
    :math:`\text{diag}\left( \mathbf{\Sigma}^{(5)} \right)` are
    returned for every resource element :math:`(m,n)`.

    **Remark:** No scaling is performed after the last interpolation or
    smoothing step.

    **Remark:** All passes assume that the estimation error covariance matrix
    (:math:`\mathbf{\Sigma}`,
    :math:`\tilde{\mathbf{\Sigma}}^{(2)}`, or
    :math:`\tilde{\mathbf{\Sigma}}^{(4)}`) is diagonal, which
    may not be accurate. When this assumption does not hold, this interpolator
    is only an approximation of LMMSE interpolation.

    **Remark:** The order in which frequency interpolation, temporal
    interpolation, and, optionally, spatial smoothing are applied, is
    controlled using the ``order`` parameter.

    :param pilot_pattern: Used pilot pattern
    :param cov_mat_time: Time covariance matrix of the channel
    :param cov_mat_freq: Frequency covariance matrix of the channel
    :param cov_mat_space: Spatial covariance matrix of the channel.
        Only required if spatial smoothing is requested (see ``order``).
    :param order: Order in which to perform interpolation and optional
        smoothing. For example, ``"t-f-s"`` means that interpolation across
        the OFDM symbols is performed first (``"t"``: time), followed by
        interpolation across the sub-carriers (``"f"``: frequency), and
        finally smoothing across the receive antennas (``"s"``: space).
        Similarly, ``"f-t"`` means interpolation across the sub-carriers
        followed by interpolation across the OFDM symbols and no spatial
        smoothing. The spatial covariance matrix (``cov_mat_space``) is
        only required when spatial smoothing is requested. Time and frequency
        interpolation are not optional to ensure that a channel estimate is
        computed for all resource elements.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates for the pilot-carrying resource elements.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Channel estimation error variances for the pilot-carrying resource
        elements.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], `torch.complex`.
        Channel estimates across the entire resource grid for all
        transmitters and streams.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.ofdm import (
            ResourceGrid, LSChannelEstimator, LMMSEInterpolator,
            tdl_freq_cov_mat, tdl_time_cov_mat
        )

        rg = ResourceGrid(num_ofdm_symbols=14,
                          fft_size=64,
                          subcarrier_spacing=30e3,
                          num_tx=1,
                          num_streams_per_tx=1,
                          pilot_pattern="kronecker",
                          pilot_ofdm_symbol_indices=[2, 11])

        # Compute covariance matrices
        cov_mat_freq = tdl_freq_cov_mat("A", 30e3, 64, 100e-9)
        cov_mat_time = tdl_time_cov_mat("A", 3.0, 3.5e9, 35.7e-6, 14)

        # Create LMMSE interpolator
        interpolator = LMMSEInterpolator(
            rg.pilot_pattern, cov_mat_time, cov_mat_freq, order="f-t"
        )

        # Use with LS estimator
        estimator = LSChannelEstimator(rg, interpolator=interpolator)
    """

    def __init__(
        self,
        pilot_pattern,
        cov_mat_time: torch.Tensor,
        cov_mat_freq: torch.Tensor,
        cov_mat_space: Optional[torch.Tensor] = None,
        order: str = "t-f",
    ) -> None:
        super().__init__()

        # Check the specified order
        order_list = order.split("-")
        assert 2 <= len(order_list) <= 3, "Invalid order for interpolation."
        spatial_smoothing = False
        freq_smoothing = False
        time_smoothing = False
        for o in order_list:
            assert o in ("s", "f", "t"), f"Unknown dimension {o}"
            if o == "s":
                assert (
                    not spatial_smoothing
                ), "Spatial smoothing can be specified at most once"
                spatial_smoothing = True
            elif o == "t":
                assert (
                    not time_smoothing
                ), "Temporal interpolation can be specified once only"
                time_smoothing = True
            elif o == "f":
                assert (
                    not freq_smoothing
                ), "Frequency interpolation can be specified once only"
                freq_smoothing = True

        if spatial_smoothing:
            assert (
                cov_mat_space is not None
            ), "A spatial covariance matrix is required for spatial smoothing"
        assert freq_smoothing, "Frequency interpolation is required"
        assert time_smoothing, "Time interpolation is required"

        self._order = order_list
        self._num_ofdm_symbols = pilot_pattern.num_ofdm_symbols
        self._num_effective_subcarriers = pilot_pattern.num_effective_subcarriers

        # Build pilot masks for every stream
        pilot_mask = self._build_pilot_mask(pilot_pattern)

        # Build indices for mapping channel estimates to resource grid
        num_pilots = pilot_pattern.pilots.shape[2]
        inputs_to_rg_indices, scatter_indices = self._build_inputs2rg_indices(
            pilot_mask, num_pilots
        )
        # Register scatter indices as tensor buffer for vectorized operations
        self.register_buffer(
            "_scatter_tx",
            torch.tensor(
                scatter_indices[:, 0], dtype=torch.int64, device=config.device
            ),
        )
        self.register_buffer(
            "_scatter_st",
            torch.tensor(
                scatter_indices[:, 1], dtype=torch.int64, device=config.device
            ),
        )
        self.register_buffer(
            "_scatter_p",
            torch.tensor(
                scatter_indices[:, 2], dtype=torch.int64, device=config.device
            ),
        )
        self.register_buffer(
            "_scatter_sb",
            torch.tensor(
                scatter_indices[:, 3], dtype=torch.int64, device=config.device
            ),
        )
        self.register_buffer(
            "_scatter_sc",
            torch.tensor(
                scatter_indices[:, 4], dtype=torch.int64, device=config.device
            ),
        )
        # Register as buffer for CUDAGraph compatibility (on config.device)
        self.register_buffer(
            "_inputs_to_rg_indices",
            torch.tensor(inputs_to_rg_indices, dtype=torch.int64, device=config.device),
        )

        # Build interpolators according to requested order
        interpolators = []
        for i, o in enumerate(order_list):
            last_step = i == len(order_list) - 1
            if o == "f":
                interpolator = LMMSEInterpolator1D(
                    pilot_mask, cov_mat_freq, last_step=last_step
                )
                pilot_mask = self._update_pilot_mask_interp(pilot_mask)
                err_var_mask = torch.tensor(
                    pilot_mask == 1, dtype=cov_mat_freq.real.dtype, device=config.device
                )
            elif o == "t":
                pilot_mask_t = np.transpose(pilot_mask, [0, 1, 3, 2])
                interpolator = LMMSEInterpolator1D(
                    pilot_mask_t, cov_mat_time, last_step=last_step
                )
                pilot_mask = self._update_pilot_mask_interp(pilot_mask_t)
                pilot_mask = np.transpose(pilot_mask, [0, 1, 3, 2])
                err_var_mask = torch.tensor(
                    pilot_mask == 1, dtype=cov_mat_freq.real.dtype, device=config.device
                )
            else:  # 's'
                interpolator = SpatialChannelFilter(cov_mat_space, last_step=last_step)
                err_var_mask = torch.tensor(
                    pilot_mask == 1, dtype=cov_mat_freq.real.dtype, device=config.device
                )
            interpolators.append(interpolator)
            # Register each err_var_mask as a buffer for CUDAGraph compatibility
            self.register_buffer(f"_err_var_mask_{i}", err_var_mask)

        self._interpolators = interpolators
        # Build list from registered buffers
        self._err_var_masks = [
            getattr(self, f"_err_var_mask_{i}") for i in range(len(order_list))
        ]

    def _build_pilot_mask(self, pilot_pattern) -> np.ndarray:
        """Build pilot mask indicating which REs are pilots, data, or unused."""
        mask = pilot_pattern.mask.cpu().numpy()
        pilots = pilot_pattern.pilots.cpu().numpy()
        num_tx = mask.shape[0]
        num_streams_per_tx = mask.shape[1]
        num_ofdm_symbols = mask.shape[2]
        num_effective_subcarriers = mask.shape[3]

        pilot_mask = np.zeros(
            [num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers],
            int,
        )
        for tx, st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0
            for sb, sc in itertools.product(
                range(num_ofdm_symbols), range(num_effective_subcarriers)
            ):
                if mask[tx, st, sb, sc] == 1:
                    if np.abs(pilots[tx, st, pil_index]) > 0.0:
                        pilot_mask[tx, st, sb, sc] = 1
                    else:
                        pilot_mask[tx, st, sb, sc] = 2
                    pil_index += 1

        return pilot_mask

    def _build_inputs2rg_indices(
        self, pilot_mask: np.ndarray, num_pilots: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build indices for mapping channel estimates to a resource grid."""
        num_tx = pilot_mask.shape[0]
        num_streams_per_tx = pilot_mask.shape[1]
        num_ofdm_symbols = pilot_mask.shape[2]
        num_effective_subcarriers = pilot_mask.shape[3]

        inputs_to_rg_indices = np.zeros(
            [num_tx, num_streams_per_tx, num_pilots, 4], int
        )
        # Pre-compute scatter indices as numpy array for vectorized operations
        scatter_indices_list = []

        for tx, st in itertools.product(range(num_tx), range(num_streams_per_tx)):
            pil_index = 0
            for sb, sc in itertools.product(
                range(num_ofdm_symbols), range(num_effective_subcarriers)
            ):
                if pilot_mask[tx, st, sb, sc] == 0:
                    continue
                if pilot_mask[tx, st, sb, sc] == 1:
                    inputs_to_rg_indices[tx, st, pil_index] = [tx, st, sb, sc]
                    scatter_indices_list.append([tx, st, pil_index, sb, sc])
                pil_index += 1

        scatter_indices = np.array(scatter_indices_list, dtype=np.int64)
        return inputs_to_rg_indices, scatter_indices

    def _update_pilot_mask_interp(self, pilot_mask: np.ndarray) -> np.ndarray:
        """Update pilot mask to label interpolated resource elements."""
        interpolated = np.any(pilot_mask == 1, axis=-1, keepdims=True)
        pilot_mask = np.where(interpolated, 1, pilot_mask)
        return pilot_mask

    @torch.compiler.disable  # Torchinductor doesn't support complex number codegen
    def __call__(
        self, h_hat: torch.Tensor, err_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = h_hat.shape[0]
        num_rx = h_hat.shape[1]
        num_rx_ant = h_hat.shape[2]
        num_tx = h_hat.shape[3]
        num_tx_stream = h_hat.shape[4]
        num_ofdm_symbols = self._num_ofdm_symbols
        num_effective_subcarriers = self._num_effective_subcarriers
        device = h_hat.device

        # Broadcast err_var if needed
        err_var = err_var.broadcast_to(h_hat.shape)

        # Map channel estimates to resource grid using scatter
        # Create output tensors
        h_hat_rg = torch.zeros(
            batch_size,
            num_rx,
            num_rx_ant,
            num_tx,
            num_tx_stream,
            num_ofdm_symbols,
            num_effective_subcarriers,
            dtype=h_hat.dtype,
            device=device,
        )
        err_var_rg = torch.zeros_like(h_hat_rg, dtype=err_var.dtype)

        # Scatter values to resource grid using vectorized indexing
        # Uses pre-computed tensor indices for compile-friendly operations
        # Move indices to input device to avoid device mismatch errors
        tx_idx = self._scatter_tx.to(device)  # [N]
        st_idx = self._scatter_st.to(device)  # [N]
        p_idx = self._scatter_p.to(device)  # [N]
        sb_idx = self._scatter_sb.to(device)  # [N]
        sc_idx = self._scatter_sc.to(device)  # [N]

        # Vectorized scatter: gather from h_hat and scatter to h_hat_rg
        h_hat_rg[:, :, :, tx_idx, st_idx, sb_idx, sc_idx] = h_hat[
            :, :, :, tx_idx, st_idx, p_idx
        ]
        err_var_rg[:, :, :, tx_idx, st_idx, sb_idx, sc_idx] = err_var[
            :, :, :, tx_idx, st_idx, p_idx
        ]

        h_hat = h_hat_rg
        err_var = err_var_rg

        # Apply interpolators
        for o, interp, err_var_mask in zip(
            self._order, self._interpolators, self._err_var_masks
        ):
            # Move mask to input device to avoid device mismatch
            err_var_mask = err_var_mask.to(device)
            if o == "f":
                h_hat, err_var = interp(h_hat, err_var)
                err_var_mask = expand_to_rank(err_var_mask, err_var.dim(), 0)
                err_var = err_var * err_var_mask
            elif o == "t":
                h_hat = h_hat.permute(0, 1, 2, 3, 4, 6, 5)
                err_var = err_var.permute(0, 1, 2, 3, 4, 6, 5)
                h_hat, err_var = interp(h_hat, err_var)
                h_hat = h_hat.permute(0, 1, 2, 3, 4, 6, 5)
                err_var = err_var.permute(0, 1, 2, 3, 4, 6, 5)
                err_var_mask = expand_to_rank(err_var_mask, err_var.dim(), 0)
                err_var = err_var * err_var_mask
            elif o == "s":
                h_hat = h_hat.permute(0, 1, 3, 4, 5, 6, 2)
                err_var = err_var.permute(0, 1, 3, 4, 5, 6, 2)
                h_hat, err_var = interp(h_hat, err_var)
                h_hat = h_hat.permute(0, 1, 6, 2, 3, 4, 5)
                err_var = err_var.permute(0, 1, 6, 2, 3, 4, 5)
                err_var_mask = expand_to_rank(err_var_mask, err_var.dim(), 0)
                err_var = err_var * err_var_mask

        return h_hat, err_var


#######################################################
# Utilities
#######################################################


def tdl_freq_cov_mat(
    model: str,
    subcarrier_spacing: float,
    fft_size: int,
    delay_spread: float,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Compute the frequency covariance matrix of a
    :class:`~sionna.phy.channel.tr38901.TDL` channel model.

    The channel frequency covariance matrix :math:`\mathbf{R}^{(f)}` of a TDL
    channel model is

    .. math::
        \mathbf{R}^{(f)}_{u,v} = \sum_{\ell=1}^L P_\ell e^{-j 2 \pi \tau_\ell \Delta_f (u-v)}, 1 \leq u,v \leq M

    where :math:`M` is the FFT size, :math:`L` is the number of paths for the
    selected TDL model, :math:`P_\ell` and :math:`\tau_\ell` are the average
    power and delay for the :math:`\ell^{\text{th}}` path, respectively, and
    :math:`\Delta_f` is the sub-carrier spacing.

    :param model: TDL model (``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"E"``)
    :param subcarrier_spacing: Sub-carrier spacing [Hz]
    :param fft_size: FFT size
    :param delay_spread: Delay spread [s]
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.ofdm import tdl_freq_cov_mat

        cov_mat = tdl_freq_cov_mat("A", 30e3, 64, 100e-9)
        print(cov_mat.shape)
        # torch.Size([64, 64])
    """
    if precision is None:
        precision = Config.precision
    cdtype = torch.complex64 if precision == "single" else torch.complex128

    # Load the power delay profile
    assert model in ("A", "B", "C", "D", "E"), "Invalid TDL model"
    parameters_fname = f"TDL-{model}.json"
    source = files(models).joinpath(parameters_fname)
    with open(source) as parameter_file:
        params = json.load(parameter_file)

    los = bool(params["los"])
    delays = np.array(params["delays"]) * delay_spread
    mean_powers = np.power(10.0, np.array(params["powers"]) / 10.0)

    if los:
        mean_powers[0] = mean_powers[0] + mean_powers[1]
        mean_powers = np.concatenate([mean_powers[:1], mean_powers[2:]], axis=0)
        delays = delays[1:]

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    # Build frequency covariance matrix
    n = np.arange(fft_size)
    p = -2.0 * np.pi * subcarrier_spacing * n
    p = np.expand_dims(p, axis=0)
    delays = np.expand_dims(delays, axis=1)
    p = p * delays
    p = np.exp(1j * p)
    p = np.expand_dims(p, axis=-1)
    cov_mat = np.matmul(p, np.transpose(np.conj(p), [0, 2, 1]))
    mean_powers = np.expand_dims(mean_powers, axis=(1, 2))
    cov_mat = np.sum(mean_powers * cov_mat, axis=0)

    return torch.tensor(cov_mat, dtype=cdtype)


def tdl_time_cov_mat(
    model: str,
    speed: float,
    carrier_frequency: float,
    ofdm_symbol_duration: float,
    num_ofdm_symbols: int,
    los_angle_of_arrival: float = PI / 4.0,
    precision: Optional[Precision] = None,
) -> torch.Tensor:
    r"""Compute the time covariance matrix of a
    :class:`~sionna.phy.channel.tr38901.TDL` channel model.

    For non-line-of-sight (NLoS) model, the channel time covariance matrix
    :math:`\mathbf{R^{(t)}}` of a TDL channel model is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = J_0 \left( \nu \Delta_t \left( u-v \right) \right)

    where :math:`J_0` is the zero-order Bessel function of the first kind,
    :math:`\Delta_t` the duration of an OFDM symbol, and :math:`\nu` the
    Doppler spread defined by

    .. math::
        \nu = 2 \pi \frac{v}{c} f_c

    where :math:`v` is the movement speed, :math:`c` the speed of light, and
    :math:`f_c` the carrier frequency.

    For line-of-sight (LoS) channel models, the channel time covariance matrix
    is

    .. math::
        \mathbf{R^{(t)}}_{u,v} = P_{\text{NLoS}} J_0 \left( \nu \Delta_t \left( u-v \right) \right) + P_{\text{LoS}}e^{j \nu \Delta_t \left( u-v \right) \cos{\alpha_{\text{LoS}}}}

    where :math:`\alpha_{\text{LoS}}` is the angle-of-arrival for the LoS
    path, :math:`P_{\text{NLoS}}` the total power of NLoS paths, and
    :math:`P_{\text{LoS}}` the power of the LoS path. The power delay profile
    is assumed to have unit power, i.e.,
    :math:`P_{\text{NLoS}} + P_{\text{LoS}} = 1`.

    :param model: TDL model (``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"E"``)
    :param speed: Speed [m/s]
    :param carrier_frequency: Carrier frequency [Hz]
    :param ofdm_symbol_duration: Duration of an OFDM symbol [s]
    :param num_ofdm_symbols: Number of OFDM symbols
    :param los_angle_of_arrival: Angle-of-arrival for LoS path [radian].
        Only used with LoS models.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.ofdm import tdl_time_cov_mat

        cov_mat = tdl_time_cov_mat("A", 3.0, 3.5e9, 35.7e-6, 14)
        print(cov_mat.shape)
        # torch.Size([14, 14])
    """
    if precision is None:
        precision = Config.precision
    cdtype = torch.complex64 if precision == "single" else torch.complex128

    # Doppler spread
    doppler_spread = 2.0 * PI * speed / SPEED_OF_LIGHT * carrier_frequency

    # Load the power delay profile
    assert model in ("A", "B", "C", "D", "E"), "Invalid TDL model"
    parameters_fname = f"TDL-{model}.json"
    source = files(models).joinpath(parameters_fname)
    with open(source) as parameter_file:
        params = json.load(parameter_file)

    los = bool(params["los"])
    mean_powers = np.power(10.0, np.array(params["powers"]) / 10.0)

    # Normalize the PDP
    norm_factor = np.sum(mean_powers)
    mean_powers = mean_powers / norm_factor

    if los:
        los_power = mean_powers[0]
        nlos_power = np.sum(mean_powers[1:])
    else:
        nlos_power = np.sum(mean_powers)

    # Build time covariance matrix
    indices = np.arange(num_ofdm_symbols)
    s1 = np.expand_dims(indices, axis=1)
    s2 = np.expand_dims(indices, axis=0)
    exp = doppler_spread * ofdm_symbol_duration * (s1 - s2)
    cov_mat_nlos = jv(0.0, exp) * nlos_power

    if los:
        cov_mat_los = np.exp(1j * exp * np.cos(los_angle_of_arrival)) * los_power
        cov_mat = cov_mat_nlos + cov_mat_los
    else:
        cov_mat = cov_mat_nlos

    return torch.tensor(cov_mat, dtype=cdtype)
