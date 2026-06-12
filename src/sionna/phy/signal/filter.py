#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks implementing filters"""

from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.config import Precision
from .utils import convolve, empirical_aclr
from .window import Window, HannWindow, HammingWindow, BlackmanWindow

__all__ = [
    "Filter",
    "RaisedCosineFilter",
    "RootRaisedCosineFilter",
    "SincFilter",
    "CustomFilter",
]


class Filter(Block):
    r"""Abstract class defining a filter of ``length`` K which can be applied to an input ``x`` of length N.

    The filter length K is equal to the filter span in symbols
    (``span_in_symbols``) multiplied by the oversampling factor
    (``samples_per_symbol``). If this product is even, a value of one will
    be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    :param span_in_symbols: Filter span as measured by the number of symbols
    :param samples_per_symbol: Number of samples per symbol, i.e., the
        oversampling factor
    :param window: Window that is applied to the filter coefficients.
        Can be `None`, a :class:`~sionna.phy.signal.Window` instance, or one of
        ``"hann"``, ``"hamming"``, ``"blackman"``.
    :param normalize: If `True`, the filter is normalized to have unit
        power. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        Input to which the filter is applied along the last dimension.

    :input padding: "full" (default) | "valid" | "same".
        Padding mode for convolving ``x`` and the filter.

    :input conjugate: `bool`, (default `False`).
        If `True`, the complex conjugate of the filter is applied.

    :output y: [..., M], `torch.complex` or `torch.float`.
        Filtered input. The length M depends on the ``padding``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import RootRaisedCosineFilter

        rrc = RootRaisedCosineFilter(span_in_symbols=8, samples_per_symbol=4, beta=0.35)
        x = torch.randn(32, 100)
        y = rrc(x, padding="same")
        print(y.shape)
        # torch.Size([32, 100])
    """

    def __init__(
        self,
        span_in_symbols: int,
        samples_per_symbol: int,
        window: Optional[Union[Window, str]] = None,
        normalize: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        assert span_in_symbols > 0, "span_in_symbols must be positive"
        self._span_in_symbols = span_in_symbols

        assert samples_per_symbol > 0, "samples_per_symbol must be positive"
        self._samples_per_symbol = samples_per_symbol

        self.window = window

        assert isinstance(normalize, bool), "normalize must be bool"
        self._normalize = normalize

        self.register_buffer("_coefficients", None, persistent=False)

    @property
    def span_in_symbols(self) -> int:
        """Filter span in symbols"""
        return self._span_in_symbols

    @property
    def samples_per_symbol(self) -> int:
        """Number of samples per symbol, i.e., the oversampling factor"""
        return self._samples_per_symbol

    @property
    def length(self) -> int:
        """Filter length in samples"""
        length = self._span_in_symbols * self._samples_per_symbol
        length = 2 * (length // 2) + 1  # Force length to be the next odd number
        return length

    @property
    def window(self) -> Optional[Window]:
        """Get/set window function applied to filter coefficients"""
        return self._window

    @window.setter
    def window(self, value: Optional[Union[Window, str]]) -> None:
        if isinstance(value, str):
            if value == "hann":
                self._window = HannWindow(precision=self.precision, device=self.device)
            elif value == "hamming":
                self._window = HammingWindow(
                    precision=self.precision, device=self.device
                )
            elif value == "blackman":
                self._window = BlackmanWindow(
                    precision=self.precision, device=self.device
                )
            else:
                raise AssertionError("Invalid window type")
        elif isinstance(value, Window) or value is None:
            self._window = value
        else:
            raise AssertionError("Invalid window type")

        if value is not None:
            assert (
                self._window.precision == self.precision
            ), "Window and Filter must have the same precision."
            # Run window once to initialize coefficients
            self._window(
                torch.ones([self.length], dtype=self.cdtype, device=self.device)
            )

    @property
    def normalize(self) -> bool:
        """If `True` the filter is normalized to have unit power"""
        return self._normalize

    @property
    def coefficients(self) -> torch.Tensor:
        """Set/get raw filter coefficients"""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, v: Union[torch.Tensor, np.ndarray]) -> None:
        is_parameter = False
        requires_grad = False
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        else:
            is_parameter = isinstance(v, torch.nn.Parameter)
            requires_grad = v.requires_grad
            # Preserve gradient if already a tensor with requires_grad
            # Only convert if dtype or device differs
            target_dtype = self.cdtype if v.is_complex() else self.dtype
            if v.dtype != target_dtype or str(v.device) != self.device:
                v = v.to(dtype=target_dtype, device=self.device)
        self._parameters.pop("_coefficients", None)
        self._buffers.pop("_coefficients", None)
        if is_parameter:
            if isinstance(v, torch.nn.Parameter):
                self._coefficients = v
            else:
                self._coefficients = torch.nn.Parameter(
                    v, requires_grad=requires_grad
                )
        else:
            self.register_buffer("_coefficients", v, persistent=False)

    @property
    def sampling_times(self) -> np.ndarray:
        """Sampling times in multiples of the symbol duration"""
        n_min = -(self.length // 2)
        n_max = n_min + self.length
        t = np.arange(n_min, n_max, dtype=np.float32)
        t /= self.samples_per_symbol
        return t

    def show(
        self,
        response: Literal["impulse", "magnitude"] = "impulse",
        scale: Literal["lin", "db"] = "lin",
    ) -> None:
        r"""Plot the impulse or magnitude response.

        Plots the impulse response (time domain) or magnitude response
        (frequency domain) of the filter.

        For the computation of the magnitude response, a minimum DFT size
        of 1024 is assumed which is obtained through zero padding of
        the filter coefficients in the time domain.

        :param response: Desired response type.
            Must be ``"impulse"`` (default) or ``"magnitude"``.
        :param scale: y-scale of the magnitude response.
            Can be ``"lin"`` (i.e., linear) or ``"db"`` (i.e., Decibel).
        """
        assert response in ["impulse", "magnitude"], "Invalid response"

        h = self.coefficients

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = torch.sum(torch.abs(h) ** 2)
            h = h / torch.sqrt(energy)

        h_np = h.detach().cpu().numpy()

        if response == "impulse":
            plt.figure(figsize=(12, 6))
            plt.plot(self.sampling_times, np.real(h_np))
            if self.coefficients.is_complex():
                plt.plot(self.sampling_times, np.imag(h_np))
                plt.legend(["Real part", "Imaginary part"])
            plt.title("Impulse response")
            plt.grid()
            plt.xlabel(r"Normalized time $(t/T)$")
            plt.ylabel(r"$h(t)$")
            plt.xlim(self.sampling_times[0], self.sampling_times[-1])
        else:
            assert scale in ["lin", "db"], "Invalid scale"
            fft_size = max(1024, h.shape[-1])
            h_fft = np.fft.fft(h_np, fft_size)
            h_fft = np.fft.fftshift(h_fft)
            h_fft = np.abs(h_fft)
            plt.figure(figsize=(12, 6))
            if scale == "db":
                h_fft = np.maximum(h_fft, 1e-10)
                h_fft = 10 * np.log10(h_fft)
                plt.ylabel(r"$|H(f)|$ (dB)")
            else:
                plt.ylabel(r"$|H(f)|$")
            f = np.linspace(
                -self._samples_per_symbol / 2, self._samples_per_symbol / 2, fft_size
            )
            plt.plot(f, h_fft)
            plt.title("Magnitude response")
            plt.grid()
            plt.xlabel(r"Normalized frequency $(f/W)$")
            plt.xlim(f[0], f[-1])

    @property
    def aclr(self) -> torch.Tensor:
        """`torch.float` -- ACLR of the filter in linear scale.

        This ACLR corresponds to what one would obtain from using
        this filter as pulse shaping filter on an i.i.d. sequence of symbols.
        The in-band is assumed to range from [-0.5, 0.5] in normalized
        frequency.
        """
        h = self.coefficients

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = torch.sum(torch.abs(h) ** 2)
            h = h / torch.sqrt(energy)

        fft_size = 1024
        n = fft_size - h.shape[-1]
        z = torch.zeros([n], dtype=h.dtype, device=h.device)
        c = torch.cat([h, z], dim=-1).to(self.cdtype)

        return empirical_aclr(
            c, oversampling=self._samples_per_symbol, precision=self.precision
        )

    def call(
        self,
        x: torch.Tensor,
        padding: Literal["full", "same", "valid"] = "full",
        conjugate: bool = False,
    ) -> torch.Tensor:
        h = self.coefficients

        # Apply window
        if self.window is not None:
            h = self._window(h)

        # Ensure unit L2-norm of the coefficients
        if self.normalize:
            energy = torch.sum(torch.abs(h) ** 2)
            h = h / torch.sqrt(energy)

        # (Optionally) compute the complex conjugate
        if conjugate and h.is_complex():
            h = torch.conj(h)

        y = convolve(x, h, padding=padding, precision=self.precision)
        return y


class RaisedCosineFilter(Filter):
    r"""Block for applying a raised-cosine filter of ``length`` K to an input ``x`` of length N.

    The raised-cosine filter is defined by

    .. math::
        h(t) =
        \begin{cases}
        \frac{\pi}{4T} \text{sinc}\left(\frac{1}{2\beta}\right), & \text { if }t = \pm \frac{T}{2\beta}\\
        \frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)\frac{\cos\left(\frac{\pi\beta t}{T}\right)}{1-\left(\frac{2\beta t}{T}\right)^2}, & \text{otherwise}
        \end{cases}

    where :math:`\beta` is the roll-off factor and :math:`T` the symbol duration.

    The filter length K is equal to the filter span in symbols
    (``span_in_symbols``) multiplied by the oversampling factor
    (``samples_per_symbol``). If this product is even, a value of one will
    be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The dtype of the output is `torch.float` if both ``x`` and the filter
    coefficients have dtype `torch.float`. Otherwise, the dtype of the output
    is `torch.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    :param span_in_symbols: Filter span as measured by the number of symbols
    :param samples_per_symbol: Number of samples per symbol, i.e., the
        oversampling factor
    :param beta: Roll-off factor.
        Must be in the range :math:`[0,1]`.
    :param window: Window that is applied to the filter coefficients.
        Can be `None`, a :class:`~sionna.phy.signal.Window` instance, or one of
        ``"hann"``, ``"hamming"``, ``"blackman"``.
    :param normalize: If `True`, the filter is normalized to have unit
        power. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        Input to which the filter is applied along the last dimension.

    :input padding: "full" (default) | "valid" | "same".
        Padding mode for convolving ``x`` and the filter.

    :input conjugate: `bool`, (default `False`).
        If `True`, the complex conjugate of the filter is applied.

    :output y: [..., M], `torch.complex` or `torch.float`.
        Filtered input. The length M depends on the ``padding``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import RaisedCosineFilter

        rc = RaisedCosineFilter(span_in_symbols=8, samples_per_symbol=4, beta=0.35)
        x = torch.randn(32, 100)
        y = rc(x, padding="same")
        print(y.shape)
        # torch.Size([32, 100])
    """

    def __init__(
        self,
        span_in_symbols: int,
        samples_per_symbol: int,
        beta: float,
        window: Optional[Union[Window, str]] = None,
        normalize: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            span_in_symbols,
            samples_per_symbol,
            window=window,
            normalize=normalize,
            precision=precision,
            device=device,
            **kwargs,
        )

        assert 0 <= beta <= 1, "beta must be from the interval [0, 1]"
        self._beta = beta
        self.coefficients = self._raised_cosine(self.sampling_times, 1.0, self.beta)

    @property
    def beta(self) -> float:
        """Roll-off factor"""
        return self._beta

    def _raised_cosine(
        self, t: np.ndarray, symbol_duration: float, beta: float
    ) -> np.ndarray:
        """Raised-cosine filter from Wikipedia
        https://en.wikipedia.org/wiki/Raised-cosine_filter"""
        h = np.zeros([len(t)], np.float32)
        for i, tt in enumerate(t):
            tt = np.abs(tt)
            if beta > 0 and (tt - np.abs(symbol_duration / 2 / beta) == 0):
                h[i] = np.pi / 4 / symbol_duration * np.sinc(1 / 2 / beta)
            else:
                h[i] = (
                    1.0
                    / symbol_duration
                    * np.sinc(tt / symbol_duration)
                    * np.cos(np.pi * beta * tt / symbol_duration)
                    / (1 - (2 * beta * tt / symbol_duration) ** 2)
                )
        return h


class RootRaisedCosineFilter(Filter):
    r"""Block for applying a root-raised-cosine filter of ``length`` K to an input ``x`` of length N.

    The root-raised-cosine filter is defined by

    .. math::
        h(t) =
        \begin{cases}
        \frac{1}{T} \left(1 + \beta\left(\frac{4}{\pi}-1\right) \right), & \text { if }t = 0\\
        \frac{\beta}{T\sqrt{2}} \left[ \left(1+\frac{2}{\pi}\right)\sin\left(\frac{\pi}{4\beta}\right) + \left(1-\frac{2}{\pi}\right)\cos\left(\frac{\pi}{4\beta}\right) \right], & \text { if }t = \pm\frac{T}{4\beta} \\
        \frac{1}{T} \frac{\sin\left(\pi\frac{t}{T}(1-\beta)\right) + 4\beta\frac{t}{T}\cos\left(\pi\frac{t}{T}(1+\beta)\right)}{\pi\frac{t}{T}\left(1-\left(4\beta\frac{t}{T}\right)^2\right)}, & \text { otherwise}
        \end{cases}

    where :math:`\beta` is the roll-off factor and :math:`T` the symbol duration.

    The filter length K is equal to the filter span in symbols
    (``span_in_symbols``) multiplied by the oversampling factor
    (``samples_per_symbol``). If this product is even, a value of one will
    be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The dtype of the output is `torch.float` if both ``x`` and the filter
    coefficients have dtype `torch.float`. Otherwise, the dtype of the output
    is `torch.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    :param span_in_symbols: Filter span as measured by the number of symbols
    :param samples_per_symbol: Number of samples per symbol, i.e., the
        oversampling factor
    :param beta: Roll-off factor.
        Must be in the range :math:`[0,1]`.
    :param window: Window that is applied to the filter coefficients.
        Can be `None`, a :class:`~sionna.phy.signal.Window` instance, or one of
        ``"hann"``, ``"hamming"``, ``"blackman"``.
    :param normalize: If `True`, the filter is normalized to have unit
        power. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        Input to which the filter is applied along the last dimension.

    :input padding: "full" (default) | "valid" | "same".
        Padding mode for convolving ``x`` and the filter.

    :input conjugate: `bool`, (default `False`).
        If `True`, the complex conjugate of the filter is applied.

    :output y: [..., M], `torch.complex` or `torch.float`.
        Filtered input. The length M depends on the ``padding``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import RootRaisedCosineFilter

        rrc = RootRaisedCosineFilter(span_in_symbols=8, samples_per_symbol=4, beta=0.35)
        x = torch.randn(32, 100)
        y = rrc(x, padding="same")
        print(y.shape)
        # torch.Size([32, 100])
    """

    def __init__(
        self,
        span_in_symbols: int,
        samples_per_symbol: int,
        beta: float,
        window: Optional[Union[Window, str]] = None,
        normalize: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            span_in_symbols,
            samples_per_symbol,
            window=window,
            normalize=normalize,
            precision=precision,
            device=device,
            **kwargs,
        )

        assert 0 <= beta <= 1, "beta must be from the interval [0, 1]"
        self._beta = beta
        self.coefficients = self._root_raised_cosine(
            self.sampling_times, 1.0, self.beta
        )

    @property
    def beta(self) -> float:
        """Roll-off factor"""
        return self._beta

    def _root_raised_cosine(
        self, t: np.ndarray, symbol_duration: float, beta: float
    ) -> np.ndarray:
        """Root-raised-cosine filter from Wikipedia
        https://en.wikipedia.org/wiki/Root-raised-cosine_filter"""
        h = np.zeros([len(t)], np.float32)
        for i, tt in enumerate(t):
            tt = np.abs(tt)
            if tt == 0:
                h[i] = 1 / symbol_duration * (1 + beta * (4 / np.pi - 1))
            elif beta > 0 and (tt - np.abs(symbol_duration / 4 / beta) == 0):
                h[i] = (
                    beta
                    / symbol_duration
                    / np.sqrt(2)
                    * (
                        (1 + 2 / np.pi) * np.sin(np.pi / 4 / beta)
                        + (1 - 2 / np.pi) * np.cos(np.pi / 4 / beta)
                    )
                )
            else:
                h[i] = (
                    1
                    / symbol_duration
                    / (
                        np.pi
                        * tt
                        / symbol_duration
                        * (1 - (4 * beta * tt / symbol_duration) ** 2)
                    )
                    * (
                        np.sin(np.pi * tt / symbol_duration * (1 - beta))
                        + 4
                        * beta
                        * tt
                        / symbol_duration
                        * np.cos(np.pi * tt / symbol_duration * (1 + beta))
                    )
                )
        return h


class SincFilter(Filter):
    r"""Block for applying a sinc filter of ``length`` K to an input ``x`` of length N.

    The sinc filter is defined by

    .. math::
        h(t) = \frac{1}{T}\text{sinc}\left(\frac{t}{T}\right)

    where :math:`T` is the symbol duration.

    The filter length K is equal to the filter span in symbols
    (``span_in_symbols``) multiplied by the oversampling factor
    (``samples_per_symbol``). If this product is even, a value of one will
    be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The dtype of the output is `torch.float` if both ``x`` and the filter
    coefficients have dtype `torch.float`. Otherwise, the dtype of the output
    is `torch.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    :param span_in_symbols: Filter span as measured by the number of symbols
    :param samples_per_symbol: Number of samples per symbol, i.e., the
        oversampling factor
    :param window: Window that is applied to the filter coefficients.
        Can be `None`, a :class:`~sionna.phy.signal.Window` instance, or one of
        ``"hann"``, ``"hamming"``, ``"blackman"``.
    :param normalize: If `True`, the filter is normalized to have unit
        power. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        Input to which the filter is applied along the last dimension.

    :input padding: "full" (default) | "valid" | "same".
        Padding mode for convolving ``x`` and the filter.

    :input conjugate: `bool`, (default `False`).
        If `True`, the complex conjugate of the filter is applied.

    :output y: [..., M], `torch.complex` or `torch.float`.
        Filtered input. The length M depends on the ``padding``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import SincFilter

        sinc = SincFilter(span_in_symbols=8, samples_per_symbol=4)
        x = torch.randn(32, 100)
        y = sinc(x, padding="same")
        print(y.shape)
        # torch.Size([32, 100])
    """

    def __init__(
        self,
        span_in_symbols: int,
        samples_per_symbol: int,
        window: Optional[Union[Window, str]] = None,
        normalize: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            span_in_symbols,
            samples_per_symbol,
            window=window,
            normalize=normalize,
            precision=precision,
            device=device,
            **kwargs,
        )

        self.coefficients = self._sinc(self.sampling_times, 1.0)

    def _sinc(self, t: np.ndarray, symbol_duration: float) -> np.ndarray:
        """Sinc filter"""
        return 1 / symbol_duration * np.sinc(t / symbol_duration)


class CustomFilter(Filter):
    r"""Block for applying a custom filter of ``length`` K to an input ``x`` of length N.

    The filter length K is equal to the filter span in symbols
    (``span_in_symbols``) multiplied by the oversampling factor
    (``samples_per_symbol``). If this product is even, a value of one will
    be added.

    The filter is applied through discrete convolution.

    An optional windowing function ``window`` can be applied to the filter.

    The dtype of the output is `torch.float` if both ``x`` and the filter
    coefficients have dtype `torch.float`. Otherwise, the dtype of the output
    is `torch.complex`.

    Three padding modes are available for applying the filter:

    *   "full" (default): Returns the convolution at each point of overlap between ``x`` and the filter.
        The length of the output is N + K - 1. Zero-padding of the input ``x`` is performed to
        compute the convolution at the borders.
    *   "same": Returns an output of the same length as the input ``x``. The convolution is computed such
        that the coefficients of the input ``x`` are centered on the coefficient of the filter with index
        (K-1)/2. Zero-padding of the input signal is performed to compute the convolution at the borders.
    *   "valid": Returns the convolution only at points where ``x`` and the filter completely overlap.
        The length of the output is N - K + 1.

    :param samples_per_symbol: Number of samples per symbol, i.e., the
        oversampling factor
    :param coefficients: [K], `torch.float` or `torch.complex` --
        Filter coefficients. The number of coefficients must be odd.
    :param window: Window that is applied to the filter coefficients.
        Can be `None`, a :class:`~sionna.phy.signal.Window` instance, or one of
        ``"hann"``, ``"hamming"``, ``"blackman"``.
    :param normalize: If `True`, the filter is normalized to have unit
        power. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [..., N], `torch.complex` or `torch.float`.
        Input to which the filter is applied along the last dimension.

    :input padding: "full" (default) | "valid" | "same".
        Padding mode for convolving ``x`` and the filter.

    :input conjugate: `bool`, (default `False`).
        If `True`, the complex conjugate of the filter is applied.

    :output y: [..., M], `torch.complex` or `torch.float`.
        Filtered input. The length M depends on the ``padding``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.signal import CustomFilter

        coefficients = torch.randn(33)
        filt = CustomFilter(samples_per_symbol=4, coefficients=coefficients)
        x = torch.randn(32, 100)
        y = filt(x, padding="same")
        print(y.shape)
        # torch.Size([32, 100])
    """

    def __init__(
        self,
        samples_per_symbol: int,
        coefficients: Union[torch.Tensor, np.ndarray],
        window: Optional[Union[Window, str]] = None,
        normalize: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        assert samples_per_symbol > 0, "samples_per_symbol must be positive"

        if isinstance(coefficients, np.ndarray):
            coeff_len = coefficients.shape[-1]
        else:
            coeff_len = coefficients.shape[-1]

        assert coeff_len % 2 == 1, "The number of coefficients must be odd"
        span_in_symbols = coeff_len // samples_per_symbol

        super().__init__(
            span_in_symbols,
            samples_per_symbol,
            window=window,
            normalize=normalize,
            precision=precision,
            device=device,
            **kwargs,
        )

        self.coefficients = coefficients
        assert (
            self.length == coeff_len
        ), f"`coefficients` must have length {self.length}"
