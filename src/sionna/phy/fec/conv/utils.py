#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for convolutional codes."""

from typing import Optional, Tuple

import numpy as np
import torch

from sionna.phy.fec.utils import int2bin, bin2int


__all__ = ["polynomial_selector", "resolve_gen_poly", "Trellis"]


def polynomial_selector(
    rate: float,
    constraint_length: int
) -> Tuple[str, ...]:
    """Returns generator polynomials for given code parameters.

    The polynomials are chosen from :cite:p:`Moon` which are tabulated by searching
    for polynomials with best free distances for a given rate and
    constraint length.

    :param rate: Desired rate of the code.
        Currently, only r=1/3 and r=1/2 are supported.
    :param constraint_length: Desired constraint length of the encoder.
        Must be between 3 and 8 inclusive.

    :output gen_poly: Tuple of strings with each string being a 0,1
        sequence where each polynomial is represented in binary form.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.conv import polynomial_selector

        gen_poly = polynomial_selector(rate=0.5, constraint_length=5)
        print(gen_poly)
        # ('10011', '11011')
    """
    if not isinstance(constraint_length, int):
        raise TypeError("constraint_length must be int.")
    if not 2 < constraint_length < 9:
        raise ValueError("Unsupported constraint_length.")
    if rate not in (1/2, 1/3):
        raise ValueError("Unsupported rate.")

    rate_half_dict = {
        3: ('101', '111'),  # (5,7)
        4: ('1101', '1011'),  # (15, 13)
        5: ('10011', '11011'),  # (23, 33) # taken from GSM05.03, 4.1.3
        6: ('110101', '101111'),  # (65, 57)
        7: ('1011011', '1111001'),  # (133, 171)
        8: ('11100101', '10011111'),  # (345, 237)
    }

    rate_third_dict = {
        3: ('101', '111', '111'),  # (5,7,7)
        4: ('1011', '1101', '1111'),  # (54, 64, 74)
        5: ('10101', '11011', '11111'),  # (52, 66, 76)
        6: ('100111', '101011', '111101'),  # (47,53,75)
        7: ('1111001', '1100101', '1011011'),  # (554, 744)
        8: ('10010101', '11011001', '11110111'),  # (452, 662, 756)
    }

    gen_poly_dict = {
        1/2: rate_half_dict,
        1/3: rate_third_dict,
    }

    gen_poly = gen_poly_dict[rate][constraint_length]
    return gen_poly


def resolve_gen_poly(
    gen_poly: Optional[Tuple[str, ...]] = None,
    rate: Optional[float] = None,
    constraint_length: Optional[int] = None,
) -> Tuple[str, ...]:
    """Validates explicit generator polynomials or selects them from tables.

    If ``gen_poly`` is provided, validates that it consists of equal-length
    binary strings. Otherwise, selects polynomials via
    :func:`polynomial_selector` using ``rate`` and ``constraint_length``.

    :param gen_poly: Explicit generator polynomials as binary strings.
    :param rate: Code rate (1/2 or 1/3). Required when ``gen_poly`` is `None`.
    :param constraint_length: Constraint length (3--8). Required when
        ``gen_poly`` is `None`.

    :output gen_poly: Validated or selected generator polynomials.
    """
    if gen_poly is not None:
        if not all(isinstance(poly, str) for poly in gen_poly):
            raise TypeError("Each element of gen_poly must be a string.")
        if not all(len(poly) == len(gen_poly[0]) for poly in gen_poly):
            raise ValueError("Each polynomial must be of same length.")
        if not all(
            all(char in ['0', '1'] for char in poly) for poly in gen_poly
        ):
            raise ValueError("Each polynomial must be a binary string of "
                             "0/1 characters.")
        return gen_poly

    valid_rates = (1/2, 1/3)
    valid_constraint_length = (3, 4, 5, 6, 7, 8)

    if constraint_length not in valid_constraint_length:
        raise ValueError("Constraint length must be between 3 and 8.")
    if rate not in valid_rates:
        raise ValueError("Rate must be 1/3 or 1/2.")

    return polynomial_selector(rate, constraint_length)


class Trellis(torch.nn.Module):
    r"""Trellis structure for a given generator polynomial.

    Defines state transitions and output symbols (and bits) for each current
    state and input.

    :param gen_poly: Sequence of strings with each string being a 0,1 sequence.
        If ``rsc`` is `True`, the first polynomial will act as denominator for
        the remaining generator polynomials. For example, ``rsc`` = `True` and
        ``gen_poly`` = (``'111'``, ``'101'``, ``'011'``) implies generator
        matrix equals
        :math:`G(D)=[\frac{1+D^2}{1+D+D^2}, \frac{D+D^2}{1+D+D^2}]`.
        Currently Trellis is only implemented for generator matrices of
        size :math:`\frac{1}{n}`.
    :param rsc: Boolean flag indicating whether the Trellis is recursive
        systematic or not. If `True`, the encoder is recursive systematic in
        which case the first polynomial in ``gen_poly`` is used as the feedback
        polynomial. Defaults to `False`.
    :param device: Device for computation (e.g., 'cpu', 'cuda:0').
        If `None`, uses CPU.

    .. rubric:: Examples


    .. code-block:: python

        from sionna.phy.fec.conv import Trellis

        trellis = Trellis(gen_poly=('101', '111'))
        print(f"Number of states: {trellis.ns}")
        # Number of states: 4
    """

    def __init__(
        self,
        gen_poly: Tuple[str, ...],
        rsc: bool = False,
        device: str = None,
    ):
        super().__init__()
        self.rsc = rsc
        self.gen_poly = gen_poly
        self.constraint_length = len(self.gen_poly[0])

        self.conv_k = 1
        self.conv_n = len(self.gen_poly)
        self.ni = 2**self.conv_k
        self.ns = 2**(self.constraint_length - 1)
        self._mu = len(gen_poly[0]) - 1

        if self.rsc:
            self.fb_poly = [int(x) for x in self.gen_poly[0]]
            if self.fb_poly[0] != 1:
                raise ValueError(
                    "RSC feedback polynomial must start with 1."
                )
            if self.conv_k != 1:
                raise ValueError("RSC only supports conv_k=1.")

        resolved_device = device if device is not None else "cpu"
        self.register_buffer(
            "_device_ref",
            torch.empty(0, device=resolved_device),
            persistent=False,
        )

        # For current state i and input j, state transitions i->to_nodes[i][j]
        self.register_buffer("to_nodes", None, persistent=False)

        # For current state i, valid state transitions are from_nodes[i][:]-> i
        self.register_buffer("from_nodes", None, persistent=False)

        # Given states i and j, Trellis emits op_mat[i][j] symbol if neq -1
        self.register_buffer("op_mat", None, persistent=False)

        # Given next state as i, trellis emits op_by_tonode[i][:] symbols
        self.register_buffer("op_by_tonode", None, persistent=False)

        # Given ip_by_tonode[i][:] bits as input, trellis transitions to State i
        self.register_buffer("ip_by_tonode", None, persistent=False)

        # Given from state i and input j, trellis emits op_by_fromnode[i][j]
        self.register_buffer("op_by_fromnode", None, persistent=False)

        self._generate_transitions()

    @property
    def device(self) -> str:
        """Device on which trellis tensors reside."""
        return str(self._device_ref.device)

    @property
    def mu(self) -> int:
        """Memory (constraint length - 1) of the convolutional code."""
        return self._mu

    def _binary_matmul(self, st: str) -> np.ndarray:
        """For a given state st, multiplies each generator polynomial with st
        and returns the sum modulo 2 bit as output.
        """
        op = np.zeros(self.conv_n, int)
        if len(st) != len(self.gen_poly[0]):
            raise ValueError(
                "State length must match generator polynomial length."
            )
        for i, poly in enumerate(self.gen_poly):
            op_int = sum(
                int(char) * int(poly[idx]) for idx, char in enumerate(st)
            )
            op[i] = int2bin(op_int % 2, 1)[0]
        return op

    def _binary_vecmul(self, v1: list, v2: str) -> int:
        """For given vectors v1, v2, multiplies the two binary vectors
        with each other and returns binary output i.e. sum modulo 2.
        """
        if len(v1) != len(v2):
            raise ValueError("v1 and v2 must have the same length.")
        op_int = sum(x * int(v2[idx]) for idx, x in enumerate(v1))
        op = int2bin(op_int, 1)[0]
        return op

    def _generate_transitions(self):
        """Generates state transitions for different input symbols.

        This depends only on constraint_length and is independent
        of the generator polynomials.
        """
        to_nodes = np.full((self.ns, self.ni), -1, int)
        from_nodes = np.full((self.ns, self.ni), -1, int)
        op_mat = np.full((self.ns, self.ns), -1, int)
        ip_by_tonode = np.full((self.ns, self.ni), -1, int)
        op_by_tonode = np.full((self.ns, self.ni), -1, int)
        op_by_fromnode = np.full((self.ns, self.ni), -1, int)

        from_nodes_ctr = np.zeros(self.ns, int)
        for i in range(self.ni):
            ip_bit = int2bin(i, self.conv_k)[0]
            for j in range(self.ns):
                curr_st_bits = int2bin(j, self.constraint_length - 1)
                if self.rsc:
                    fb_bit = self._binary_vecmul(
                        curr_st_bits, self.gen_poly[0][1:]
                    )
                    new_bit = int2bin(ip_bit + fb_bit, 1)[0]
                else:
                    new_bit = ip_bit
                state_bits = [new_bit] + curr_st_bits
                j_to = bin2int(state_bits[:-1])

                to_nodes[j][i] = j_to
                from_nodes[j_to][from_nodes_ctr[j_to]] = j

                # Convert state_bits list to string for _binary_matmul
                state_bits_str = "".join(str(b) for b in state_bits)
                op_bits = self._binary_matmul(state_bits_str)
                op_sym = bin2int(list(op_bits))
                op_mat[j, j_to] = op_sym
                op_by_tonode[j_to, from_nodes_ctr[j_to]] = op_sym
                ip_by_tonode[j_to, from_nodes_ctr[j_to]] = i
                op_by_fromnode[j][i] = op_sym
                from_nodes_ctr[j_to] += 1

        self.register_buffer(
            "to_nodes",
            torch.tensor(to_nodes, dtype=torch.int32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "from_nodes",
            torch.tensor(from_nodes, dtype=torch.int32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "op_mat",
            torch.tensor(op_mat, dtype=torch.int32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "ip_by_tonode",
            torch.tensor(ip_by_tonode, dtype=torch.int32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "op_by_tonode",
            torch.tensor(op_by_tonode, dtype=torch.int32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "op_by_fromnode",
            torch.tensor(op_by_fromnode, dtype=torch.int32, device=self.device),
            persistent=False,
        )

    def to(self, device: str) -> "Trellis":
        """Moves all tensors to the specified device.

        :param device: Target device (e.g., 'cpu', 'cuda:0').

        :output trellis: Self reference for chaining.
        """
        return super().to(device)


