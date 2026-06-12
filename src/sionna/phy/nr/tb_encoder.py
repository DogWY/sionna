#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""5G NR transport block encoding for Sionna PHY."""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch

from sionna.phy import Block
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.scrambling import TB5GScrambler
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.nr.utils import calculate_tb_size


__all__ = ["TBEncoder"]


class TBEncoder(Block):
    # pylint: disable=line-too-long
    r"""5G NR transport block (TB) encoder as defined in TS 38.214
    :cite:p:`3GPPTS38214` and TS 38.211 :cite:p:`3GPPTS38211`

    The transport block (TB) encoder takes as input a `transport block` of
    information bits and generates a sequence of codewords for transmission.
    For this, the information bit sequence is segmented into multiple codewords,
    protected by additional CRC checks and FEC encoded. Further, interleaving
    and scrambling is applied before a codeword concatenation generates the
    final bit sequence. Fig. 1 provides an overview of the TB encoding
    procedure and we refer the interested reader to :cite:p:`3GPPTS38214` and
    :cite:p:`3GPPTS38211` for further details.

    ..  figure:: ../figures/tb_encoding.png

        Fig. 1: Overview TB encoding (CB CRC does not always apply).

    If ``n_rnti`` and ``n_id`` are given as list, the TBEncoder encodes
    `num_tx = len(` ``n_rnti`` `)` parallel input streams with different
    scrambling sequences per user.

    :param target_tb_size: Target transport block size, i.e., how many
        information bits are encoded into the TB. Note that the effective
        TB size can be slightly different due to quantization. If required,
        zero padding is internally applied.
    :param num_coded_bits: Number of coded bits after TB encoding.
    :param target_coderate: Target coderate.
    :param num_bits_per_symbol: Modulation order, i.e., number of bits per
        QAM symbol.
    :param num_layers: Number of transmission layers. Must be in [1, ..., 8].
        Defaults to 1.
    :param n_rnti: RNTI identifier provided by higher layer. Defaults to 1
        and must be in range `[0, 65535]`. Defines a part of the random seed
        of the scrambler. If provided as list, every list entry defines the
        RNTI of an independent input stream.
    :param n_id: Data scrambling ID :math:`n_\text{ID}` related to cell id
        and provided by higher layer. Defaults to 1 and must be in range
        `[0, 1023]`. If provided as list, every list entry defines the
        scrambling id of an independent input stream.
    :param channel_type: Can be either "PUSCH" or "PDSCH". Defaults to
        "PUSCH".
    :param codeword_index: Scrambler can be configured for two codeword
        transmission. ``codeword_index`` can be either 0 or 1. Must be 0 for
        ``channel_type`` = "PUSCH". Defaults to 0.
    :param use_scrambler: If `False`, no data scrambling is applied
        (non standard-compliant). Defaults to `True`.
    :param verbose: If `True`, additional parameters are printed during
        initialization. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation ('cpu' or 'cuda').

    :input inputs: [..., target_tb_size] or [..., num_tx, target_tb_size], `torch.float`.
        2+D tensor containing the information bits to be encoded. If
        ``n_rnti`` and ``n_id`` are a list of size `num_tx`, the input must
        be of shape ``[..., num_tx, target_tb_size]``.

    :output codeword: [..., num_coded_bits], `torch.float`.
        2+D tensor containing the sequence of the encoded codeword bits of
        the transport block.

    .. rubric:: Notes

    The parameters ``tb_size`` and ``num_coded_bits`` can be derived by the
    :meth:`~sionna.phy.nr.calculate_tb_size` function or
    by accessing the corresponding :class:`~sionna.phy.nr.PUSCHConfig`
    attributes.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.nr import TBEncoder

        encoder = TBEncoder(
            target_tb_size=1000,
            num_coded_bits=2000,
            target_coderate=0.5,
            num_bits_per_symbol=4,
            n_rnti=1,
            n_id=1
        )

        bits = torch.randint(0, 2, (10, 1000), dtype=torch.float32)
        coded_bits = encoder(bits)
        print(coded_bits.shape)
        # torch.Size([10, 2000])
    """

    def __init__(
        self,
        target_tb_size: int,
        num_coded_bits: int,
        target_coderate: float,
        num_bits_per_symbol: int,
        num_layers: int = 1,
        n_rnti: Union[int, List[int]] = 1,
        n_id: Union[int, List[int]] = 1,
        channel_type: str = "PUSCH",
        codeword_index: int = 0,
        use_scrambler: bool = True,
        verbose: bool = False,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        if not isinstance(use_scrambler, bool):
            raise TypeError("use_scrambler must be bool.")
        self._use_scrambler = use_scrambler

        if not isinstance(verbose, bool):
            raise TypeError("verbose must be bool.")
        self._verbose = verbose

        if channel_type not in ("PDSCH", "PUSCH"):
            raise ValueError("channel_type must be 'PDSCH' or 'PUSCH'.")
        self._channel_type = channel_type

        if target_tb_size % 1 != 0:
            raise ValueError("target_tb_size must be int.")
        self._target_tb_size = int(target_tb_size)

        if num_coded_bits % 1 != 0:
            raise ValueError("num_coded_bits must be int.")
        self._num_coded_bits = int(num_coded_bits)

        if not (0. < target_coderate <= 948 / 1024):
            raise ValueError("target_coderate must be in range (0, 0.925].")
        if isinstance(target_coderate, torch.Tensor):
            self.register_buffer(
                "_target_coderate",
                target_coderate.to(dtype=self.dtype, device=self.device),
                persistent=False,
            )
        else:
            self._target_coderate = target_coderate

        if num_bits_per_symbol % 1 != 0:
            raise ValueError("num_bits_per_symbol must be int.")
        self._num_bits_per_symbol = int(num_bits_per_symbol)

        if num_layers % 1 != 0:
            raise ValueError("num_layers must be int.")
        self._num_layers = int(num_layers)

        if channel_type == "PDSCH":
            if codeword_index not in (0, 1):
                raise ValueError("codeword_index must be 0 or 1.")
        else:
            if codeword_index != 0:
                raise ValueError('codeword_index must be 0 for "PUSCH".')
        self._codeword_index = int(codeword_index)

        # Handle n_rnti and n_id
        if isinstance(n_rnti, (list, tuple)):
            if not isinstance(n_id, (list, tuple)):
                raise ValueError("n_id must also be a list.")
            if len(n_rnti) != len(n_id):
                raise ValueError("n_id and n_rnti must be of same length.")
            self._n_rnti = list(n_rnti)
            self._n_id = list(n_id)
        else:
            self._n_rnti = [n_rnti]
            self._n_id = [n_id]

        for idx, n in enumerate(self._n_rnti):
            if n % 1 != 0:
                raise ValueError("n_rnti must be int.")
            self._n_rnti[idx] = int(n)
        for idx, n in enumerate(self._n_id):
            if n % 1 != 0:
                raise ValueError("n_id must be int.")
            self._n_id[idx] = int(n)

        self._num_tx = len(self._n_id)

        # Calculate TB parameters
        tbconfig = calculate_tb_size(
            target_tb_size=self._target_tb_size,
            num_coded_bits=self._num_coded_bits,
            target_coderate=self._target_coderate,
            modulation_order=self._num_bits_per_symbol,
            num_layers=self._num_layers,
            verbose=verbose,
        )
        self._tb_size = int(tbconfig[0])
        self._cb_size = int(tbconfig[1])
        self._num_cbs = int(tbconfig[2])
        self._tb_crc_length = int(tbconfig[3])
        self._cb_crc_length = int(tbconfig[4])
        # Convert cw_lengths to numpy if it's a tensor (calculate_tb_size returns tensors)
        cw_lengths = tbconfig[5]
        if isinstance(cw_lengths, torch.Tensor):
            self._cw_lengths = cw_lengths.cpu().numpy()
        else:
            self._cw_lengths = np.asarray(cw_lengths)
        # Flatten to 1-D array in case input had extra dimensions (e.g., [1, num_cb])
        self._cw_lengths = self._cw_lengths.flatten()
        # Cache as Python ints for torch.compile compatibility
        self._cw_lengths_sum = int(np.sum(self._cw_lengths))
        self._cw_lengths_max = int(np.max(self._cw_lengths))
        self._cw_lengths_min = int(np.min(self._cw_lengths))

        if self._tb_size > self._tb_crc_length + self._cw_lengths_sum:
            raise ValueError("Invalid TB parameters.")

        # Zero padding for quantization
        self._k_padding = self._tb_size - self._target_tb_size
        if self._tb_size != self._target_tb_size:
            print(f"Note: actual tb_size={self._tb_size} is slightly "
                  f"different than requested target_tb_size="
                  f"{self._target_tb_size} due to quantization. "
                  f"Internal zero padding will be applied.")

        # Effective coderate
        self._coderate = self._tb_size / self._num_coded_bits

        # Initialize CRC encoders with pre-built generator matrices
        # (k parameter enables torch.compile compatibility)
        if self._tb_crc_length == 16:
            self._tb_crc_encoder = CRCEncoder(
                "CRC16", k=self._tb_size, precision=precision, device=device)
        else:
            self._tb_crc_encoder = CRCEncoder(
                "CRC24A", k=self._tb_size, precision=precision, device=device)

        if self._cb_crc_length == 24:
            cb_info_size = self._cb_size - self._cb_crc_length
            self._cb_crc_encoder = CRCEncoder(
                "CRC24B", k=cb_info_size, precision=precision, device=device)
        else:
            self._cb_crc_encoder = None

        # Initialize scrambler
        if self._use_scrambler:
            self._scrambler = TB5GScrambler(
                n_rnti=self._n_rnti,
                n_id=self._n_id,
                binary=True,
                channel_type=channel_type,
                codeword_index=codeword_index,
                precision=precision,
                device=device,
            )
        else:
            self._scrambler = None

        # Initialize LDPC encoder
        self._encoder = LDPC5GEncoder(
            self._cb_size,
            self._cw_lengths_max,
            num_bits_per_symbol=1,  # Disable interleaver
            precision=precision,
            device=device,
        )

        # Initialize interleaver
        perm_seq_short, _ = self._encoder.generate_out_int(
            self._cw_lengths_min, num_bits_per_symbol)
        perm_seq_long, _ = self._encoder.generate_out_int(
            self._cw_lengths_max, num_bits_per_symbol)

        perm_seq = []
        perm_seq_punc = []

        payload_bit_pos = 0
        for length in self._cw_lengths:
            # Convert numpy scalar to Python int for reliable comparison
            length_val = int(length)
            # Skip zero-padded entries (calculate_tb_size pads with zeros)
            if length_val == 0:
                continue
            if self._cw_lengths_min == length_val:
                perm_seq = np.concatenate([perm_seq,
                                          perm_seq_short + payload_bit_pos])
                r = np.arange(payload_bit_pos + self._cw_lengths_min,
                              payload_bit_pos + self._cw_lengths_max)
                perm_seq_punc = np.concatenate([perm_seq_punc, r])
                payload_bit_pos += self._cw_lengths_max
            elif self._cw_lengths_max == length_val:
                perm_seq = np.concatenate([perm_seq,
                                          perm_seq_long + payload_bit_pos])
                payload_bit_pos += length_val
            else:
                raise ValueError("Invalid cw_lengths.")

        perm_seq = np.concatenate([perm_seq, perm_seq_punc])

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_output_perm", torch.tensor(
            perm_seq, dtype=torch.long, device=self.device))
        self.register_buffer("_output_perm_inv", torch.argsort(
            torch.tensor(perm_seq, dtype=torch.long, device=self.device)))

    #########################################
    # Public methods and properties
    #########################################

    @property
    def tb_size(self) -> int:
        r"""Effective number of information bits per TB.
        Note that (if required) internal zero padding can be
        applied to match the requested exact ``target_tb_size``."""
        return self._tb_size

    @property
    def k(self) -> int:
        r"""Number of input information bits.
        Equals ``tb_size`` except for zero padding of the last positions if the
        ``target_tb_size`` is quantized."""
        return self._target_tb_size

    @property
    def k_padding(self) -> int:
        """Number of zero padded bits at the end of the TB."""
        return self._k_padding

    @property
    def n(self) -> int:
        """Total number of output bits."""
        return self._num_coded_bits

    @property
    def num_cbs(self) -> int:
        """Number of code blocks."""
        return self._num_cbs

    @property
    def coderate(self) -> float:
        """Effective coderate of the TB after rate-matching including overhead for the CRC."""
        return self._coderate

    @property
    def ldpc_encoder(self) -> LDPC5GEncoder:
        """LDPC encoder used for TB encoding."""
        return self._encoder

    @property
    def scrambler(self) -> Optional[TB5GScrambler]:
        """Scrambler used for TB scrambling. `None` if no scrambler is used."""
        return self._scrambler

    @property
    def tb_crc_encoder(self) -> CRCEncoder:
        """TB CRC encoder."""
        return self._tb_crc_encoder

    @property
    def cb_crc_encoder(self) -> Optional[CRCEncoder]:
        """CB CRC encoder. `None` if no CB CRC is applied."""
        return self._cb_crc_encoder

    @property
    def num_tx(self) -> int:
        """Number of independent streams."""
        return self._num_tx

    @property
    def cw_lengths(self) -> np.ndarray:
        r"""Each list element defines the codeword length of each of the
        codewords after LDPC encoding and rate-matching. The total number of
        coded bits is :math:`\sum` ``cw_lengths``."""
        return self._cw_lengths

    @property
    def cw_lengths_sum(self) -> int:
        """Sum of codeword lengths (cached for torch.compile compatibility)."""
        return self._cw_lengths_sum

    @property
    def cw_lengths_max(self) -> int:
        """Maximum codeword length (cached for torch.compile compatibility)."""
        return self._cw_lengths_max

    @property
    def output_perm_inv(self) -> torch.Tensor:
        """Inverse interleaver pattern for output bit interleaver."""
        return self._output_perm_inv

    def build(self, input_shape: tuple) -> None:
        """Test input shapes for consistency."""
        if input_shape[-1] != self.k:
            raise ValueError(f"Invalid input shape. Expected TB length is {self.k}.")

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply transport block encoding procedure."""
        input_shape = list(inputs.shape)
        u = inputs.float()

        # Handle tb_size vs target_tb_size mismatch due to quantization
        if self._k_padding > 0:
            # tb_size > target_tb_size: pad with zeros
            padding_shape = list(u.shape)
            padding_shape[-1] = self._k_padding
            padding = torch.zeros(padding_shape, dtype=u.dtype, device=u.device)
            u = torch.cat([u, padding], dim=-1)
        elif self._k_padding < 0:
            # tb_size < target_tb_size: truncate to tb_size
            u = u[..., :self._tb_size]

        # Apply TB CRC
        u_crc = self._tb_crc_encoder(u)

        # CB segmentation
        u_cb = u_crc.reshape(-1, self._num_tx, self._num_cbs,
                             self._cb_size - self._cb_crc_length)

        # Apply CB CRC if relevant
        if self._cb_crc_length == 24:
            u_cb_crc = self._cb_crc_encoder(u_cb)
        else:
            u_cb_crc = u_cb

        # LDPC encode
        c_cb = self._encoder(u_cb_crc)

        # CB concatenation
        c = c_cb.reshape(-1, self._num_tx,
                         self._num_cbs * self._cw_lengths_max)

        # Apply interleaver
        c = torch.index_select(c, -1, self._output_perm)

        # Puncture last bits
        c = c[..., :self._cw_lengths_sum]

        # Apply scrambler
        if self._use_scrambler:
            c_scr = self._scrambler(c)
        else:
            c_scr = c

        # Cast to output dtype
        c_scr = c_scr.to(self.dtype)

        # Ensure output shapes
        output_shape = input_shape.copy()
        output_shape[-1] = self._cw_lengths_sum
        c_tb = c_scr.reshape(output_shape)

        return c_tb
