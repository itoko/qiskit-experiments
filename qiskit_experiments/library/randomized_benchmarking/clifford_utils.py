# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utilities for using the Clifford group in randomized benchmarking
"""
import itertools
import os
from functools import lru_cache
from numbers import Integral
from typing import Optional, Union, Tuple

import numpy as np
from numpy.random import Generator, default_rng

from qiskit.circuit import Gate, Instruction
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import SdgGate, HGate, SGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, random_clifford


@lru_cache(maxsize=None)
def _clifford_1q_int_to_instruction(num: Integral) -> Instruction:
    return CliffordUtils.clifford_1_qubit_circuit(num).to_instruction()


@lru_cache(maxsize=11520)
def _clifford_2q_int_to_instruction(num: Integral) -> Instruction:
    return CliffordUtils.clifford_2_qubit_circuit(num).to_instruction()


class VGate(Gate):
    """V Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new V Gate."""
        super().__init__("v", 1, [])

    def _define(self):
        """V Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(SdgGate(), [q[0]], []), (HGate(), [q[0]], [])]
        self.definition = qc


class WGate(Gate):
    """W Gate used in Clifford synthesis."""

    def __init__(self):
        """Create new W Gate."""
        super().__init__("w", 1, [])

    def _define(self):
        """W Gate definition."""
        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q)
        qc.data = [(HGate(), [q[0]], []), (SGate(), [q[0]], [])]
        self.definition = qc


class CliffordUtils:
    """Utilities for generating 1 and 2 qubit clifford circuits and elements"""

    NUM_CLIFFORD_1_QUBIT = 24
    NUM_CLIFFORD_2_QUBIT = 11520
    CLIFFORD_1_QUBIT_SIG = (2, 3, 4)
    CLIFFORD_2_QUBIT_SIGS = [
        (2, 2, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 3, 3, 4, 4),
        (2, 2, 3, 3, 4, 4),
    ]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit(cls, num):
        """Return the 1-qubit clifford element corresponding to `num`
        where `num` is between 0 and 23.
        """
        return Clifford(cls.clifford_1_qubit_circuit(num), validate=False)

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit(cls, num):
        """Return the 2-qubit clifford element corresponding to `num`
        where `num` is between 0 and 11519.
        """
        return Clifford(cls.clifford_2_qubit_circuit(num), validate=False)

    def random_cliffords(
        self, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford elements"""
        if num_qubits > 2:
            return random_clifford(num_qubits, seed=rng)

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [Clifford(self.clifford_1_qubit_circuit(i), validate=False) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [Clifford(self.clifford_2_qubit_circuit(i), validate=False) for i in samples]

    def random_clifford_circuits(
        self, num_qubits: int, size: int = 1, rng: Optional[Union[int, Generator]] = None
    ):
        """Generate a list of random clifford circuits"""
        if num_qubits > 2:
            return [random_clifford(num_qubits, seed=rng).to_circuit() for _ in range(size)]

        if rng is None:
            rng = default_rng()

        if isinstance(rng, int):
            rng = default_rng(rng)

        if num_qubits == 1:
            samples = rng.integers(24, size=size)
            return [self.clifford_1_qubit_circuit(i) for i in samples]
        else:
            samples = rng.integers(11520, size=size)
            return [self.clifford_2_qubit_circuit(i) for i in samples]

    @classmethod
    @lru_cache(maxsize=24)
    def clifford_1_qubit_circuit(cls, num):
        """Return the 1-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 23.
        """
        unpacked = cls._unpack_num(num, (2, 3, 4))
        i, j, p = unpacked[0], unpacked[1], unpacked[2]
        qc = QuantumCircuit(1, name=f"Clifford-1Q({num})")
        if i == 1:
            qc.h(0)
        if j == 1:
            qc.sxdg(0)
        if j == 2:
            qc.s(0)
        if p == 1:
            qc.x(0)
        if p == 2:
            qc.y(0)
        if p == 3:
            qc.z(0)

        return qc

    @classmethod
    @lru_cache(maxsize=11520)
    def clifford_2_qubit_circuit(cls, num):
        """Return the 2-qubit clifford circuit corresponding to `num`
        where `num` is between 0 and 11519.
        """
        vals = cls._unpack_num_multi_sigs(num, cls.CLIFFORD_2_QUBIT_SIGS)
        qc = QuantumCircuit(2, name=f"Clifford-2Q({num})")
        if vals[0] == 0 or vals[0] == 3:
            (form, i0, i1, j0, j1, p0, p1) = vals
        else:
            (form, i0, i1, j0, j1, k0, k1, p0, p1) = vals
        if i0 == 1:
            qc.h(0)
        if i1 == 1:
            qc.h(1)
        if j0 == 1:
            qc.sxdg(0)
        if j0 == 2:
            qc.s(0)
        if j1 == 1:
            qc.sxdg(1)
        if j1 == 2:
            qc.s(1)
        if form in (1, 2, 3):
            qc.cx(0, 1)
        if form in (2, 3):
            qc.cx(1, 0)
        if form == 3:
            qc.cx(0, 1)
        if form in (1, 2):
            if k0 == 1:  # V gate
                qc.sdg(0)
                qc.h(0)
            if k0 == 2:  # W gate
                qc.h(0)
                qc.s(0)
            if k1 == 1:  # V gate
                qc.sdg(1)
                qc.h(1)
            if k1 == 2:  # W gate
                qc.h(1)
                qc.s(1)
        if p0 == 1:
            qc.x(0)
        if p0 == 2:
            qc.y(0)
        if p0 == 3:
            qc.z(0)
        if p1 == 1:
            qc.x(1)
        if p1 == 2:
            qc.y(1)
        if p1 == 3:
            qc.z(1)

        return qc

    @staticmethod
    def _unpack_num(num, sig):
        r"""Returns a tuple :math:`(a_1, \ldots, a_n)` where
        :math:`0 \le a_i \le \sigma_i` where
        sig=:math:`(\sigma_1, \ldots, \sigma_n)` and num is the sequential
        number of the tuple
        """
        res = []
        for k in sig:
            res.append(num % k)
            num //= k
        return res

    @staticmethod
    def _unpack_num_multi_sigs(num, sigs):
        """Returns the result of `_unpack_num` on one of the
        signatures in `sigs`
        """
        for i, sig in enumerate(sigs):
            sig_size = 1
            for k in sig:
                sig_size *= k
            if num < sig_size:
                return [i] + CliffordUtils._unpack_num(num, sig)
            num -= sig_size
        return None


NUM_CLIFFORD_1Q = 24
NUM_CLIFFORD_2Q = 11520
CLIFF_SINGLE_GATE_MAP_1Q = {
    ("id", (0,)): 0,
    ("h", (0,)): 1,
    ("sxdg", (0,)): 2,
    ("s", (0,)): 4,
    ("x", (0,)): 6,
    ("sx", (0,)): 8,
    ("y", (0,)): 12,
    ("z", (0,)): 18,
    ("sdg", (0,)): 22,
}
CLIFF_SINGLE_GATE_MAP_2Q = {
    ("id", (0,)): 0,
    ("id", (1,)): 0,
    ("h", (0,)): 1,
    ("h", (1,)): 2,
    ("sxdg", (0,)): 4,
    ("sxdg", (1,)): 12,
    ("s", (0,)): 8,
    ("s", (1,)): 24,
    ("x", (0,)): 36,
    ("x", (1,)): 144,
    ("sx", (0,)): 40,
    ("sx", (1,)): 156,
    ("y", (0,)): 72,
    ("y", (1,)): 288,
    ("z", (0,)): 108,
    ("z", (1,)): 432,
    ("sdg", (0,)): 116,
    ("sdg", (1,)): 456,
    ("cx", (0, 1)): 576,
    ("cx", (1, 0)): 851,
    ("cz", (0, 1)): 806,
    ("cz", (1, 0)): 806,
}


########
# Functions for 1-qubit integer Clifford operations
def int_clifford_1q_gate(op: Instruction) -> int:
    """
    Convert a given 1-qubit clifford operation to the corresponding integer.
    Note that supported operations are limited to ones in `CLIFF_SINGLE_GATE_MAP_1Q` or Rz gate.

    Args:
        op: operation to be converted.

    Returns:
        An integer representing a Clifford consisting of a single operation.

    Raises:
        QiskitError: if the input instruction is not a Clifford instruction.
        QiskitError: if rz is given with a angle that is not Clifford.
    """
    if op.name in {"delay", "barrier"}:
        return 0
    try:
        name = _deparameterized_name(op)
        return CLIFF_SINGLE_GATE_MAP_1Q[(name, (0,))]
    except QiskitError as err:
        raise QiskitError(
            f"Parameterized instruction {op.name} could not be converted to integer Clifford"
        ) from err
    except KeyError as err:
        raise QiskitError(
            f"Instruction {op.name} could not be converted to integer Clifford"
        ) from err


def _hash_cliff(cliff):
    """Produce a hashable value that is unique for each different Clifford.  This should only be
    used internally when the classes being hashed are under our control, because classes of this
    type are mutable."""
    table = cliff.table
    abits = np.packbits(table.array)
    pbits = np.packbits(table.phase)
    return abits.tobytes(), pbits.tobytes()


_TO_CLIFF = {i: CliffordUtils.clifford_1_qubit(i) for i in range(NUM_CLIFFORD_1Q)}
_TO_INT = {_hash_cliff(cliff): i for i, cliff in _TO_CLIFF.items()}


def _create_compose_map_1q():
    products = np.zeros((NUM_CLIFFORD_1Q, NUM_CLIFFORD_1Q), dtype=int)
    for i in range(NUM_CLIFFORD_1Q):
        for j in range(NUM_CLIFFORD_1Q):
            cliff = _TO_CLIFF[i].compose(_TO_CLIFF[j])
            products[i][j] = _TO_INT[_hash_cliff(cliff)]

    return products


def _create_inverse_map_1q():
    invs = np.zeros(NUM_CLIFFORD_1Q, dtype=int)
    for i in range(NUM_CLIFFORD_1Q):
        invs[i] = _TO_INT[_hash_cliff(_TO_CLIFF[i].adjoint())]

    return invs


CLIFFORD_COMPOSE_1Q = _create_compose_map_1q()
CLIFFORD_INVERSE_1Q = _create_inverse_map_1q()


def compose_1q(lhs: Integral, rhs: Integral) -> Integral:
    """Return the composition of 1-qubit clifford integers."""
    return CLIFFORD_COMPOSE_1Q[lhs][rhs]


def inverse_1q(num: Integral) -> Integral:
    """Return the inverse of 1-qubit clifford integers."""
    return CLIFFORD_INVERSE_1Q[num]


def _deparameterized_name(inst: Instruction) -> str:
    if inst.name == "rz":
        if np.isclose(inst.params[0], np.pi) or np.isclose(inst.params[0], -np.pi):
            return "z"
        elif np.isclose(inst.params[0], np.pi / 2):
            return "s"
        elif np.isclose(inst.params[0], -np.pi / 2):
            return "sdg"
        else:
            raise QiskitError("Wrong param {} for rz in clifford".format(inst.params[0]))

    return inst.name


########
# Functions for 2-qubit integer Clifford operations
def int_clifford_2q_gate(op: Instruction, qubits: Tuple[int, ...]) -> int:
    """
    Convert a given 1-qubit clifford operation to the corresponding integer.
    Note that supported operations are limited to ones in `CLIFF_SINGLE_GATE_MAP_2Q` or Rz gate.

    Args:
        op: operation of instruction to be converted.
        qubits: qubits to which the operation applies

    Returns:
        An integer representing a Clifford consisting of a single operation.

    Raises:
        QiskitError: if the input instruction is not a Clifford instruction.
        QiskitError: if rz is given with a angle that is not Clifford.
    """
    if op.name in {"delay", "barrier"}:
        return 0
    try:
        name = _deparameterized_name(op)
        return CLIFF_SINGLE_GATE_MAP_2Q[(name, qubits)]
    except QiskitError as err:
        raise QiskitError(
            f"Parameterized instruction {op.name} could not be converted to integer Clifford"
        ) from err
    except KeyError as err:
        raise QiskitError(
            f"Instruction {op.name} on {qubits} could not be converted to integer Clifford"
        ) from err


def _append_v_w(qc, vw0, vw1):
    if vw0 == "v":
        qc.sdg(0)
        qc.h(0)
    elif vw0 == "w":
        qc.h(0)
        qc.s(0)
    if vw1 == "v":
        qc.sdg(1)
        qc.h(1)
    elif vw1 == "w":
        qc.h(1)
        qc.s(1)


def _create_cliff_2q_layer_0():
    """Layer 0 consists of 0 or 1 H gates on each qubit, followed by 0/1/2 V gates on each qubit.
    Number of Cliffords == 36."""
    circuits = []
    num_h = [0, 1]
    v_w_gates = ["i", "v", "w"]
    for h0, h1, v0, v1 in itertools.product(num_h, num_h, v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        for _ in range(h0):
            qc.h(0)
        for _ in range(h1):
            qc.h(1)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)
    return circuits


def _create_cliff_2q_layer_1():
    """Layer 1 consists of one of the following:
    - nothing
    - cx(0,1) followed by 0/1/2 V gates on each qubit
    - cx(0,1), cx(1,0) followed by 0/1/2 V gates on each qubit
    - cx(0,1), cx(1,0), cx(0,1)
    Number of Cliffords == 20."""
    circuits = [QuantumCircuit(2)]  # identity at the beginning

    v_w_gates = ["i", "v", "w"]
    for v0, v1 in itertools.product(v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)

    for v0, v1 in itertools.product(v_w_gates, v_w_gates):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        _append_v_w(qc, v0, v1)
        circuits.append(qc)

    qc = QuantumCircuit(2)  # swap at the end
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    circuits.append(qc)
    return circuits


def _create_cliff_2q_layer_2():
    """Layer 2 consists of a Pauli gate on each qubit {Id, X, Y, Z}.
    Number of Cliffords == 16."""
    circuits = []
    pauli = ("i", XGate(), YGate(), ZGate())
    for p0, p1 in itertools.product(pauli, pauli):
        qc = QuantumCircuit(2)
        if p0 != "i":
            qc.append(p0, [0])
        if p1 != "i":
            qc.append(p1, [1])
        circuits.append(qc)
    return circuits


def _load_clifford_compose_2q():
    dirname = os.path.dirname(__file__)
    data = np.load(f"{dirname}/data/clifford_compose_2q_gate.npz")
    table = []
    for row in data["table"]:
        dic = {rhs: result for result, rhs in zip(row, CLIFF_SINGLE_GATE_MAP_2Q.values())}
        table.append(dic)
    return table


def _load_clifford_inverse_2q():
    dirname = os.path.dirname(__file__)
    data = np.load(f"{dirname}/data/clifford_inverse_2q.npz")
    return data["table"]


_NUM_LAYER_0 = 36
_NUM_LAYER_1 = 20
_NUM_LAYER_2 = 16
_CLIFFORD_LAYER = (
    _create_cliff_2q_layer_0(),
    _create_cliff_2q_layer_1(),
    _create_cliff_2q_layer_2(),
)

_CLIFFORD_COMPOSE_2Q_GATE = _load_clifford_compose_2q()
CLIFFORD_INVERSE_2Q = _load_clifford_inverse_2q()


def compose_2q(lhs: Integral, rhs: Integral) -> Integral:
    """Return the composition of 2-qubit clifford integers."""
    num = lhs
    for layour, idx in enumerate(_layer_indices_from_num(rhs)):
        circ = _CLIFFORD_LAYER[layour][idx]
        num = _compose_num_with_circuit_2q(num, circ)
    return num


def inverse_2q(num: Integral) -> Integral:
    """Return the inverse of 2-qubit clifford integers."""
    return CLIFFORD_INVERSE_2Q[num]


def _compose_num_with_circuit_2q(num: Integral, qc: QuantumCircuit) -> Integral:
    """Compose a number that represents a Clifford, with a Clifford circuit, and return the
    number that represents the resulting Clifford."""
    lhs = num
    for inst in qc:
        qubits = tuple(qc.find_bit(q).index for q in inst.qubits)
        rhs = int_clifford_2q_gate(op=inst.operation, qubits=qubits)
        try:
            lhs = _CLIFFORD_COMPOSE_2Q_GATE[lhs][rhs]
        except KeyError as err:
            raise Exception(f"_CLIFFORD_COMPOSE_2Q_GATE[{lhs}][{rhs}]") from err
    return lhs


def _num_from_layer_indices(triplet: Tuple[Integral, Integral, Integral]) -> Integral:
    """Return the clifford number corresponding to the input triplet."""
    num = triplet[0] * _NUM_LAYER_1 * _NUM_LAYER_2 + triplet[1] * _NUM_LAYER_2 + triplet[2]
    return num


def _layer_indices_from_num(num: Integral) -> Tuple[Integral, Integral, Integral]:
    """Return the triplet of layer indices corresponding to the input number."""
    idx2 = num % _NUM_LAYER_2
    num = num // _NUM_LAYER_2
    idx1 = num % _NUM_LAYER_1
    idx0 = num // _NUM_LAYER_1
    return idx0, idx1, idx2
