# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Double Interleaved RB Experiment class.
"""
import copy
import itertools
import warnings
from typing import Union, Iterable, Optional, List, Sequence, Tuple

from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit, Instruction, Gate, Delay
from qiskit.compiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.quantum_info import Clifford
from qiskit.transpiler.exceptions import TranspilerError
from qiskit_experiments.framework.backend_timing import BackendTiming
from .clifford_utils import _truncate_inactive_qubits
from .clifford_utils import num_from_1q_circuit, num_from_2q_circuit
from .double_interleaved_rb_analysis import DoubleInterleavedRBAnalysis
from .standard_rb import StandardRB, SequenceElementType


class DoubleInterleavedRB(StandardRB):
    """Interleaved randomized benchmarking experiment with two interleaved gates.

    # section: overview
        Interleaved Randomized Benchmarking (RB) is a method
        to estimate the average error-rate of a certain quantum gate.

        An interleaved RB experiment generates a standard RB sequences of random Cliffords
        and another sequence with the interleaved given gate.
        After running the two sequences on a backend, it calculates the probabilities to get back to
        the ground state, fits the two exponentially decaying curves, and estimates
        the interleaved gate error. See Ref. [1] for details.

    # section: analysis_ref
        :py:class:`InterleavedRBAnalysis`

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    """

    def __init__(
        self,
        first_interleaved_op: Union[QuantumCircuit, Gate, Delay, Clifford],
        second_interleaved_op: Union[QuantumCircuit, Gate, Delay, Clifford],
        qubits: Sequence[int],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: bool = False,
    ):
        """Initialize an interleaved randomized benchmarking experiment.

        Args:
            interleaved_element: The element to interleave,
                    given either as a Clifford element, gate, delay or circuit.
                    If the element contains any non-basis gates,
                    it will be transpiled with ``transpiled_options`` of this experiment.
                    If it is/contains a delay, its duration and unit must comply with
                    the timing constraints of the ``backend``
                    (:class:`~qiskit_experiments.framework.backend_timing.BackendTiming`
                    is useful to obtain valid delays).
                    Parameterized circuits/instructions are not allowed.
            qubits: list of physical qubits for the experiment.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for
                           all lengths. If False for sample of lengths longer
                           sequences are constructed by appending additional
                           Clifford samples to shorter sequences.

        Raises:
            QiskitError: If the ``interleaved_element`` is invalid because:
                * it has different number of qubits from the qubits argument
                * it is not convertible to Clifford object
                * it has an invalid delay (e.g. violating the timing constraints of the backend)
        """
        # Validations of interleaved_element
        # - validate number of qubits of interleaved_element
        if len(qubits) != first_interleaved_op.num_qubits:
            raise QiskitError(
                f"Mismatch in number of qubits between qubits ({len(qubits)})"
                f" and interleaved element ({first_interleaved_op.num_qubits})."
            )
        # - validate if interleaved_element is Clifford
        try:
            interleaved_clifford_1 = Clifford(first_interleaved_op)
            interleaved_clifford_2 = Clifford(second_interleaved_op)
        except QiskitError as err:
            raise QiskitError(
                f"Interleaved element {first_interleaved_op.name} could not be converted to Clifford."
            ) from err
        # - validate delays in interleaved_element
        delay_ops = []
        if isinstance(first_interleaved_op, Delay):
            delay_ops = [first_interleaved_op]
        elif isinstance(first_interleaved_op, QuantumCircuit):
            delay_ops = [
                delay.operation for delay in first_interleaved_op.get_instructions("delay")
            ]
        if delay_ops:
            timing = BackendTiming(backend)
        for delay_op in delay_ops:
            if delay_op.unit != timing.delay_unit:
                raise QiskitError(
                    f"Interleaved delay for backend {backend} must have time unit {timing.delay_unit}."
                    " Use BackendTiming to set valid duration and unit for delays."
                )
            if timing.delay_unit == "dt":
                valid_duration = timing.round_delay(samples=delay_op.duration)
                if delay_op.duration != valid_duration:
                    raise QiskitError(
                        f"Interleaved delay duration {delay_op.duration}[dt] violates the timing"
                        f" constraints of the backend {backend}. It could be {valid_duration}[dt]."
                        " Use BackendTiming to set valid duration for delays."
                    )
        # Warnings
        if isinstance(first_interleaved_op, QuantumCircuit) and first_interleaved_op.calibrations:
            warnings.warn("Calibrations in interleaved circuit are ignored", UserWarning)

        super().__init__(
            qubits,
            lengths,
            backend=backend,
            num_samples=num_samples,
            seed=seed,
            full_sampling=full_sampling,
        )
        # Convert interleaved element to integer for speed in 1Q or 2Q case
        if self.num_qubits == 1:
            self._interleaved_cliff_1 = num_from_1q_circuit(interleaved_clifford_1.to_circuit())
            self._interleaved_cliff_2 = num_from_1q_circuit(interleaved_clifford_2.to_circuit())
        elif self.num_qubits == 2:
            self._interleaved_cliff_1 = num_from_2q_circuit(interleaved_clifford_1.to_circuit())
            self._interleaved_cliff_2 = num_from_2q_circuit(interleaved_clifford_2.to_circuit())
        # Convert interleaved element to circuit for speed in 3Q or more case
        else:
            self._interleaved_cliff_1 = interleaved_clifford_1.to_circuit()
            self._interleaved_cliff_2 = interleaved_clifford_2.to_circuit()
        self._interleaved_element_1 = first_interleaved_op  # Original interleaved element
        self._interleaved_element_2 = second_interleaved_op  # Original interleaved element
        self._interleaved_op_1 = None  # Transpiled interleaved element for speed
        self._interleaved_op_2 = None  # Transpiled interleaved element for speed
        self.analysis = DoubleInterleavedRBAnalysis()
        self.analysis.set_options(outcome="0" * self.num_qubits)

    def circuits(self) -> List[QuantumCircuit]:
        """Return a list of RB circuits.

        Returns:
            A list of :class:`QuantumCircuit`.

        Raises:
            QiskitError: If the ``interleaved_element`` provided to the constructor
                cannot be transpiled.
        """
        # Convert interleaved element to transpiled circuit operation and store it for speed
        self.__set_up_interleaved_op()

        # Build circuits of reference sequences
        reference_sequences = self._sample_sequences()
        reference_circuits = self._sequences_to_circuits(reference_sequences)
        for circ, seq in zip(reference_circuits, reference_sequences):
            circ.metadata = {
                "xval": len(seq),
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": False,
                "pos": 0,
            }
        # Build circuits of interleaved sequences
        interleaved_sequences_1 = []
        for seq in reference_sequences:
            new_seq = []
            for elem in seq:
                new_seq.append(elem)
                new_seq.append(self._interleaved_cliff_1)
            interleaved_sequences_1.append(new_seq)
        interleaved_circuits_1 = self._sequences_to_circuits(interleaved_sequences_1)
        for circ, seq in zip(interleaved_circuits_1, reference_sequences):
            circ.metadata = {
                "xval": len(seq),  # set length of the reference sequence
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": True,
                "pos": 1,
            }
        interleaved_sequences_2 = []
        for seq in reference_sequences:
            new_seq = []
            for elem in seq:
                new_seq.append(elem)
                new_seq.append(self._interleaved_cliff_2)
            interleaved_sequences_2.append(new_seq)
        interleaved_circuits_2 = self._sequences_to_circuits(interleaved_sequences_2)
        for circ, seq in zip(interleaved_circuits_2, reference_sequences):
            circ.metadata = {
                "xval": len(seq),  # set length of the reference sequence
                "group": "Clifford",
                "physical_qubits": self.physical_qubits,
                "interleaved": True,
                "pos": 2,
            }
            # Default order: RIIRIIRII
        return list(
            itertools.chain.from_iterable(
                zip(reference_circuits, interleaved_circuits_1, interleaved_circuits_2)
            )
        )

    def _to_instruction(
        self, elem: SequenceElementType, basis_gates: Optional[Tuple[str]] = None
    ) -> Instruction:
        if elem is self._interleaved_cliff_1:
            return self._interleaved_op_1
        if elem is self._interleaved_cliff_2:
            return self._interleaved_op_2

        return super()._to_instruction(elem, basis_gates)

    def __set_up_interleaved_op(self) -> None:
        self._interleaved_op_1 = self.__get_interleaved_op_from_element(self._interleaved_element_1)
        self._interleaved_op_2 = self.__get_interleaved_op_from_element(self._interleaved_element_2)

    def __get_interleaved_op_from_element(self, interleaved_element):
        # Convert interleaved element to transpiled circuit operation and store it for speed
        interleaved_op = copy.deepcopy(interleaved_element)
        basis_gates = self._get_basis_gates()
        # Convert interleaved element to circuit
        if isinstance(interleaved_op, Clifford):
            interleaved_op = interleaved_op.to_circuit()

        if isinstance(interleaved_op, QuantumCircuit):
            interleaved_circ = interleaved_op
        elif isinstance(interleaved_op, Gate):
            interleaved_circ = QuantumCircuit(self.num_qubits, name=interleaved_op.name)
            interleaved_circ.append(interleaved_op, list(range(self.num_qubits)))
        else:  # Delay
            interleaved_circ = []

        if basis_gates and any(i.operation.name not in basis_gates for i in interleaved_circ):
            # Transpile circuit with non-basis gates and remove idling qubits
            try:
                interleaved_circ = transpile(
                    interleaved_circ, self.backend, **vars(self.transpile_options)
                )
            except TranspilerError as err:
                raise QiskitError("Failed to transpile interleaved_element.") from err
            interleaved_circ = _truncate_inactive_qubits(
                interleaved_circ, active_qubits=interleaved_circ.qubits[: self.num_qubits]
            )
            # Convert transpiled circuit to operation
            if len(interleaved_circ) == 1:
                interleaved_op = interleaved_circ.data[0].operation
            else:
                interleaved_op = interleaved_circ

        # Store interleaved operation as Instruction
        if isinstance(interleaved_op, QuantumCircuit):
            if not interleaved_op.name.startswith("Clifford"):
                interleaved_op.name = f"Clifford-{interleaved_op.name}"
            interleaved_op = interleaved_op.to_instruction()

        return interleaved_op
