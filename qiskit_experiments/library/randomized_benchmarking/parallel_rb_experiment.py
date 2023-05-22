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
Parallel Standard RB Experiment class.
"""
import logging
from typing import Union, Iterable, Optional, Sequence

from numpy.random import Generator
from numpy.random.bit_generator import BitGenerator, SeedSequence

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit_experiments.framework import ParallelExperiment
from .standard_rb import StandardRB
from .clifford_utils import _decompose_clifford_ops, _circuit_compose

LOG = logging.getLogger(__name__)


class ParallelStandardRB(ParallelExperiment):
    """Parallel Standard randomized benchmarking experiment."""

    def __init__(
        self,
        qubits_list: Sequence[Sequence[int]],
        lengths: Iterable[int],
        backend: Optional[Backend] = None,
        num_samples: int = 3,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        full_sampling: Optional[bool] = True,
        parallel_full_sampling: Optional[bool] = False,
    ):
        """Initialize a parallel standard randomized benchmarking experiment.

        A common Clifford sequence is used for all qubits for each length.
        If necessary to sample a different Clifford sequence for each qubit,
        use ordinary ``ParallelExperiment`` with ``StandardRB``s.

        Args:
            qubits_list: A list of physical qubits for the experiment.
                Each physical qubits must have the same length and has no overlap.
            lengths: A list of RB sequences lengths.
            backend: The backend to run the experiment on.
            num_samples: Number of samples to generate for each sequence length.
            seed: Optional, seed used to initialize ``numpy.random.default_rng``.
                  when generating circuits. The ``default_rng`` will be initialized
                  with this seed value everytime :meth:`circuits` is called.
            full_sampling: If True all Cliffords are independently sampled for all lengths.
                           If False for sample of lengths longer sequences are constructed
                           by appending additional samples to shorter sequences.
                           The default is True.
            parallel_full_sampling: If True, a different Clifford sequence is sampled for each qubit.
                           If False, a common Clifford sequence is used for all qubits.
                           The default is False.

        Raises:
            QiskitError: if any invalid argument is supplied.
        """
        common_num_qubits = len(qubits_list[0])
        if not all(len(qubits) == common_num_qubits for qubits in qubits_list):
            raise QiskitError("The number of physical qubits in qubits_list must be the same.")

        used_qubits = set()
        for qubits in qubits_list:
            for q in qubits:
                if q in used_qubits:
                    raise QiskitError(f"Duplicated qubit ({q}) in qubits_list is not allowed.")
                used_qubits.add(q)

        experiments = []
        for i, qubits in enumerate(qubits_list):
            if parallel_full_sampling:
                seed = seed + i
            srb = StandardRB(
                physical_qubits=qubits,
                lengths=lengths,
                backend=backend,
                num_samples=num_samples,
                seed=seed,
                full_sampling=full_sampling,
            )
            srb.set_experiment_options(lengths=list(lengths), num_samples=num_samples, seed=seed)
            srb.analysis.set_options(outcome="0" * srb.num_qubits)
            experiments.append(srb)

        # Initialize base experiment
        super().__init__(
            experiments=experiments,
            backend=backend,
            flatten_results=False
        )

        self._parallel_full_sampling = parallel_full_sampling

    def circuits(self):
        if self._parallel_full_sampling:
            return super()._combined_circuits(device_layout=False)

        # Use common circuits for all qubits
        common_circuits = self._experiments[0].circuits()
        num_qubits = 1 + max(self.physical_qubits)
        joint_circuits = []
        for circ_idx, sub_circ in enumerate(common_circuits):
            sub_circ.remove_final_measurements(inplace=True)
            sub_circ = _decompose_clifford_ops(sub_circ)  # Unrolled circuits
            circuit = QuantumCircuit(num_qubits, name=f"parallel_exp_{circ_idx}", metadata={})
            for sub_exp in self._experiments:
                circuit = _circuit_compose(circuit, sub_circ, qubits=sub_exp.physical_qubits)
                # circuit.compose(sub_circ, qubits=sub_exp.physical_qubits, inplace=True)
            circuit.measure_all()

            # Add subcircuit metadata
            circuit.metadata["composite_index"] = []
            circuit.metadata["composite_metadata"] = []
            circuit.metadata["composite_qubits"] = []
            circuit.metadata["composite_clbits"] = []
            for exp_idx, sub_exp in enumerate(self._experiments):
                clbits = [circuit.clbits[q] for q in sub_exp.physical_qubits]
                circuit.metadata["composite_index"].append(exp_idx)
                circuit.metadata["composite_metadata"].append(sub_circ.metadata)
                circuit.metadata["composite_qubits"].append(sub_exp.physical_qubits)
                circuit.metadata["composite_clbits"].append(clbits)

            # Add the calibrations
            for gate, cals in sub_circ.calibrations.items():
                for key, sched in cals.items():
                    circuit.add_calibration(gate, qubits=key[0], schedule=sched, params=key[1])

            joint_circuits.append(circuit)

        return joint_circuits

    def _transpiled_circuits(self):
        if self._parallel_full_sampling:
            return super()._combined_circuits(device_layout=True)

        return self.circuits()
