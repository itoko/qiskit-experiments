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
Interleaved RB analysis class.
"""
from typing import List, Union

import lmfit
import numpy as np

import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import AnalysisResultData, ExperimentData


class DoubleInterleavedRBAnalysis(curve.CurveAnalysis):
    r"""A class to analyze interleaved randomized benchmarking experiment.

    # section: overview
        This analysis takes only two series for standard and interleaved RB curve fitting.
        From the fit :math:`\alpha` and :math:`\alpha_c` value this analysis estimates
        the error per Clifford (EPC) of the interleaved gate.

        The EPC estimate is obtained using the equation

        .. math::

            r_{\mathcal{C}}^{\text{est}} =
                \frac{\left(d-1\right)\left(1-\alpha_{\overline{\mathcal{C}}}/\alpha\right)}{d}

        The systematic error bounds are given by

        .. math::

            E = \min\left\{
                \begin{array}{c}
                    \frac{\left(d-1\right)\left[\left|\alpha-\alpha_{\overline{\mathcal{C}}}\right|
                    +\left(1-\alpha\right)\right]}{d} \\
                    \frac{2\left(d^{2}-1\right)\left(1-\alpha\right)}
                    {\alpha d^{2}}+\frac{4\sqrt{1-\alpha}\sqrt{d^{2}-1}}{\alpha}
                \end{array}
            \right.

        See Ref. [1] for more details.

    # section: fit_model
        The fit is based on the following decay functions:

        Fit model for standard RB

        .. math::

            F(x) = a \alpha^{x} + b

        Fit model for interleaved RB

        .. math::

            F(x) = a (\alpha_c \alpha)^{x_2} + b

    # section: fit_parameters
        defpar a:
            desc: Height of decay curve.
            init_guess: Determined by :math:`1 - b`.
            bounds: [0, 1]
        defpar b:
            desc: Base line.
            init_guess: Determined by :math:`(1/2)^n` where :math:`n` is number of qubit.
            bounds: [0, 1]
        defpar \alpha:
            desc: Depolarizing parameter.
            init_guess: Determined by :func:`~rb_decay` with standard RB curve.
            bounds: [0, 1]
        defpar \alpha_c:
            desc: Ratio of the depolarizing parameter of interleaved RB to standard RB curve.
            init_guess: Determined by alpha of interleaved RB curve divided by one of
                standard RB curve. Both alpha values are estimated by :func:`~rb_decay`.
            bounds: [0, 1]

    # section: reference
        .. ref_arxiv:: 1 1203.4550

    """

    def __init__(self):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a * alpha ** x + b",
                    name="standard",
                ),
                lmfit.models.ExpressionModel(
                    expr="a * (alpha_c1 * alpha) ** x + b",
                    name="interleaved-1",
                ),
                lmfit.models.ExpressionModel(
                    expr="a * (alpha_c2 * alpha) ** x + b",
                    name="interleaved-2",
                ),
            ]
        )
        self._num_qubits = None

    @classmethod
    def _default_options(cls):
        """Default analysis options."""
        default_options = super()._default_options()
        default_options.data_subfit_map = {
            "standard": {"interleaved": False, "pos": 0},
            "interleaved-1": {"interleaved": True, "pos": 1},
            "interleaved-2": {"interleaved": True, "pos": 2},
        }
        default_options.result_parameters = ["alpha", "alpha_c1", "alpha_c2"]
        default_options.average_method = "sample"
        return default_options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            List of fit options that are passed to the fitter function.
        """
        user_opt.bounds.set_if_empty(
            a=(0, 1),
            alpha=(0, 1),
            alpha_c1=(0, 1),
            alpha_c2=(0, 1),
            b=(0, 1),
        )

        b_guess = 1 / 2**self._num_qubits

        # for standard RB curve
        std_curve = curve_data.get_subset_of("standard")
        alpha_std = curve.guess.rb_decay(std_curve.x, std_curve.y, b=b_guess)
        a_std = (std_curve.y[0] - b_guess) / (alpha_std ** std_curve.x[0])

        # for interleaved RB curve
        int_curve1 = curve_data.get_subset_of("interleaved-1")
        alpha_int1 = curve.guess.rb_decay(int_curve1.x, int_curve1.y, b=b_guess)
        a_int1 = (int_curve1.y[0] - b_guess) / (alpha_int1 ** int_curve1.x[0])
        int_curve2 = curve_data.get_subset_of("interleaved-2")
        alpha_int2 = curve.guess.rb_decay(int_curve2.x, int_curve2.y, b=b_guess)
        a_int2 = (int_curve2.y[0] - b_guess) / (alpha_int2 ** int_curve2.x[0])

        alpha_c1 = min(alpha_int1 / alpha_std, 1.0)
        alpha_c2 = min(alpha_int2 / alpha_std, 1.0)

        user_opt.p0.set_if_empty(
            b=b_guess,
            a=np.mean([a_std, a_int1, a_int2]),
            alpha=alpha_std,
            alpha_c1=alpha_c1,
            alpha_c2=alpha_c2,
        )

        return user_opt

    def _create_analysis_results(
        self,
        fit_data: curve.CurveFitResult,
        quality: str,
        **metadata,
    ) -> List[AnalysisResultData]:
        """Create analysis results for important fit parameters.

        Args:
            fit_data: Fit outcome.
            quality: Quality of fit outcome.

        Returns:
            List of analysis result data.
        """
        outcomes = super()._create_analysis_results(fit_data, quality, **metadata)

        nrb = 2**self._num_qubits
        scale = (nrb - 1) / nrb

        alpha = fit_data.ufloat_params["alpha"]
        alpha_c1 = fit_data.ufloat_params["alpha_c1"]
        alpha_c2 = fit_data.ufloat_params["alpha_c2"]

        # Calculate epc_est (=r_c^est) - Eq. (4):
        epc1 = scale * (1 - alpha_c1)
        epc2 = scale * (1 - alpha_c2)

        # # Calculate the systematic error bounds - Eq. (5):
        # systematic_err_1 = scale * (abs(alpha.n - alpha_c.n) + (1 - alpha.n))
        # systematic_err_2 = (
        #     2 * (nrb * nrb - 1) * (1 - alpha.n) / (alpha.n * nrb * nrb)
        #     + 4 * (np.sqrt(1 - alpha.n)) * (np.sqrt(nrb * nrb - 1)) / alpha.n
        # )
        #
        # systematic_err = min(systematic_err_1, systematic_err_2)
        # systematic_err_l = epc.n - systematic_err
        # systematic_err_r = epc.n + systematic_err

        outcomes.append(
            AnalysisResultData(
                name="EPC1",
                value=epc1,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                # extra={
                #     "EPC_systematic_err": systematic_err,
                #     "EPC_systematic_bounds": [max(systematic_err_l, 0), systematic_err_r],
                #     **metadata,
                # },
            )
        )
        outcomes.append(
            AnalysisResultData(
                name="EPC2",
                value=epc2,
                chisq=fit_data.reduced_chisq,
                quality=quality,
                # extra={
                #     "EPC_systematic_err": systematic_err,
                #     "EPC_systematic_bounds": [max(systematic_err_l, 0), systematic_err_r],
                #     **metadata,
                # },
            )
        )

        return outcomes

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        """Initialize curve analysis with experiment data.

        This method is called ahead of other processing.

        Args:
            experiment_data: Experiment data to analyze.
        """
        super()._initialize(experiment_data)

        # Get qubit number
        self._num_qubits = len(experiment_data.metadata["physical_qubits"])
