import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rvp import AllocationData, AllocationProblem, CapacityDimension, DesignSpace
from rvp.constraints import CoverageConstraint
from rvp.design.cost import CostSurface
from rvp.design.dimension import DesignDimension
from rvp.design.dimensions import PredictionSetDimension
from rvp.plotting import plot_welfare_surface
from rvp.policies import RankingPolicy
from rvp.utilities import PartitionedUtility


class _PredictionScaleDimension(DesignDimension):
    name = "prediction_scale"
    target_component = "data"

    def at(self, problem, theta):
        dfs = []
        for dataset_idx in range(problem.data.n_datasets):
            df = problem.data.get_dataset(dataset_idx).df_single.copy()
            df["predictions"] = df["predictions"] * float(theta)
            dfs.append(df)
        return AllocationProblem(
            data=AllocationData(dfs if len(dfs) > 1 else dfs[0]),
            utility=problem.utility,
            constraint=problem.constraint,
            policy=problem.policy,
        )

    @property
    def bounds(self):
        return (0.5, 1.5)


def _problem():
    data = AllocationData(
        pd.DataFrame(
            {
                "predictions": [0.1, 0.2, 0.3, 0.4],
                "ground_truth": [0.0, 1.0, 2.0, 3.0],
            }
        )
    )
    return AllocationProblem(
        data=data,
        utility=PartitionedUtility(thresholds=[0.5], values=[0.0, 1.0]),
        constraint=CoverageConstraint(max_coverage=0.5, population_size=data.n),
        policy=RankingPolicy(ascending=False),
    )


def test_plot_welfare_surface_returns_figure_and_axis():
    capacity = CapacityDimension(bounds=(0.25, 0.75), name="alpha")
    prediction_scale = _PredictionScaleDimension()
    space = DesignSpace(_problem(), [capacity, prediction_scale])

    fig, ax = plot_welfare_surface(
        space,
        dim_x=capacity,
        dim_y=prediction_scale,
        n=3,
        contour_labels=False,
        cbar_label="Welfare",
    )

    assert fig is ax.figure
    assert ax.get_xlabel() == "alpha"
    assert ax.get_ylabel() == "prediction_scale"
    plt.close(fig)


def test_plot_welfare_surface_draws_discrete_contour_guides():
    capacity = CapacityDimension(bounds=(0.25, 1.0), name="alpha")
    prediction_set = PredictionSetDimension(
        name="features",
        data_by_value={
            "random": _problem().data,
            "model": AllocationData(
                pd.DataFrame(
                    {
                        "predictions": [0.4, 0.3, 0.2, 0.1],
                        "ground_truth": [0.0, 1.0, 2.0, 3.0],
                    }
                )
            ),
        },
    )
    space = DesignSpace(_problem(), [capacity, prediction_set])

    fig, ax = plot_welfare_surface(
        space,
        dim_x=capacity,
        dim_y=prediction_set,
        n=4,
        contour_levels=np.array([0.25]),
        contour_labels=False,
        discrete_contour_guides=True,
    )

    guides = ax._rvp_discrete_contour_guides
    assert guides["orientation"] == "y_discrete"
    assert len(guides["levels"]) == 1
    plt.close(fig)


def test_plot_welfare_surface_draws_status_quo_and_cost_overlays():
    capacity = CapacityDimension(bounds=(0.25, 1.0), name="alpha")
    prediction_scale = _PredictionScaleDimension()
    space = DesignSpace(_problem(), [capacity, prediction_scale])
    status_quo = {capacity: 0.5, prediction_scale: 1.0}
    cost_surface = CostSurface(lambda config: config[capacity] + config[prediction_scale])

    fig, ax = plot_welfare_surface(
        space,
        dim_x=capacity,
        dim_y=prediction_scale,
        status_quo=status_quo,
        cost_surface=cost_surface,
        n=5,
        contour_labels=False,
    )

    assert len(ax.lines) >= 1  # status-quo star
    assert r"$\nabla V_*$" in [text.get_text() for text in ax.texts]
    assert r"$\nabla c$" in [text.get_text() for text in ax.texts]
    assert len(ax.collections) >= 4  # heatmap, welfare contours, and cost frontier
    plt.close(fig)
