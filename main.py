import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))

batch_files = sorted(glob.glob(os.path.join("datasets", "batch_*.csv")))

dfs = []

for filepath in batch_files:
    batch_id = int(os.path.basename(filepath).split("_")[1].split(".")[0])
    df_batch = pd.read_csv(filepath)
    df_batch["Batch ID"] = batch_id
    dfs.append(df_batch)

df = pd.concat(dfs, ignore_index=True)

batch_ids = sorted(df["Batch ID"].unique())

df_ic_selected = df_ic[df_ic["Batch ID"].isin(batch_ids)].copy()

df_ic_selected["Cost [$/L]"] = (
    10 * df_ic_selected["Glucose [mM]"]
    + 20 * df_ic_selected["Glutamine [mM]"]
    + 0.0005 * df_ic_selected["VCD [cells/mL]"]
)

df = df.merge(
    df_ic_selected[
        [
            "Batch ID",
            "Glucose [mM]",
            "Glutamine [mM]",
            "VCD [cells/mL]",
            "Cost [$/L]"
        ]
    ],
    on="Batch ID",
    suffixes=("", " Initial")
)

df["Harvest Time [d]"] = df["Time [h]"] / 24
df["Value [$/L]"] = 1000 * df["mAb [mg/L]"]
df["Profit [$/L]"] = df["Value [$/L]"] - df["Cost [$/L]"]
df["Space-Time Profit [$/L/d]"] = (
    df["Profit [$/L]"] / (df["Harvest Time [d]"] + 1)
)
df["Space-Time Yield [mg/L/d]"] = (
    df["mAb [mg/L]"] / (df["Harvest Time [d]"] + 1)
)
degrees = [1, 2, 3]
degree_results = []

for degree in degrees:
    degree_rmses = []

    for test_batch in batch_ids:
        train_df = df[df["Batch ID"] != test_batch]
        test_df = df[df["Batch ID"] == test_batch]

        x_train = train_df[
            [
                "Glucose [mM] Initial",
                "Glutamine [mM] Initial",
                "VCD [cells/mL] Initial",
                "Harvest Time [d]"
            ]
        ].values

        y_train = train_df["mAb [mg/L]"].values

        x_test = test_df[
            [
                "Glucose [mM] Initial",
                "Glutamine [mM] Initial",
                "VCD [cells/mL] Initial",
                "Harvest Time [d]"
            ]
        ].values

        y_test = test_df["mAb [mg/L]"].values

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree)),
            ("model", LinearRegression())
        ])

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        degree_rmses.append(rmse)

    degree_results.append([
        degree,
        np.mean(degree_rmses),
        np.max(degree_rmses)
    ])

degree_results = pd.DataFrame(
    degree_results,
    columns=[
        "Degree",
        "Mean RMSE [mg/L]",
        "Maximum RMSE [mg/L]"
    ]
)
degree_results.to_csv(
    os.path.join("tables", "model_comparison.csv"),
    index=False
)
plt.figure()

plt.plot(
    degree_results["Degree"],
    degree_results["Mean RMSE [mg/L]"],
    marker="o"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean RMSE [mg/L]")
plt.xticks(degree_results["Degree"])
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "polynomial_degree_validation.png"),
    dpi=300
)

plt.close()
print()
print("Polynomial degree comparison:")
print(degree_results)
x_all = df[
    [
        "Glucose [mM] Initial",
        "Glutamine [mM] Initial",
        "VCD [cells/mL] Initial",
        "Harvest Time [d]"
    ]
].values

y_all = df["mAb [mg/L]"].values

model_final = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=3)),
    ("model", LinearRegression())
])

model_final.fit(x_all, y_all)

glucose_levels = [5, 10, 15, 20, 25, 30, 35]
glutamine_levels = [1, 2, 3, 4, 5, 6, 7]
vcd_levels = [
    200000,
    400000,
    600000,
    800000,
    1000000,
    1200000,
    1400000
]

harvest_times = sorted(
    df.loc[df["Harvest Time [d]"] > 0, "Harvest Time [d]"].unique()
)

optimization_results = []

for glucose in glucose_levels:
    for glutamine in glutamine_levels:
        for vcd in vcd_levels:
            cost = 10 * glucose + 20 * glutamine + 0.0005 * vcd

            for harvest_time in harvest_times:
                x_candidate = np.array([[
                    glucose,
                    glutamine,
                    vcd,
                    harvest_time
                ]])

                mab = model_final.predict(x_candidate)[0]
                value = 1000 * mab
                profit = value - cost
                stp = profit / (harvest_time + 1)
                sty = mab / (harvest_time + 1)

                optimization_results.append([
                    glucose,
                    glutamine,
                    vcd,
                    harvest_time,
                    mab,
                    cost,
                    value,
                    profit,
                    stp,
                    sty
                ])

optimization_results = pd.DataFrame(
    optimization_results,
    columns=[
        "Glucose [mM]",
        "Glutamine [mM]",
        "VCD [cells/mL]",
        "Harvest Time [d]",
        "mAb [mg/L]",
        "Cost [$/L]",
        "Value [$/L]",
        "Profit [$/L]",
        "Space-Time Profit [$/L/d]",
        "Space-Time Yield [mg/L/d]"
    ]
)

best_final = optimization_results.loc[
    optimization_results["Space-Time Profit [$/L/d]"].idxmax()
]

print()
print("Optimal operating conditions:")
print(best_final)
validation_residuals = []

for test_batch in batch_ids:
    train_df = df[df["Batch ID"] != test_batch]
    test_df = df[df["Batch ID"] == test_batch]

    x_train = train_df[
        [
            "Glucose [mM] Initial",
            "Glutamine [mM] Initial",
            "VCD [cells/mL] Initial",
            "Harvest Time [d]"
        ]
    ].values

    y_train = train_df["mAb [mg/L]"].values

    x_test = test_df[
        [
            "Glucose [mM] Initial",
            "Glutamine [mM] Initial",
            "VCD [cells/mL] Initial",
            "Harvest Time [d]"
        ]
    ].values

    y_test = test_df["mAb [mg/L]"].values

    model_validation = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=3)),
        ("model", LinearRegression())
    ])

    model_validation.fit(x_train, y_train)
    y_pred = model_validation.predict(x_test)

    validation_residuals.extend(y_test - y_pred)

validation_residuals = np.array(validation_residuals)

lower_residual = np.percentile(validation_residuals, 2.5)
upper_residual = np.percentile(validation_residuals, 97.5)

mab_lower = max(
    0,
    best_final["mAb [mg/L]"] + lower_residual
)

mab_upper = (
    best_final["mAb [mg/L]"] + upper_residual
)

value_lower = 1000 * mab_lower
value_upper = 1000 * mab_upper

stp_lower = (
    value_lower - best_final["Cost [$/L]"]
) / (best_final["Harvest Time [d]"] + 1)

stp_upper = (
    value_upper - best_final["Cost [$/L]"]
) / (best_final["Harvest Time [d]"] + 1)

sty_lower = (
    mab_lower / (best_final["Harvest Time [d]"] + 1)
)

sty_upper = (
    mab_upper / (best_final["Harvest Time [d]"] + 1)
)

print()
print("95% prediction intervals:")
print(
    f"Value: ${best_final['Value [$/L]']:.2f}/L "
    f"(${value_lower:.2f} to ${value_upper:.2f})"
)
print(
    f"STP: ${best_final['Space-Time Profit [$/L/d]']:.2f}/L/d "
    f"(${stp_lower:.2f} to ${stp_upper:.2f})"
)
print(
    f"STY: {best_final['Space-Time Yield [mg/L/d]']:.4f} mg/L/d "
    f"({sty_lower:.4f} to {sty_upper:.4f})"
)
optimization_summary = pd.DataFrame({
    "Metric": [
        "Initial Glucose [mM]",
        "Initial Glutamine [mM]",
        "Initial VCD [cells/mL]",
        "Harvest Time [d]",
        "mAb [mg/L]",
        "Cost [$/L]",
        "Value [$/L]",
        "Value 95% PI Lower [$/L]",
        "Value 95% PI Upper [$/L]",
        "STP [$/L/d]",
        "STP 95% PI Lower [$/L/d]",
        "STP 95% PI Upper [$/L/d]",
        "STY [mg/L/d]",
        "STY 95% PI Lower [mg/L/d]",
        "STY 95% PI Upper [mg/L/d]"
    ],
    "Value": [
        best_final["Glucose [mM]"],
        best_final["Glutamine [mM]"],
        best_final["VCD [cells/mL]"],
        best_final["Harvest Time [d]"],
        best_final["mAb [mg/L]"],
        best_final["Cost [$/L]"],
        best_final["Value [$/L]"],
        value_lower,
        value_upper,
        best_final["Space-Time Profit [$/L/d]"],
        stp_lower,
        stp_upper,
        best_final["Space-Time Yield [mg/L/d]"],
        sty_lower,
        sty_upper
    ]
})

optimization_summary.to_csv(
    os.path.join("tables", "optimization_summary.csv"),
    index=False
)
optimal_harvest = optimization_results[
    (optimization_results["Glucose [mM]"] == best_final["Glucose [mM]"])
    & (optimization_results["Glutamine [mM]"] == best_final["Glutamine [mM]"])
    & (optimization_results["VCD [cells/mL]"] == best_final["VCD [cells/mL]"])
]

plt.figure()

plt.plot(
    optimal_harvest["Harvest Time [d]"],
    optimal_harvest["Space-Time Profit [$/L/d]"],
    marker="o"
)

plt.xlabel("Harvest Time [d]")
plt.ylabel("Space-Time Profit [$/L/d]")
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "stp_vs_harvest_time.png"),
    dpi=300
)

plt.close()
glucose_effect = optimization_results[
    (optimization_results["Glutamine [mM]"] == best_final["Glutamine [mM]"])
    & (optimization_results["VCD [cells/mL]"] == best_final["VCD [cells/mL]"])
    & (optimization_results["Harvest Time [d]"] == best_final["Harvest Time [d]"])
]

plt.figure()

plt.plot(
    glucose_effect["Glucose [mM]"],
    glucose_effect["Space-Time Profit [$/L/d]"],
    marker="o"
)

plt.xlabel("Initial Glucose [mM]")
plt.ylabel("Space-Time Profit [$/L/d]")
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "stp_vs_glucose.png"),
    dpi=300
)

plt.close()
glutamine_effect = optimization_results[
    (optimization_results["Glucose [mM]"] == best_final["Glucose [mM]"])
    & (optimization_results["VCD [cells/mL]"] == best_final["VCD [cells/mL]"])
    & (optimization_results["Harvest Time [d]"] == best_final["Harvest Time [d]"])
]

plt.figure()

plt.plot(
    glutamine_effect["Glutamine [mM]"],
    glutamine_effect["Space-Time Profit [$/L/d]"],
    marker="o"
)

plt.xlabel("Initial Glutamine [mM]")
plt.ylabel("Space-Time Profit [$/L/d]")
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "stp_vs_glutamine.png"),
    dpi=300
)

plt.close()
vcd_effect = optimization_results[
    (optimization_results["Glucose [mM]"] == best_final["Glucose [mM]"])
    & (optimization_results["Glutamine [mM]"] == best_final["Glutamine [mM]"])
    & (optimization_results["Harvest Time [d]"] == best_final["Harvest Time [d]"])
]

plt.figure()

plt.plot(
    vcd_effect["VCD [cells/mL]"],
    vcd_effect["Space-Time Profit [$/L/d]"],
    marker="o"
)

plt.xlabel("Initial VCD [cells/mL]")
plt.ylabel("Space-Time Profit [$/L/d]")
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "stp_vs_vcd.png"),
    dpi=300
)

plt.close()
plt.figure()

plt.plot(
    optimal_harvest["Harvest Time [d]"],
    optimal_harvest["Space-Time Yield [mg/L/d]"],
    marker="o"
)

plt.xlabel("Harvest Time [d]")
plt.ylabel("Space-Time Yield [mg/L/d]")
plt.tight_layout()

plt.savefig(
    os.path.join("figures", "sty_vs_harvest_time.png"),
    dpi=300
)

plt.close()
