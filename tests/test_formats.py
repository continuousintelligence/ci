"""Public forecast containers, tabular labels, and input-device placement."""

import numpy as np
import pandas as pd
import pytest
import torch
from test_forecast import TTA, Probe

from continuousintelligence import load_model
from continuousintelligence.ci_proto import Forecaster


@pytest.mark.parametrize("cohort", [False, True])
@pytest.mark.parametrize("kind", ["numpy", "dataframe", "series", "csv_path", "csv_string"])
def test_forecast_returns_expected_container(release_files, tmp_path, kind, cohort):
    model = load_model(*release_files, device="cpu")
    history = torch.tensor([[0.25, -0.5, torch.nan, 0.75], [1.5, -2.5, 3.5, -4.5]])
    if kind == "series":
        history = history[:1]
    labels = ["load", "temperature"][: len(history)]
    frame = pd.DataFrame(history.numpy().T, columns=labels)
    if kind == "numpy":
        source = history.numpy().copy()
        # Reversed arrays can have negative strides; read-only inputs are valid.
        source = source[:, ::-1].copy()[:, ::-1]
        source.setflags(write=False)
    elif kind == "series":
        source = frame["load"]
    elif kind == "dataframe":
        source = frame.copy()
    else:
        path = tmp_path / "history.csv"
        frame.to_csv(path, index=False)
        source = path if kind == "csv_path" else str(path)
    arguments = {"series_ids": list(reversed(labels))} if cohort else {}
    expected = model.forecast(history, 3, **arguments).numpy()
    actual = model.forecast(source, 3, **arguments)
    if kind == "numpy":
        assert isinstance(actual, np.ndarray) and actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source, history.numpy())
    else:
        assert isinstance(actual, pd.DataFrame)
        pd.testing.assert_index_equal(actual.index, pd.RangeIndex(3, name="step"))
        assert actual.columns.names == ["variable", "quantile"]
        assert actual.columns.tolist() == [(label, q) for label in labels for q in (0.1, 0.5, 0.9)]
        np.testing.assert_array_equal(actual.to_numpy(), expected.transpose(2, 0, 1).reshape(3, -1))
        assert all(dtype == np.float32 for dtype in actual.dtypes)
        if kind == "series":
            pd.testing.assert_series_equal(source, frame["load"])
        elif kind == "dataframe":
            pd.testing.assert_frame_equal(source, frame)
        else:
            pd.testing.assert_frame_equal(pd.read_csv(path), frame, check_dtype=False)


@pytest.mark.parametrize("cohort", [False, True])
def test_covariates_accept_mixed_formats(tmp_path, cohort):
    model = Forecaster(Probe(), TTA)
    history = torch.tensor([[0.25, -0.5, 0.75, -0.25]])
    past = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    future = torch.tensor([[0, 0, 1, 1, 0, 1]])
    arguments = {"series_ids": ["load"]} if cohort else {}
    expected = model.forecast(
        history,
        2,
        past_covariates=past,
        future_covariates=future,
        categorical_future=(0,),
        **arguments,
    )
    future_path = tmp_path / "future.csv"
    pd.DataFrame(future.numpy().T).to_csv(future_path, index=False)
    actual = model.forecast(
        history.numpy(),
        2,
        past_covariates=pd.Series(past[0].numpy()),
        future_covariates=future_path,
        categorical_future=(0,),
        **arguments,
    )
    np.testing.assert_array_equal(actual, expected.numpy())


def test_nullable_numeric_pandas_columns():
    frame = pd.DataFrame(
        {
            "load": pd.Series([0.25, pd.NA, -0.5, 0.75], dtype="Float64"),
            "count": pd.Series([1, 2, pd.NA, 4], dtype="Int64"),
            "flag": pd.Series([True, False, pd.NA, True], dtype="boolean"),
        }
    )
    before = frame.copy(deep=True)
    model = Forecaster(Probe(), TTA)
    expected = model.forecast(
        torch.tensor([[0.25, torch.nan, -0.5, 0.75], [1, 2, torch.nan, 4], [1, 0, torch.nan, 1]]),
        2,
    ).numpy()
    actual = model.forecast(frame, 2)
    np.testing.assert_array_equal(actual.to_numpy(), expected.transpose(2, 0, 1).reshape(2, -1))
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("kind", ["numpy", "dataframe", "csv"])
def test_non_tensor_history_selects_long_model(tmp_path, kind):
    standard = Probe()
    long = Probe([[11.0, 12.0, 13.0]])
    long.config.context_len = 32
    model = Forecaster(standard, {**TTA, "postprocess": False})
    model._long_model = lambda: Forecaster(long, {**TTA, "postprocess": False})
    history = np.sin(np.arange(17, dtype=np.float32))[None]
    if kind != "numpy":
        history = pd.DataFrame(history.T, columns=["load"])
    if kind == "csv":
        path = tmp_path / "history.csv"
        history.to_csv(path, index=False)
        history = path
    model.forecast(history, 2)
    assert not standard.batches
    assert long.batches[0]["x"].shape[-1] == 32 + long.config.future_len


@pytest.mark.parametrize(
    "history",
    [
        np.ones(4),
        np.empty((0, 4)),
        [[1, 2, 3]],
        pd.DataFrame({"load": [1.0], "label": ["bad"]}),
        np.array([[complex(1, 2)]]),
    ],
)
def test_invalid_formats_fail_before_inference(history):
    neural = Probe()
    with pytest.raises(ValueError, match="history"):
        Forecaster(neural, TTA).forecast(history, 2)
    assert not neural.batches


@pytest.mark.parametrize("name", ["history", "past_covariates", "future_covariates"])
@pytest.mark.parametrize("dated", [False, True])
def test_csv_rejects_unnamed_columns_before_inference(tmp_path, name, dated):
    length = 6 if name == "future_covariates" else 4
    frame = pd.DataFrame({"load": np.arange(length)})
    if dated:
        frame.index = pd.date_range("2026-11-01", periods=length, freq="h")
    path = tmp_path / "history.csv"
    frame.to_csv(path)
    neural = Probe()
    arguments = {"history": torch.ones(1, 4), name: path}
    with pytest.raises(ValueError, match="unnamed column.*index=False.*timestamp"):
        Forecaster(neural, TTA).forecast(**arguments, horizon=2)
    assert not neural.batches


@pytest.mark.parametrize("count", [9, 99])
@pytest.mark.parametrize("kind", ["dataframe", "series", "csv"])
def test_quantile_labels_are_probabilities(tmp_path, count, kind):
    neural = Probe([list(range(count))])
    neural.quantiles = torch.arange(1, count + 1, dtype=torch.float32) / (count + 1)
    frame = pd.DataFrame({"load": [0.25, -0.5, 0.75, -0.25]})
    source = frame["load"] if kind == "series" else frame
    if kind == "csv":
        source = tmp_path / "history.csv"
        frame.to_csv(source, index=False)
    actual = Forecaster(neural, TTA).forecast(source, 3)
    lower = actual.xs(0.1, level="quantile", axis=1)["load"]
    upper = actual[("load", 0.9)]
    pd.testing.assert_series_equal(lower, actual.iloc[:, (count + 1) // 10 - 1], check_names=False)
    pd.testing.assert_series_equal(
        upper, actual.iloc[:, 9 * (count + 1) // 10 - 1], check_names=False
    )
    assert actual.columns.get_level_values("quantile").tolist() == [
        i / (count + 1) for i in range(1, count + 1)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
@pytest.mark.parametrize("cohort", [False, True])
@pytest.mark.parametrize(
    "model_device,input_device",
    [
        ("cuda:0", "cpu"),
        ("cuda:0", "cuda:0"),
        ("cpu", "cuda:0"),
        pytest.param(
            "cuda:0",
            "cuda:1",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Two CUDA devices are required"
            ),
        ),
    ],
)
def test_tensor_output_keeps_input_device(release_files, model_device, input_device, cohort):
    model = load_model(*release_files, device="cpu")
    model._model.to(model_device)
    history = torch.tensor(
        [[0.25, -0.5, 0.75, -0.25]], device=input_device, dtype=torch.float64, requires_grad=True
    )
    arguments = {"series_ids": ["load"]} if cohort else {}
    current = torch.cuda.current_device()
    expected = model.forecast(history.detach().cpu(), 3, **arguments)
    actual = model.forecast(history, 3, **arguments)
    assert actual.device == history.device
    assert actual.dtype == torch.float32 and not actual.requires_grad
    assert model.device == torch.device(model_device)
    assert torch.cuda.current_device() == current
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
@pytest.mark.parametrize("cohort", [False, True])
def test_gpu_model_returns_host_numpy_and_pandas(release_files, cohort):
    model = load_model(*release_files, device="cpu")
    model._model.cuda()
    history = np.array([[0.25, -0.5, 0.75, -0.25]], dtype=np.float32)
    arguments = {"series_ids": ["load"]} if cohort else {}
    expected = model.forecast(torch.from_numpy(history), 3, **arguments)
    actual = model.forecast(history, 3, **arguments)
    np.testing.assert_array_equal(actual, expected.numpy())
    frame = model.forecast(pd.DataFrame(history.T, columns=["load"]), 3, **arguments)
    np.testing.assert_array_equal(frame["load"].to_numpy().T, actual[0])
