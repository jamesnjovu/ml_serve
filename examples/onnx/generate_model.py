#!/usr/bin/env python3
"""Builds the fraud_detection.onnx used by serve.exs.

Committed alongside the model so the artifact is reproducible rather than magic. A real project
exports from scikit-learn or PyTorch instead; the graph below is written by hand only to keep
this example free of a training dependency.

    pip install onnx
    python examples/onnx/generate_model.py

The graph is a logistic regression over three normalised features:

    probability = sigmoid(features @ weights + bias)

with a dynamic first dimension, so the same model serves one row or a batch of a thousand.
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper

FEATURES = ["amount", "transaction_count_24h", "failed_transactions_24h"]

# Chosen so the three sample transactions in serve.exs land at meaningfully different scores
# rather than all saturating at 0 or 1.
WEIGHTS = np.array([[8.0], [2.0], [10.0]], dtype=np.float32)
BIAS = np.array([-2.5], dtype=np.float32)


def build() -> onnx.ModelProto:
    features = helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, ["batch", len(FEATURES)]
    )
    probability = helper.make_tensor_value_info(
        "probability", TensorProto.FLOAT, ["batch", 1]
    )

    weights = helper.make_tensor(
        "weights", TensorProto.FLOAT, WEIGHTS.shape, WEIGHTS.flatten().tolist()
    )
    bias = helper.make_tensor("bias", TensorProto.FLOAT, BIAS.shape, BIAS.tolist())

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["features", "weights"], ["logits_raw"]),
            helper.make_node("Add", ["logits_raw", "bias"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["probability"]),
        ],
        name="fraud_detection",
        inputs=[features],
        outputs=[probability],
        initializer=[weights, bias],
    )

    model = helper.make_model(
        graph,
        producer_name="ml_serve-examples",
        # Opset 13 is old enough to be universally supported and new enough for dynamic axes.
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.doc_string = "Toy fraud classifier for the MLServe ONNX example."

    # Pin the IR version. Recent `onnx` releases default to an IR version newer than the ONNX
    # Runtime bundled with Ortex accepts, and the failure is an opaque load error rather than
    # anything a reader would connect to the exporter. IR 8 is understood by every runtime that
    # supports opset 13.
    model.ir_version = 8

    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    destination = pathlib.Path(__file__).parent / "fraud_detection.onnx"
    onnx.save(build(), destination)
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")
