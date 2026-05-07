#!/usr/bin/env python3
"""CDK app entry point for the Splendor game deployment."""

import pathlib
import sys

# Ensure the project root is on sys.path so `infra` package is importable.
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import aws_cdk as cdk

from infra.stack import SplendorStack

app = cdk.App()

SplendorStack(
    app,
    "SplendorStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-west-2",
    ),
)

app.synth()
