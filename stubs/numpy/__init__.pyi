# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# pyright: reportIncompleteStub=false

# Code purpose:  Allow for tackling pyright errors related to numpy. This file is not meant to be imported directly. It
# is only meant to be used for type checking and static analysis.
from typing import Any

ndarray = Any

def __getattr__(name: str) -> Any: ...
