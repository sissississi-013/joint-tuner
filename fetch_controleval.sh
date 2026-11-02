#!/usr/bin/env bash
# ControlEval (ControlAgent, arXiv 2410.19811) is not redistributed here. Fetch it from upstream.
set -e; mkdir -p data
[ -d data/controlagent ] || git clone --depth 1 https://github.com/ControlAgent/ControlAgent.git data/controlagent
