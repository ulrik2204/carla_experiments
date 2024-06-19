#! /bin/bash
Start-Process -FilePath "..\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe"
Start-Sleep -Seconds 10
poetry run python -m openpilot_exploration.carla_eval.eval_op_deepdive