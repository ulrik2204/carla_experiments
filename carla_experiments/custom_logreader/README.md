# Custom LogReader

This module implements the LogReader from Openpilot [here](https://github.com/commaai/openpilot/blob/master/tools/lib/logreader.py). It is the code necessary to read the log files from the Comma 3x device, but in a singular, smaller folder for easier porting. In addition, the code is converted to support python 3.8.