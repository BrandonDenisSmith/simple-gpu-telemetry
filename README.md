# Super Simple GPU Telemetry

![Example of what telemetry looks like for Ollama](telemetry.jpg)

This is a deliberately simple application for collecting telemetry on a single
GPU application. It utilizes `psutil` and `nvidia-ml-py` to collect CPU, GPU,
and RAM/VRAM utilization and presents them in a live `matplotlib` plot. It also
performs simple calculations of CPU and GPU time used and displays them with the
live telemetry. Process monitoring and logging are performed in separate threads
while the main thread handles plotting.

# Installation

I highly recommend that you set up set up a virtual environment for running .
For example, using Python 3's `venv` module:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Usage

```sh
python simple-gpu-telemetry.py [-h] [-f FREQUENCY] [-o OUTPUT] [-g GPU] process_name
```

# License

Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org/>