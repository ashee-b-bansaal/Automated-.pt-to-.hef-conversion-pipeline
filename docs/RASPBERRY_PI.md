# Raspberry Pi + Hailo deployment

## Install and identify

For a Raspberry Pi 5 with Hailo-8/Hailo-8L vision hardware:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y hailo-all
sudo reboot
hailortcli fw-control identify
lspci | grep -i hailo
```

Do not use `hailo-h10-all` unless the device is the Hailo-10H AI HAT+ 2.

## Copy a HEF

```bash
ssh pi@raspberrypi.local 'mkdir -p ~/models'
scp outputs/*.hef pi@raspberrypi.local:~/models/
```

## Inspect and benchmark

```bash
hailortcli parse-hef ~/models/model.hef
hailortcli benchmark ~/models/model.hef
./scripts/benchmark_hefs.sh ~/models
```

`hailortcli benchmark` measures the accelerator-side network performance. It is
not camera-to-display latency. For a real system also measure capture, resize,
color conversion, input transfer, postprocessing, drawing, and queueing.

## Classification inference

Copy `scripts/run_classification.py` to the Pi and ensure its preprocessing
matches training and the compile-time normalization. A numerically correct HEF
can still produce wrong predictions if RGB/BGR order, resize/crop, mean/std, or
label ordering differs.
