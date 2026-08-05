# HW 6 — OpenCL Image Convolution

An OpenCL image-convolution pipeline covering both the device kernel and host-side execution flow.

## Highlights

- Buffer creation and host-device transfers
- Kernel argument and NDRange setup
- Boundary-aware convolution in the OpenCL kernel

`src/host_fe.c` contains the host pipeline and `src/kernel.cl` contains the device kernel. Course headers and image data are not redistributed.
