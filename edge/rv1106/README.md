# RV1106 daughter detector

This directory contains the versioned board-side implementation used with the
NAS service. Vendor SDK binaries and models are deliberately not committed.

The production pipeline is:

```text
H.264 low stream -> RockIt software VDEC -> YUV420 DMA buffer
  -> RockIVA PFP person/face detection + objId tracking at 1 FPS
  -> on-demand RetinaFace -> MobileFaceNet -> daughter.db
  -> confirmed/probable session events over MQTT
```

Build with `make -C edge/rv1106/board_service`. `LUCKFOX_SDK_DIR`,
`RKNN_SDK_DIR`, and `TOOLCHAIN_DIR` can be overridden on the make command line.
Run `rockiva_probe config.ini 30` on the board before installing a new binary.
Production enablement requires p95 detection latency below 150 ms, average CPU
below 65%, at least 80 MB available RAM, and temperature below 75 C.

The board firmware currently decodes H.264 to planar YUV420P and exposes the
RockIt buffer as a dma-buf fd. The integration passes the real pixel format and
fd directly to RockIVA; using a CPU-address buffer is unsupported on the
minimal image because its RockIVA build has no libyuv conversion dependency.

`make install` creates a self-contained `board_service/install/` directory.
Copy that directory to the board and run:

```sh
./install_on_board.sh .
```

The installer stops the old service, snapshots the current binary/config/
database, runs the live RTSP RockIVA probe, and only replaces the production
binary if the probe succeeds. It installs the persistent rollback command as
`/root/daughter_watch/rollback_on_board.sh`.
