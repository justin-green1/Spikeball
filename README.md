# Spikeball

An Android app that tracks a spikeball ball during play. It analyses live
camera frames or a recorded video, detects the net, follows the ball, and
works out when rallies start and end. Each rally's footage can be saved to
the phone.

## Features

- **Live camera** – analyses the back camera in landscape and keeps the last
  minute of footage in memory, so a rally can be saved after it has been
  detected.
- **Video playback** – plays a recorded clip through the same pipeline at
  its true frame rate.
- **Net detection** – a YOLO26-nano model (`app/src/main/assets/yolo_net.onnx`)
  finds the net box in the static background.
- **Ball tracking** – colour and motion masks feed a blob classifier (SVM),
  and a tracker links the blobs into flights.
- **Rally detection** – a score over a sliding window, with hysteresis,
  decides where rallies start and end.
- **Save last rally** – writes the most recent rally to
  `DCIM/Spikeball Videos/` as an MP4, without re-encoding.

## Requirements

- Android 15 (API 35) or newer
- A phone with a back camera (for live analysis)
- Android Studio with the Android SDK for API 37

## Building

1. Open the `Spikeball` folder in Android Studio.
2. Let Gradle sync. Android Studio creates `local.properties` with your SDK
   path, so it isn't committed.
3. Run the `app` configuration on a device.

The app is locked to landscape. OpenCV 4.12.0 (`org.opencv:opencv`) and
CameraX are pulled in through Gradle.

## Using the app

| Button | What it does |
| --- | --- |
| **Open Camera** | Starts live analysis from the back camera. |
| **Select Video** | Opens a video file and plays it through the pipeline. |
| **Stop** | Stops whichever source is running. |
| **Clear** | Resets the learned background, net and tracks. Use it if the camera has moved. |
| **Save Last Rally** | Saves the most recent rally to `DCIM/Spikeball Videos/`. Greyed out until one is detected. |

A toast appears when a rally is detected, and another when it is saved.

## Project layout

```
app/src/main/java/com/justin/spikeball/
├── ui/          MainActivity: buttons, the single image view, rally saving
├── source/      CameraSource (live), VideoPlayer (files), listener interfaces
├── video/       Frame decoding, YUV conversion and range handling
├── vision/      Colour and motion masks, blob SVM, net detector, FramePipeline
├── tracking/    Ball tracker, track fitting, rally detector, net zone
├── record/      Replay ring buffer, rally copy, MP4 writing
└── WorkerThread the background thread helper
```

Threads: the UI thread, one **Analysis** thread (shared by both sources, so
only one runs at a time), a **Recorder** thread (camera replay buffer) and a
**Saver** thread (writing MP4s).