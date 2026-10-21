package com.justin.spikeballtracker.vision;

import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.io.InputStream;

/**
 * Finds the net in a KNN background plate with a YOLO26-nano detector.
 *
 * <p>Replaces the edge-and-contour detector. Same contract: the only caller
 * is {@link FramePipeline}, which hands over a CV_8UC3 Y,U,V plate at the
 * working resolution (854x480).
 */
public class NetDetector {
    private static final String TAG = "NetDetector";

    /** File name in app/src/main/assets. */
    public static final String MODEL_ASSET = "yolo_net.onnx";

    /** The network's fixed input size*/
    private static final int IN_W = 640;
    private static final int IN_H = 384;
    private static final double PAD_GREY = 114.0;

    /** Output values per anchor: centre x, centre y, width, height, net confidence. */
    private static final int ROWS = 5;

    /**
     * Below this the plate is treated as "no net", and the pipeline keeps the
     * box it already has.
     */
    private static final float MIN_CONFIDENCE = 0.2f;

    /**
     * The ONNX file, read once by {@link #loadModel} on the UI thread before
     * any source exists.
     */
    private static byte[] modelBytes;

    private Net net;
    private boolean netUnavailable;

    // ---- reusable buffers ----
    private final Mat rgb     = new Mat();
    private final Mat resized = new Mat();
    private final Mat padded  = new Mat();

    /**
     * Reads the model from the app's assets. Call once, from
     * {@code MainActivity.onCreate}, after OpenCV has loaded
     */
    public static void loadModel(AssetManager assets) {
        try (InputStream in = assets.open(MODEL_ASSET)) {
            modelBytes = in.readAllBytes();
        } catch (IOException e) {
            Log.e(TAG, "could not read " + MODEL_ASSET + "; net detection is off", e);
        }
    }

    /**
     * @param plate CV_8UC3 Y,U,V background plate from the KNN model, at the
     *              working resolution (854x480)
     * @return the net box in plate coordinates, or null if the model is
     *         missing or its best box is less than {@link #MIN_CONFIDENCE}
     */
    public Rect detect(Mat plate) {
        if (plate == null || plate.empty()) return null;
        Net n = net();
        if (n == null) return null;

        Mat blob = null;
        Mat out = null;
        try {
            int w = plate.cols();
            int h = plate.rows();

            double r  = Math.min(IN_W / (double) w, IN_H / (double) h);
            int nw    = (int) Math.round(w * r);
            int nh    = (int) Math.round(h * r);
            int padX  = (IN_W - nw) / 2;
            int padY  = (IN_H - nh) / 2;

            Imgproc.cvtColor(plate, rgb, Imgproc.COLOR_YUV2RGB);
            Imgproc.resize(rgb, resized, new Size(nw, nh), 0, 0, Imgproc.INTER_LINEAR);
            Core.copyMakeBorder(resized, padded,
                    padY, IN_H - nh - padY, padX, IN_W - nw - padX,
                    Core.BORDER_CONSTANT, new Scalar(PAD_GREY, PAD_GREY, PAD_GREY));

            blob = Dnn.blobFromImage(padded, 1.0 / 255.0, new Size(IN_W, IN_H),
                    new Scalar(0, 0, 0), false, false);
            n.setInput(blob);
            out = n.forward();                      // 1 x 5 x anchors, float

            int anchors = (int) (out.total() / ROWS);
            float[] data = new float[ROWS * anchors];
            out.get(new int[] {0, 0, 0}, data);     // row-major: all cx, all cy, ... all conf

            int best = -1;
            float bestConf = -1f;
            int confRow = 4 * anchors;
            for (int i = 0; i < anchors; i++) {
                float c = data[confRow + i];
                if (c > bestConf) {
                    bestConf = c;
                    best = i;
                }
            }

            if (best < 0 || bestConf < MIN_CONFIDENCE) {
                return null;
            }

            // Back from the 640x384 input to plate coordinates.
            double cx = data[best];
            double cy = data[anchors + best];
            double bw = data[2 * anchors + best];
            double bh = data[3 * anchors + best];
            double x0 = clamp((cx - bw / 2 - padX) / r, 0, w);
            double y0 = clamp((cy - bh / 2 - padY) / r, 0, h);
            double x1 = clamp((cx + bw / 2 - padX) / r, 0, w);
            double y1 = clamp((cy + bh / 2 - padY) / r, 0, h);

            Rect box = new Rect((int) Math.round(x0), (int) Math.round(y0),
                    (int) Math.round(x1 - x0), (int) Math.round(y1 - y0));
            return box.width > 0 && box.height > 0 ? box : null;

        } finally {
            if (blob != null) blob.release();
            if (out != null) out.release();
        }
    }

    /** Frees the buffers.*/
    public void release() {
        rgb.release();
        resized.release();
        padded.release();
        net = null;
    }

    private Net net() {
        if (net != null || netUnavailable) return net;

        if (modelBytes == null) {
            Log.e(TAG, "model not loaded (NetDetector.loadModel was not called)");
            netUnavailable = true;
            return null;
        }

        MatOfByte buffer = new MatOfByte(modelBytes);
        try {
            net = Dnn.readNetFromONNX(buffer);
            net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
            net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
        } catch (Exception e) {
            Log.e(TAG, "could not build the network; net detection is off", e);
            net = null;
            netUnavailable = true;
        } finally {
            buffer.release();
        }
        return net;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
