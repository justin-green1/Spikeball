package com.justin.spikeballtracker.vision;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * Masks the pixels inside the ball's colour box.
 */
public class ColourDetector {

    /**
     * The box in front of the blob SVM. Deliberately loose.
     */
    public static final Scalar LO = new Scalar(0, 35, 129);
    public static final Scalar HI = new Scalar(255, 124, 171);
    private final Mat colourMask = new Mat();

    /**
     * @param frame CV_8UC3 Y,U,V frame
     * @return 255 inside the box, 0 outside.
     */
    public Mat processFrame(Mat frame) {
        Core.inRange(frame, LO, HI, colourMask);
        return colourMask;
    }

    public void release(){
        colourMask.release();
    }
}
