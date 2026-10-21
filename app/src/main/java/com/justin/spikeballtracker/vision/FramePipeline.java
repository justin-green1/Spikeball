package com.justin.spikeballtracker.vision;

import android.util.Log;

import com.justin.spikeballtracker.tracking.BallCandidate;
import com.justin.spikeballtracker.tracking.BallTracker;
import com.justin.spikeballtracker.tracking.NetZone;
import com.justin.spikeballtracker.tracking.RallyDetector;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * One frame of the detection pipeline, shared by the recorded-video path and
 * the live-camera path.
 *
 * <p>Input is always a CV_8UC3 Y,U,V Mat at the working resolution
 * (854x480, landscape).
 */
public final class FramePipeline {

    private static final String TAG = "FramePipeline";

    /** The display Mat is RGB, so this is (R, G, B). */
    private static final Scalar NET_COLOUR = new Scalar(0, 255, 255);

    // ---------------------------------------------------------------- net timing

    private static final long NET_WARMUP_US           = 10_000_000L;
    private static final long NET_ATTEMPT_INTERVAL_US = 30_000_000L;

    // ---------------------------------------------------------------- detectors

    private final MotionDetector  motionDetector  = new MotionDetector();
    private final ColourDetector  colourDetector  = new ColourDetector();
    private final SvmBlobDetector svmBlobDetector =
            new SvmBlobDetector(ColourDetector.LO, ColourDetector.HI);
    private final NetDetector     netDetector     = new NetDetector();
    private final BallTracker     tracker         = new BallTracker();
    private RallyDetector rallyDetector = new RallyDetector();

    // ---------------------------------------------------------------- reusable Mats

    private final Mat ballMask   = new Mat();
    private final Mat background = new Mat();

    // ---------------------------------------------------------------- run state

    private boolean resetRequested = false;

    private int frameIndex = 0;
    private long warmStartUs = 0L;
    private Rect cachedNetBox = null;
    private long lastNetAttemptUs = -NET_ATTEMPT_INTERVAL_US;
    private NetZone cachedNetZone = null;

    private int ralliesReported = 0;
    private final List<RallyDetector.Rally> confirmedRallies = new ArrayList<>();



    // ---------------------------------------------------------------- lifecycle

    /**
     * Runs one frame through detection, tracking and rally detection, and
     * draws the display.
     *
     * @param yuv         CV_8UC3 Y,U,V frame at the working resolution
     * @param timestampUs microseconds since this run's first frame
     * @param displayOut  written with this frame in RGB, plus the net box
     * @return rallies confirmed on this frame.
     */
    public List<RallyDetector.Rally> process(Mat yuv, long timestampUs, Mat displayOut) {
        int index = frameIndex++;
        if (index == 0) {
            warmStartUs = timestampUs;
        }
        if (resetRequested) {
            resetRequested = false;
            applyReset(timestampUs);
        }

        // Detection: the colour box ANDed with the motion mask.
        Mat colourMask = colourDetector.processFrame(yuv);
        Mat motionMask = motionDetector.processFrame(yuv);
        Core.bitwise_and(colourMask, motionMask, ballMask);
        List<BallCandidate> candidates = svmBlobDetector.findCandidates(ballMask, yuv);

        // Tracking.
        tracker.process(index, timestampUs, candidates);

        // Net.
        updateNet(timestampUs);

        // Rallies.
        rallyDetector.process(tracker.latest(), tracker.ballTrack(), tracker.dropped(), cachedNetZone);
        collectRallies();

        // Display.
        Imgproc.cvtColor(yuv, displayOut, Imgproc.COLOR_YUV2RGB);
        drawNet(displayOut);

        return drainRallies();
    }

    /**
     * Ends the run for the rally detector.
     * after the last {@link #process}, before {@link #release()}.
     *
     * @return rallies confirmed by finishing
     */
    public List<RallyDetector.Rally> finish() {
        rallyDetector.finish();
        collectRallies();
        return drainRallies();
    }

    /** Asks for a background reset, applied at the start of the next frame.*/
    public void requestReset() {
        resetRequested = true;
    }

    public void release() {
        motionDetector.release();
        colourDetector.release();
        svmBlobDetector.release();
        netDetector.release();
        ballMask.release();
        background.release();
    }

    // ---------------------------------------------------------------- net

    /**
     * Re-runs net detection on the KNN background plate every
     * {@link #NET_ATTEMPT_INTERVAL_US}, once the model has had
     * {@link #NET_WARMUP_US} to settle.
     */
    private void updateNet(long timestampUs) {
        boolean detectDue =
                timestampUs - warmStartUs >= NET_WARMUP_US
                        && timestampUs - lastNetAttemptUs >= NET_ATTEMPT_INTERVAL_US;
        if (!detectDue) {
            return;
        }

        try {
            motionDetector.getBackgroundImage(background);
            if (background.empty()) {
                return;
            }

            lastNetAttemptUs = timestampUs;

            Rect box = netDetector.detect(background);
            if (box != null) {
                cachedNetBox = box;
                cachedNetZone = NetZone.ofBox(box.x, box.y, box.width, box.height);
            }
        } catch (Exception e) {
            Log.e(TAG, "net update failed", e);
        }
    }

    /** Draws the cached net box, if there is one. */
    private void drawNet(Mat img) {
        if (cachedNetBox != null) {
            Imgproc.rectangle(img, cachedNetBox.tl(), cachedNetBox.br(), NET_COLOUR, 2);
        }
    }

    // ---------------------------------------------------------------- rallies

    /** Moves rallies the detector has confirmed since the last call into {@link #confirmedRallies}. */
    private void collectRallies() {
        List<RallyDetector.Rally> all = rallyDetector.rallies();
        while (ralliesReported < all.size()) {
            confirmedRallies.add(all.get(ralliesReported++));
        }
    }

    /** Hands out the collected rallies and clears the list. */
    private List<RallyDetector.Rally> drainRallies() {
        if (confirmedRallies.isEmpty()) {
            return Collections.emptyList();
        }
        List<RallyDetector.Rally> out = new ArrayList<>(confirmedRallies);
        confirmedRallies.clear();
        return out;
    }

    // ---------------------------------------------------------------- reset

    /** Clears  */
    private void applyReset(long timestampUs) {
        rallyDetector.finish();
        collectRallies();
        rallyDetector = new RallyDetector();
        ralliesReported = 0;

        motionDetector.reset();
        svmBlobDetector.reset();
        tracker.reset();
        cachedNetBox = null;
        cachedNetZone = null;
        warmStartUs = timestampUs;
        lastNetAttemptUs = timestampUs - NET_ATTEMPT_INTERVAL_US;
    }
}
