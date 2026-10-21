package com.justin.spikeball.vision;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * The seven features {@link BlobSvm} uses, computed for one connected component
 * with OpenCV, the same calls the offline study used.
 *
 * <p>Pixel sets:
 * <ul>
 *   <li><b>silhouette</b>: the component's outer contour, filled (so holes are included);</li>
 *   <li><b>blob pixels</b>: silhouette AND the colour box;</li>
 *   <li><b>surround band</b>: silhouette dilated 13x13 minus silhouette dilated 5x5
 *       (pixels 3-6 px out), clipped to the frame.</li>
 * </ul>
 *
 * <p>Features, in {@link BlobSvm#FEATURES} order:
 * <pre>
 *   aspect      (long + 1) / (short + 1) of minAreaRect on the outer contour
 *   fill        component area / ((long + 1) * (short + 1))
 *   Y_mean      mean luma of the blob pixels
 *   Y_std       population std of that luma
 *   U_std       population std of U
 *   hue_deg     atan2(mean V - 128, mean U - 128) in degrees, 0-360
 *   Y_contrast  Y_mean minus the band's mean luma (NaN if the band is empty)
 * </pre>
 *
 * <p>Holds reusable Mats, so one instance belongs to one thread.
 */
public final class BlobFeatures_openCV {

    /** The 13x13 dilation reaches 6 px, so the working region is the box plus 6 px. */
    private static final int PAD = 6;

    private final Scalar boxLo, boxHi;
    private final Mat kernelOut = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13, 13));
    private final Mat kernelIn = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));

    // Reused between calls; OpenCV reallocates them only when the region size changes.
    private final Mat comp = new Mat(), sil = new Mat(), yuvRoi = new Mat(), yRoi = new Mat();
    private final Mat inBox = new Mat(), blobPx = new Mat(), far = new Mat(), near = new Mat(), band = new Mat();
    private final Mat hierarchy = new Mat();
    private final MatOfDouble mean = new MatOfDouble(), std = new MatOfDouble();
    private byte[] compBuf = new byte[0], yuvBuf = new byte[0];

    /**
     * @param boxLo lower Y,U,V bound of the colour box the blobs were cut with
     * @param boxHi upper bound, inclusive
     */
    public BlobFeatures_openCV(int[] boxLo, int[] boxHi) {
        this.boxLo = new Scalar(boxLo[0], boxLo[1], boxLo[2]);
        this.boxHi = new Scalar(boxHi[0], boxHi[1], boxHi[2]);
    }

    /**
     * Computes the features of one component.
     *
     * @param yuv    the whole frame, interleaved Y,U,V, row-major, {@code frameW * frameH * 3}
     * @param labels the whole label image, row-major, {@code frameW * frameH}
     * @param label  the component's label
     * @param left   bounding box, from the component's stats
     * @param area   the component's pixel count
     * @param out    written with {@link BlobSvm#N_FEATURES} values
     * @return false if the blob has fewer than 3 in-box pixels; {@code out} is then undefined
     */
    public boolean compute(byte[] yuv, int[] labels, int frameW, int frameH, int label,
                           int left, int top, int width, int height, int area,
                           double[] out) {
        // ---- working region: the bounding box plus PAD, clipped to the frame
        final int x0 = Math.max(0, left - PAD), y0 = Math.max(0, top - PAD);
        final int x1 = Math.min(frameW, left + width + PAD), y1 = Math.min(frameH, top + height + PAD);
        final int gw = x1 - x0, gh = y1 - y0, gn = gw * gh;
        if (compBuf.length < gn) {
            compBuf = new byte[gn];
            yuvBuf = new byte[3 * gn];
        }
        for (int r = 0; r < gh; r++) {
            int src = (y0 + r) * frameW + x0;
            System.arraycopy(yuv, 3 * src, yuvBuf, 3 * r * gw, 3 * gw);
            for (int c = 0; c < gw; c++) {
                compBuf[r * gw + c] = labels[src + c] == label ? (byte) 255 : 0;
            }
        }
        comp.create(gh, gw, CvType.CV_8UC1);
        comp.put(0, 0, java.util.Arrays.copyOf(compBuf, gn));
        yuvRoi.create(gh, gw, CvType.CV_8UC3);
        yuvRoi.put(0, 0, java.util.Arrays.copyOf(yuvBuf, 3 * gn));

        // ---- silhouette: the outer contour, filled
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(comp, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        if (contours.isEmpty()) {
            return false;
        }
        int best = 0;
        for (int k = 1; k < contours.size(); k++) {
            if (contours.get(k).rows() > contours.get(best).rows()) best = k;
        }
        sil.create(gh, gw, CvType.CV_8UC1);
        sil.setTo(Scalar.all(0));
        Imgproc.drawContours(sil, contours, best, Scalar.all(255), Imgproc.FILLED);

        // ---- shape: minimum-area rectangle of the outer contour
        MatOfPoint2f pts = new MatOfPoint2f(contours.get(best).toArray());
        RotatedRect rect = Imgproc.minAreaRect(pts);
        pts.release();
        for (MatOfPoint c : contours) c.release();
        double a = Math.max(rect.size.width, rect.size.height) + 1;
        double b = Math.min(rect.size.width, rect.size.height) + 1;

        // ---- colour statistics on silhouette AND colour box
        Core.inRange(yuvRoi, boxLo, boxHi, inBox);
        Core.bitwise_and(inBox, sil, blobPx);
        if (Core.countNonZero(blobPx) < 3) {
            return false;
        }
        Core.meanStdDev(yuvRoi, mean, std, blobPx);     // population std
        double[] m = mean.toArray(), s = std.toArray();

        // ---- surround band: pixels 3-6 px from the silhouette
        Imgproc.dilate(sil, far, kernelOut);
        Imgproc.dilate(sil, near, kernelIn);
        Core.subtract(far, near, band);
        Core.extractChannel(yuvRoi, yRoi, 0);
        double contrast = Double.NaN;
        if (Core.countNonZero(band) > 0) {
            contrast = m[0] - Core.mean(yRoi, band).val[0];
        }

        double hue = Math.toDegrees(Math.atan2(m[2] - 128, m[1] - 128));
        if (hue < 0) hue += 360;

        out[0] = a / b;
        out[1] = area / (a * b);
        out[2] = m[0];
        out[3] = s[0];
        out[4] = s[1];
        out[5] = hue;
        out[6] = contrast;
        return true;
    }
}