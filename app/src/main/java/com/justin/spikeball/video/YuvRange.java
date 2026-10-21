package com.justin.spikeball.video;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Expands limited-range YUV (Y 16-235, U/V 16-240) to full range (0-255), the
 * space box B, the SVM and the KNN threshold were tuned in. Shared by both
 * input paths.
 *
 * <p>Holds a native lookup table, so one instance per source.
 */
public final class YuvRange {

    /** Channel-wise 16-235 / 16-240 to 0-255 map, built once. */
    private final Mat lut = buildFullRangeLut();

    /** Expands limited range to full range, in place. */
    public void expandToFull(Mat yuv) {
        Core.LUT(yuv, lut, yuv);
    }

    public void release() {
        lut.release();
    }

    /**
     * Builds the lookup table that maps limited-range samples to full range.
     *
     * <p>Luma and chroma use different conventions: luma occupies 16-235 and
     * expands about its floor, chroma occupies 16-240 and expands about its
     * neutral point of 128. A single 3-channel LUT applies both in one pass,
     * which is far cheaper per frame than splitting and merging channels.
     *
     * @return a 1x256 CV_8UC3 table, channels in Y, U, V order
     */
    private static Mat buildFullRangeLut() {
        Mat lut = new Mat(1, 256, CvType.CV_8UC3);
        byte[] table = new byte[256 * 3];

        for (int i = 0; i < 256; i++) {
            int y = clampByte((int) Math.round((i - 16) * 255.0 / 219.0));
            int c = clampByte((int) Math.round((i - 128) * 255.0 / 224.0 + 128));
            table[i * 3] = (byte) y;
            table[i * 3 + 1] = (byte) c;
            table[i * 3 + 2] = (byte) c;
        }

        lut.put(0, 0, table);
        return lut;
    }

    private static int clampByte(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }
}
