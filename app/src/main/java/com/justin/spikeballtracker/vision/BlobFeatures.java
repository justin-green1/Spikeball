package com.justin.spikeballtracker.vision;

/**
 * The seven features {@link BlobSvm} uses, computed for one connected component.
 *
 * <p>Two pixel sets are used:
 * <ul>
 *   <li><b>silhouette</b>: the component with its holes filled;</li>
 *   <li><b>surround band</b>: pixels 3-6 px from the silhouette</li>
 * </ul>
 *
 * <p>Features, in {@link BlobSvm#FEATURES} order:
 * <pre>
 *   aspect      (long + 1) / (short + 1) of the minimum-area rectangle around
 *               the silhouette's pixel centres
 *   fill        component area / ((long + 1) * (short + 1))
 *   Y_mean      mean luma of the silhouette pixels inside the colour box
 *   Y_std       population std of that luma
 *   U_std       population std of U over the same pixels
 *   hue_deg     atan2(mean V - 128, mean U - 128) in degrees, 0-360
 *   Y_contrast  Y_mean minus the band's mean luma (NaN if the band is empty)
 * </pre>
 *
 * <p>Holds reusable scratch arrays, so one instance belongs to one thread.
 */
public final class BlobFeatures {

    /** Band pixels are more than BAND_IN and at most BAND_OUT px from the silhouette. */
    private static final int BAND_IN = 2, BAND_OUT = 6;

    /** Rectangle orientations tried: 0..89 degrees (a rectangle repeats every 90). */
    private static final int ANGLES = 90;
    private static final double[] COS = new double[ANGLES], SIN = new double[ANGLES];
    static {
        for (int k = 0; k < ANGLES; k++) {
            COS[k] = Math.cos(Math.toRadians(k));
            SIN[k] = Math.sin(Math.toRadians(k));
        }
    }

    private final int yLo, yHi, uLo, uHi, vLo, vHi;

    private byte[] sil = new byte[0];
    private int[] stack = new int[0];
    private int[] sat = new int[0];
    private int[] edgeX = new int[0], edgeY = new int[0];

    /**
     * @param boxLo lower Y,U,V bound of the colour box the blobs were cut with
     * @param boxHi upper bound, inclusive
     */
    public BlobFeatures(int[] boxLo, int[] boxHi) {
        yLo = boxLo[0]; uLo = boxLo[1]; vLo = boxLo[2];
        yHi = boxHi[0]; uHi = boxHi[1]; vHi = boxHi[2];
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
        final int x0 = left - BAND_OUT, y0 = top - BAND_OUT;     // grid origin in the frame
        final int gw = width + 2 * BAND_OUT, gh = height + 2 * BAND_OUT, gn = gw * gh;
        if (sil.length < gn) {
            sil = new byte[gn];
            stack = new int[gn];
            sat = new int[(gw + 1) * (gh + 1)];
            edgeX = new int[gn];
            edgeY = new int[gn];
        } else if (sat.length < (gw + 1) * (gh + 1)) {
            sat = new int[(gw + 1) * (gh + 1)];
        }

        // ---- 1. silhouette: the component, plus every hole the outside can't reach
        java.util.Arrays.fill(sil, 0, gn, (byte) 0);
        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                if (labels[(top + r) * frameW + left + c] == label) {
                    sil[(r + BAND_OUT) * gw + c + BAND_OUT] = 1;
                }
            }
        }
        fillHoles(gw, gh);

        // ---- 2. colour statistics, and the ends of each silhouette run for the rectangle
        long n = 0, sumY = 0, sumYY = 0, sumU = 0, sumUU = 0, sumV = 0;
        int edges = 0;
        for (int r = BAND_OUT; r < BAND_OUT + height; r++) {
            for (int c = BAND_OUT; c < BAND_OUT + width; c++) {
                int i = r * gw + c;
                if (sil[i] == 0) continue;
                if (sil[i - 1] == 0 || sil[i + 1] == 0) {     // row ends: enough for the hull
                    edgeX[edges] = c;
                    edgeY[edges] = r;
                    edges++;
                }
                int p = 3 * ((y0 + r) * frameW + x0 + c);
                int y = yuv[p] & 0xFF, u = yuv[p + 1] & 0xFF, v = yuv[p + 2] & 0xFF;
                if (y < yLo || y > yHi || u < uLo || u > uHi || v < vLo || v > vHi) continue;
                n++;
                sumY += y;
                sumYY += y * y;
                sumU += u;
                sumUU += u * u;
                sumV += v;
            }
        }
        if (n < 3) {
            return false;
        }
        double meanY = (double) sumY / n, meanU = (double) sumU / n, meanV = (double) sumV / n;
        double varY = Math.max(0, (double) sumYY / n - meanY * meanY);
        double varU = Math.max(0, (double) sumUU / n - meanU * meanU);

        // ---- 3. shape: minimum-area rectangle around those points, 1 degree steps
        double bestArea = Double.MAX_VALUE, longSide = 0, shortSide = 0;
        for (int k = 0; k < ANGLES; k++) {
            double minA = Double.MAX_VALUE, maxA = -Double.MAX_VALUE;
            double minB = Double.MAX_VALUE, maxB = -Double.MAX_VALUE;
            for (int e = 0; e < edges; e++) {
                double a = edgeX[e] * COS[k] + edgeY[e] * SIN[k];
                double b = edgeY[e] * COS[k] - edgeX[e] * SIN[k];
                minA = Math.min(minA, a); maxA = Math.max(maxA, a);
                minB = Math.min(minB, b); maxB = Math.max(maxB, b);
            }
            double sa = maxA - minA, sb = maxB - minB;
            if (sa * sb < bestArea) {
                bestArea = sa * sb;
                longSide = Math.max(sa, sb);
                shortSide = Math.min(sa, sb);
            }
        }
        double a = longSide + 1, b = shortSide + 1;

        // ---- 4. band: a pixel is within d of the silhouette iff the (2d+1)^2 square
        //         around it holds a silhouette pixel, which the summed-area table answers.
        final int sw = gw + 1;
        java.util.Arrays.fill(sat, 0, sw, 0);
        for (int r = 0; r < gh; r++) {
            sat[(r + 1) * sw] = 0;
            int rowSum = 0;
            for (int c = 0; c < gw; c++) {
                rowSum += sil[r * gw + c];
                sat[(r + 1) * sw + c + 1] = sat[r * sw + c + 1] + rowSum;
            }
        }
        long nb = 0, sumBandY = 0;
        for (int r = 0; r < gh; r++) {
            int fy = y0 + r;
            if (fy < 0 || fy >= frameH) continue;
            for (int c = 0; c < gw; c++) {
                int fx = x0 + c;
                if (fx < 0 || fx >= frameW) continue;
                if (silInSquare(r, c, BAND_OUT, gw, gh) && !silInSquare(r, c, BAND_IN, gw, gh)) {
                    nb++;
                    sumBandY += yuv[3 * (fy * frameW + fx)] & 0xFF;
                }
            }
        }

        double hue = Math.toDegrees(Math.atan2(meanV - 128, meanU - 128));
        if (hue < 0) hue += 360;

        out[0] = a / b;
        out[1] = area / (a * b);
        out[2] = meanY;
        out[3] = Math.sqrt(varY);
        out[4] = Math.sqrt(varU);
        out[5] = hue;
        out[6] = nb > 0 ? meanY - (double) sumBandY / nb : Double.NaN;
        return true;
    }

    /** Whether the square of half-width d around grid cell (r, c) holds a silhouette pixel. */
    private boolean silInSquare(int r, int c, int d, int gw, int gh) {
        int r0 = Math.max(0, r - d), r1 = Math.min(gh, r + d + 1);
        int c0 = Math.max(0, c - d), c1 = Math.min(gw, c + d + 1);
        int sw = gw + 1;
        return sat[r1 * sw + c1] - sat[r0 * sw + c1] - sat[r1 * sw + c0] + sat[r0 * sw + c0] > 0;
    }

    /**
     * Sets every background cell the grid border can't reach (4-connected) to 1.
     * The grid border is BAND_OUT px from the component, so it is all background.
     */
    private void fillHoles(int w, int h) {
        final byte OUTSIDE = 2;
        int sp = 0;
        sil[0] = OUTSIDE;
        stack[sp++] = 0;
        while (sp > 0) {
            int i = stack[--sp];
            int r = i / w, c = i - r * w;
            if (c > 0 && sil[i - 1] == 0) { sil[i - 1] = OUTSIDE; stack[sp++] = i - 1; }
            if (c < w - 1 && sil[i + 1] == 0) { sil[i + 1] = OUTSIDE; stack[sp++] = i + 1; }
            if (r > 0 && sil[i - w] == 0) { sil[i - w] = OUTSIDE; stack[sp++] = i - w; }
            if (r < h - 1 && sil[i + w] == 0) { sil[i + w] = OUTSIDE; stack[sp++] = i + w; }
        }
        for (int i = 0; i < w * h; i++) {
            sil[i] = (byte) (sil[i] == OUTSIDE ? 0 : 1);
        }
    }
}
