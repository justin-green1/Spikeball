package com.justin.spikeball.tracking;

/**
 * The region a ball track must pass through to count as touching the net:
 * the net box exactly as {@code NetDetector} (YOLO) returns it.
 *
 */
public class NetZone {

    public final double left, top, right, bottom;

    public NetZone(double left, double top, double right, double bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    /** The net box given as x, y, width and height, in px. */
    public static NetZone ofBox(double x, double y, double w, double h){
        return new NetZone(x, y, x + w, y + h);
    }

    public boolean contains(double px, double py){
        return px >= left && px <= right && py >= top && py <= bottom;
    }
}
