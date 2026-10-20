package com.justin.spikeballtracker.tracking;

/**
 * The region a ball track must pass through to count as touching the net:
 * the net box exactly as {@code NetDetector} (YOLO) returns it, with no padding.
 *
 * <p>Offline test 19 Oct 2026 (studies/net_band): the bare box with a 0.2 s
 * extension linked fewer dead-time tracks than a 50 px band above the box
 * (18 % vs 24 % of dead-time ball tracks) for the same 49 of 98 labelled rallies.
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
