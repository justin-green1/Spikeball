package com.justin.spikeball.tracking;

/** One moving blob the tracker may follow: its centroid, in frame pixels. */
public final class BallCandidate {
    public final double x;
    public final double y;
    public final double pBall;

    public BallCandidate(double x, double y, double pBall) {
        this.x = x;
        this.y = y;
        this.pBall = pBall;
    }
}
