package com.justin.spikeball.tracking;


/**
 * Remembers where blobs keep appearing, so static clutter can be ignored.
 *
 * <p>The frame is cut into {@value #CELL} px cells. Every frame each cell
 * decays by {@value #DECAY} and gains 1 per blob centred in it, so a cell
 * settles at (blobs per frame) / (1 - DECAY), a memory of about 50 frames. A
 * cell is static once it holds {@value #THRESH} of a blob per frame.
 *
 * <p>Position: the grid runs in front of the blob SVM, in
 * {@code SvmBlobDetector}. It learns from every blob in its area range (30-400 px).
 */
public class OccupancyGrid {

    private static final int    FRAME_W         = 854;
    private static final int    FRAME_H         = 480;
    private static final int    CELL            = 16;
    private static final int    ROWS            = FRAME_H / CELL + 1;
    private static final int    COLS            = FRAME_W / CELL + 1;

    private static final float  DECAY           = 0.98f;
    private static final float  THRESH          = 0.65f;
    private static final int    WARMUP_FRAMES   = 60;
    //what a cell settles to;
    private static final float  FULL            = 1f / (1f - DECAY);
    private static final float  STATIC          = THRESH * FULL;

    private final float[][] grid = new float[ROWS][COLS];
    private int frames = 0;

    /** Starts a frame: every cell decays. Call once per frame, before {@link #hit}. */
    public void nextFrame(){
        for (float[] row : grid){
            for (int c = 0; c < COLS; c++){
                row[c] *= DECAY;
            }
        }
        frames ++;
    }

    /** Counts one blob centred at (x, y) in the current frame. */
    public void hit(double x, double y){
        grid[row(y)][col(x)] += 1f;
    }

    /** True when (x, y)'s cell is static. Read before this frame's update. */
    public boolean isStatic(double x, double y){
        if (frames < WARMUP_FRAMES) return false;
        return grid[row(y)][col(x)] >= STATIC;
    }

    private static int row (double y) { return clamp((int) (y / CELL) , ROWS); }
    private static int col (double x) { return clamp((int) (x / CELL) , COLS); }

    private static int clamp(int i, int n){
        return i < 0 ? 0 : (i >= n ? n - 1: i);
    }
}
