package com.justin.spikeball.tracking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Decides which stretches of a run are rallies.
 *
 * <p><b>R1: net-linked tracks</b>
 *
 * <p><b>R2: score and rallies</b>
 *
 */
public class RallyDetector {
    //---- R1 constants: extending ball tracks to the net.
    /** How far each track's fit is followed past its first and last point. */
    private static final long   EXTEND_US       = 200_000L;
    /** Sampling step along the fit: under 2 px of travel at 1500 px/s. */
    private static final long   SAMPLE_US       = 1_000L;

    //---- R2 constants: Score and rallies.
    private static final long   HALF_FRAME_US   = 16_667L;
    private static final long   COVER_REACH_US  = EXTEND_US + HALF_FRAME_US;
    private static final long   WINDOW_BACK_US  = 1_516_667L;
    private static final long   WINDOW_AHEAD_US = 1_483_333L;
    private static final double SCALE           = 0.8888888888888888;
    private static final double HI              = 0.205;
    private static final double LO              = 0.155;
    private static final long   MERGE_GAP_US    = 2_050_000L;
    private static final long   MIN_LEN_US      = 450_000L;
    private static final long   SNAP_PRE_US     = 266_667L;
    private static final long   SNAP_POST_US    = 500_000L;

    //R1
    private NetZone zone = null;
    /** Ball tracks the tracker still holds, in the order they were first reported. */
    private final List<Track> pending = new ArrayList<>();

    //R2
    private final List<Frame> frames = new ArrayList<>();
    private int nextToScore = 0;
    private long coverUntilUs = Long.MIN_VALUE;

    private boolean on = false;
    private int openStart, lastOn;
    private int pendingStart = -1, pendingEnd;
    private final List<Rally> rallies = new ArrayList<>();

    private static final class Frame{
        final long tUs;
        final boolean ball;
        boolean cover;

        private Frame(long tUs, boolean ball) {
            this.tUs = tUs;
            this.ball = ball;
        }
    }

    public static class Rally{
        public final long startUs, endUs;

        public Rally(long startUs, long endUs) {
            this.startUs = startUs;
            this.endUs = endUs;
        }
    }

    /**
     * One frame.
     *
     * @param o           the tracker's delayed answer ({@link BallTracker#latest()})
     * @param ball        the track reported as the ball this frame ({@link BallTracker#ballTrack()}), or null
     * @param dropped     tracks the tracker dropped this frame ({@link BallTracker#dropped()})
     * @param currentZone the net box now, or null if there is none yet
     */
    public void process(BallTracker.Output o, Track ball, List<Track> dropped, NetZone currentZone){
        zone = currentZone;
        if (ball != null && !containsSame(pending, ball)) pending.add(ball);
        for (Track t : dropped) {
            if (removeSame(pending, t)) evaluate(t);
        }
        if(o == null) return;

        Frame fr = new Frame(o.tUs, o.hasBall);
        fr.cover = o.tUs <= coverUntilUs;
        frames.add(fr);

        scoreReady(finalUs(), false);
    }

    /** Follows the track's fit EXTEND_US past each end; marks cover if the curve passes through the net box. */
    private void evaluate(Track t){
        if (zone == null || t.size() < 4) return;
        long t0 = t.firstUs() - EXTEND_US, t1 = t.lastUs() + EXTEND_US;
        long in0 = Long.MAX_VALUE, in1 = Long.MIN_VALUE;
        for (long u = t0; u <= t1; u += SAMPLE_US) {
            if (zone.contains(t.predictX(u), t.predictY(u))) {
                if (u < in0) in0 = u;
                in1 = u;
            }
        }
        if (in1 == Long.MIN_VALUE) return;
        markCover(Math.min(t.firstUs(), in0), Math.max(t.lastUs(), in1));
    }

    private static boolean containsSame(List<Track> list, Track t){
        for (Track x : list) if (x == t) return true;
        return false;
    }

    private static boolean removeSame(List<Track> list, Track t){
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == t) { list.remove(i); return true; }
        }
        return false;
    }

    //frames from s0 to s1 are on a net-linked track, including upcoming frames.
    private void markCover(long s0, long s1){
        for(int i = frames.size() - 1; i >= 0; i--){
            Frame f = frames.get(i);
            if(f.tUs < s0 - HALF_FRAME_US) break;
            if(f.tUs <= s1 + HALF_FRAME_US) f.cover = true;
        }
        coverUntilUs = Math.max(coverUntilUs, s1 + HALF_FRAME_US);
    }

    //Frames before this cannot gain cover: the first point of a ball track still alive, less the extension.
    private long finalUs(){
        long start = frames.getLast().tUs;
        for (Track t : pending) start = Math.min(start, t.firstUs());
        return start - COVER_REACH_US;
    }

    //score every frame whose whole window is final
    private void scoreReady(long finalUs, boolean all){
        while (nextToScore < frames.size()){
            Frame f = frames.get(nextToScore);
            if (!all && f.tUs + WINDOW_AHEAD_US > finalUs) return;
            onScore(nextToScore, score(nextToScore));
            nextToScore ++;
        }
    }

    //half the ball share + half the net-cover share over the 90 frame window
    private double score(int i){
        long t = frames.get(i).tUs;
        int j = i;

        while(j >= 0 && frames.get(j).tUs >= t - WINDOW_BACK_US) j--;
        int n = 0, balls = 0, cover = 0;

        for(j = j + 1; j < frames.size() && frames.get(j).tUs < t + WINDOW_AHEAD_US; j++){
            n++;
            if (frames.get(j).ball) balls ++;
            if (frames.get(j).cover) cover ++;
        }
        return 0.5 * ((double) balls / n + (double) cover / n);
    }

    private void onScore(int i, double s){
        Frame f = frames.get(i);

        if(pendingStart >= 0 && f.tUs - frames.get(pendingEnd).tUs > MERGE_GAP_US){
            finalisePending();
        }

        if(!on && s / SCALE >= HI){
            on = true;
            if (pendingStart >= 0){
                openStart = pendingStart;
                pendingStart = -1;
            }else{
                openStart = i;
            }
        }else if(on && s / SCALE < LO){
            on = false;
            pendingStart = openStart;
            pendingEnd = lastOn;
        }
        if(on) lastOn = i;
    }

    private void finalisePending(){
        int s = pendingStart, e = pendingEnd;
        pendingStart = -1;
        if (frames.get(e).tUs - frames.get(s).tUs < MIN_LEN_US) return;

        int firstBall = -1, lastBall = -1;
        for (int k = s; k <= e; k++){
            if (frames.get(k).ball){
                if(firstBall < 0) firstBall = k;
                lastBall = k;
            }
        }
        if(firstBall >= 0){
            rallies.add(new Rally(frames.get(firstBall).tUs - SNAP_PRE_US,
                                    frames.get(lastBall).tUs + SNAP_POST_US));
        }else{
            rallies.add(new Rally(frames.get(s).tUs, frames.get(e).tUs));
        }
    }

    public void finish(){
        for (Track t : pending) evaluate(t);
        pending.clear();
        scoreReady(Long.MAX_VALUE, true);
        if(on){
            on = false;
            pendingStart = openStart;
            pendingEnd = lastOn;
        }
        if(pendingStart >= 0) finalisePending();
    }

    public List<Rally> rallies(){ return Collections.unmodifiableList(rallies);}
}
