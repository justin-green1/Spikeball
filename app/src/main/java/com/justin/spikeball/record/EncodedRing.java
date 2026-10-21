package com.justin.spikeball.record;

import java.nio.ByteBuffer;

/**
 * The last few tens of seconds of encoded H.264 access units, held in RAM.
 *
 */
public final class EncodedRing {

    private final byte[] data;
    private final long retainUs;

    private final int maxPackets;
    private final int[] offset;
    private final int[] size;
    private final long[] ptsUs;
    private final boolean[] keyFrame;

    private long headSeq = 0;   // oldest packet still held
    private long tailSeq = 0;   // next sequence to be written
    private int headPos = 0;    // byte offset of the oldest packet
    private int usedBytes = 0;

    /**
     * @param capacityBytes size of the byte store; the hard ceiling on memory
     * @param maxPackets    metadata slots; must cover retainUs at the highest frame rate
     * @param retainUs      packets older than this, relative to the newest, are dropped
     */
    public EncodedRing(int capacityBytes, int maxPackets, long retainUs) {
        this.data = new byte[capacityBytes];
        this.maxPackets = maxPackets;
        this.retainUs = retainUs;
        this.offset = new int[maxPackets];
        this.size = new int[maxPackets];
        this.ptsUs = new long[maxPackets];
        this.keyFrame = new boolean[maxPackets];
    }

    /** Drops everything. Sequence numbers keep counting, so stale readers still fail safely. */
    public void clear() {
        headSeq = tailSeq;
        headPos = 0;
        usedBytes = 0;
    }

    /**
     * Appends one access unit, evicting from the old end as needed.
     *
     * @param src bytes from {@code src.position()} to {@code src.limit()}; the
     *            buffer's position is left unchanged
     */
    public void add(ByteBuffer src, long pts, boolean isKeyFrame) {
        int n = src.remaining();
        if (n > data.length) {
            return;
        }

        // Evict on time first, then on space and slots.
        while (count() > 0 && pts - ptsUs[slot(headSeq)] > retainUs) {
            evictOldest();
        }
        while (count() > 0 && (usedBytes + n > data.length || count() >= maxPackets)) {
            evictOldest();
        }

        int writePos = (headPos + usedBytes) % data.length;
        ByteBuffer view = src.duplicate();
        int first = Math.min(n, data.length - writePos);
        view.get(data, writePos, first);
        if (first < n) {
            view.get(data, 0, n - first);   // wrapped past the end of the array
        }

        int s = slot(tailSeq);
        offset[s] = writePos;
        size[s] = n;
        ptsUs[s] = pts;
        keyFrame[s] = isKeyFrame;
        if (count() == 0) {
            headPos = writePos;
        }
        usedBytes += n;
        tailSeq++;
    }

    /** What a save should mux: packets [startSeq, endSeq). */
    public static final class Plan {
        public final long startSeq;
        public final long endSeq;
        public final long firstPtsUs;
        public final long lastPtsUs;

        Plan(long startSeq, long endSeq, long firstPtsUs, long lastPtsUs) {
            this.startSeq = startSeq;
            this.endSeq = endSeq;
            this.firstPtsUs = firstPtsUs;
            this.lastPtsUs = lastPtsUs;
        }
    }

    /**
     * Chooses the packets covering {@code [fromPts, toPts]}, starting on a
     * keyframe so the clip decodes from its first frame.
     *
     * @return null when no keyframe falls before {@code toPts}
     */
    public Plan planRange(long fromPts, long toPts) {
        long start = -1;
        long end = -1;   // exclusive
        for (long seq = headSeq; seq < tailSeq; seq++) {
            int s = slot(seq);
            if (ptsUs[s] > toPts) {
                break;
            }
            if (keyFrame[s] && (ptsUs[s] <= fromPts || start < 0)) {
                start = seq;   // latest keyframe not after fromPts, or the first one
            }
            end = seq + 1;
        }
        if (start < 0) {
            return null;
        }
        return new Plan(start, end, ptsUs[slot(start)], ptsUs[slot(end - 1)]);
    }

    /** Some packets copied out of the ring, unaffected by anything written after. */
    public static final class Snapshot {
        public final byte[] data;
        public final int[] offset;
        public final int[] size;
        public final long[] ptsUs;
        public final boolean[] keyFrame;

        Snapshot(byte[] data, int[] offset, int[] size, long[] ptsUs, boolean[] keyFrame) {
            this.data = data;
            this.offset = offset;
            this.size = size;
            this.ptsUs = ptsUs;
            this.keyFrame = keyFrame;
        }

        public int count() {
            return size.length;
        }

        public long durationUs() {
            return ptsUs[count() - 1] - ptsUs[0];
        }

        public int maxPacketSize() {
            int max = 0;
            for (int n : size) {
                max = Math.max(max, n);
            }
            return max;
        }
    }

    /**
     * Copies the packets of {@code plan} into one new array, back to back.
     *
     * @return null if any of them has been evicted since the plan was made
     */
    public Snapshot snapshot(Plan plan) {
        if (plan.startSeq < headSeq || plan.endSeq > tailSeq || plan.endSeq <= plan.startSeq) {
            return null;
        }
        int n = (int) (plan.endSeq - plan.startSeq);
        int total = 0;
        for (long seq = plan.startSeq; seq < plan.endSeq; seq++) {
            total += size[slot(seq)];
        }

        byte[] out = new byte[total];
        int[] outOffset = new int[n];
        int[] outSize = new int[n];
        long[] outPts = new long[n];
        boolean[] outKey = new boolean[n];

        int pos = 0;
        for (int i = 0; i < n; i++) {
            int s = slot(plan.startSeq + i);
            int len = size[s];
            int first = Math.min(len, data.length - offset[s]);
            System.arraycopy(data, offset[s], out, pos, first);
            if (first < len) {
                System.arraycopy(data, 0, out, pos + first, len - first);   // wrapped
            }
            outOffset[i] = pos;
            outSize[i] = len;
            outPts[i] = ptsUs[s];
            outKey[i] = keyFrame[s];
            pos += len;
        }
        return new Snapshot(out, outOffset, outSize, outPts, outKey);
    }

    /** Timestamp of the newest packet held; only meaningful when {@link #count()} > 0. */
    public long newestPtsUs() {
        return ptsUs[slot(tailSeq - 1)];
    }

    public int count() {
        return (int) (tailSeq - headSeq);
    }

    private void evictOldest() {
        int s = slot(headSeq);
        usedBytes -= size[s];
        headPos = (offset[s] + size[s]) % data.length;
        headSeq++;
        if (count() == 0) {
            headPos = 0;
            usedBytes = 0;
        }
    }

    private int slot(long seq) {
        return (int) (seq % maxPackets);
    }
}
