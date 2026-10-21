package com.justin.spikeballtracker.record;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import androidx.annotation.NonNull;
import androidx.camera.core.Preview;
import androidx.camera.core.SurfaceRequest;

import com.justin.spikeballtracker.WorkerThread;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Keeps the last minute of camera footage in RAM, and copies a confirmed
 * rally's footage out of it before it is overwritten.
 */
public final class ReplayBuffer {

    private static final String TAG = "ReplayBuffer";

    private static final long RETAIN_US = 60 * 1_000_000L;

    private static final long CLOCK_MISMATCH_US = 1_000_000L;

    private static final String MIME = MediaFormat.MIMETYPE_VIDEO_AVC;
    private static final int BIT_RATE = 10_000_000;
    private static final int FRAME_RATE = 30;
    private static final int I_FRAME_INTERVAL_S = 1;
    private static final int MAX_FPS_FOR_SLOTS = 60;
    private static final double BYTE_HEADROOM = 1.25;

    private final WorkerThread recorder = new WorkerThread("Recorder");

    private EncodedRing ring = null;

    private MediaFormat outputFormat = null;
    private int rotationDegrees = 0;

    private Session current = null;

    private final List<Session> open = new ArrayList<>();

    public void attachTo(Preview preview) {
        preview.setSurfaceProvider(recorder.executor(), this::onSurfaceRequested);
    }

    /**
     * Copies the footage from {@code fromUs} to {@code toUs} out of the ring,
     * on the Recorder thread, and hands it to {@code done} there. The clip
     * starts on the keyframe at or before {@code fromUs}
     *
     * @param fromUs start, on the camera's sensor clock
     * @param toUs   end, same clock
     * @param nowUs  the newest analysis frame's time, same clock;
     * @param done   receives the copy, or null when the footage is not held
     */
    public void copyRally(long fromUs, long toUs, long nowUs, Consumer<RallyFootage> done) {
        recorder.post(() -> done.accept(copy(fromUs, toUs, nowUs)));
    }

    /** {@link #copyRally}'s work. Recorder thread only. A few ms per 10 s of footage. */
    private BufferedRally copy(long fromUs, long toUs, long nowUs) {
        EncodedRing r = ring;
        if (r == null || outputFormat == null || r.count() == 0) {
            return null;   // nothing recorded yet
        }

        long skewUs = r.newestPtsUs() - nowUs;
        if (Math.abs(skewUs) > CLOCK_MISMATCH_US) {
            fromUs += skewUs;
            toUs += skewUs;
        }

        EncodedRing.Plan plan = r.planRange(fromUs, toUs);
        EncodedRing.Snapshot packets = plan == null ? null : r.snapshot(plan);
        if (packets == null) {
            return null;
        }
        return new BufferedRally(packets, outputFormat, rotationDegrees);
    }

    /** Stops any encoder, then the Recorder thread, after the copies already asked for. */
    public void release() {
        recorder.post(() -> {
            for (Session s : new ArrayList<>(open)) {
                s.end();
            }
            current = null;
        });
        recorder.quit();
    }

    // --- encoder side ---------------------------------------------------------

    private void onSurfaceRequested(@NonNull SurfaceRequest request) {
        EncodedRing r = ensureRing();
        if (r == null) {
            request.willNotProvideSurface();
            return;
        }

        Size res = request.getResolution();
        MediaFormat format = MediaFormat.createVideoFormat(MIME, res.getWidth(), res.getHeight());
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface);
        format.setInteger(MediaFormat.KEY_BIT_RATE, BIT_RATE);
        format.setInteger(MediaFormat.KEY_FRAME_RATE, FRAME_RATE);
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, I_FRAME_INTERVAL_S);

        Session session;
        try {
            MediaCodec codec = MediaCodec.createEncoderByType(MIME);
            session = new Session(codec);
            codec.setCallback(session, recorder.handler());    // must precede configure
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            session.surface = codec.createInputSurface();  // after configure, before start
            codec.start();
        } catch (Exception e) {
            Log.e(TAG, "encoder setup failed for " + res + "; replay disabled this session", e);
            request.willNotProvideSurface();
            return;
        }

        // A new encoder means new stream headers, so the old packets cannot
        // follow them into one file.
        r.clear();
        outputFormat = null;
        rotationDegrees = 0;
        current = session;
        open.add(session);

        request.setTransformationInfoListener(recorder.executor(), info -> {
            if (current == session) {
                rotationDegrees = info.getRotationDegrees();
            }
        });

        // The result arrives once the camera has stopped writing to the surface,
        // which is the only safe moment to tear the encoder down.
        request.provideSurface(session.surface, recorder.executor(), result -> {
            session.end();
            if (current == session) {
                current = null;
            }
        });
    }

    private EncodedRing ensureRing() {
        if (ring != null) {
            return ring;
        }
        double retainS = RETAIN_US / 1_000_000.0;
        int capacity = (int) (BIT_RATE / 8.0 * retainS * BYTE_HEADROOM);
        int slots = (int) Math.ceil(retainS) * MAX_FPS_FOR_SLOTS;
        // At 1080p the full ring is ~100 MB of Java heap. If the heap cannot
        // spare that, a half-size ring still works: the byte limit evicts
        // first, so it simply holds about half as many seconds.
        for (int attempt = capacity; attempt >= capacity / 2; attempt /= 2) {
            try {
                ring = new EncodedRing(attempt, slots, RETAIN_US);
                return ring;
            } catch (OutOfMemoryError e) {
                // Try the smaller size.
            }
        }
        Log.e(TAG, "not enough heap for a replay ring; replay disabled");
        return ring;
    }

    /** One encoder, living exactly as long as one preview surface. */
    private final class Session extends MediaCodec.Callback {
        final MediaCodec codec;
        Surface surface;
        private boolean ended = false;

        Session(MediaCodec codec) {
            this.codec = codec;
        }

        @Override
        public void onInputBufferAvailable(@NonNull MediaCodec codec, int index) {
            // Surface input: the camera fills the encoder, never us.
        }

        @Override
        public void onOutputBufferAvailable(@NonNull MediaCodec codec, int index,
                                            @NonNull MediaCodec.BufferInfo info) {
            if (ended) {
                return;   // a callback already queued when the codec was stopped
            }
            try {
                boolean config = (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0;
                if (current == this && info.size > 0 && !config) {
                    ByteBuffer buf = codec.getOutputBuffer(index);
                    if (buf != null) {
                        buf.position(info.offset);
                        buf.limit(info.offset + info.size);
                        boolean key = (info.flags & MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0;
                        ring.add(buf, info.presentationTimeUs, key);   // false: larger than the ring, dropped
                    }
                }
                codec.releaseOutputBuffer(index, false);
            } catch (IllegalStateException ignored) {
                // The codec was stopped while this callback was queued.
            }
        }

        @Override
        public void onOutputFormatChanged(@NonNull MediaCodec codec, @NonNull MediaFormat format) {
            // Arrives before the first packet and carries the SPS/PPS the muxer needs.
            if (current == this) {
                outputFormat = format;
            }
        }

        @Override
        public void onError(@NonNull MediaCodec codec, @NonNull MediaCodec.CodecException e) {
            Log.e(TAG, "encoder error; ending this session", e);
            end();
            if (current == this) {
                current = null;
            }
        }

        void end() {
            if (ended) {
                return;
            }
            ended = true;
            open.remove(this);
            try {
                codec.stop();
            } catch (IllegalStateException ignored) {
                // Already stopped after an error; release() still frees it.
            }
            codec.release();
            if (surface != null) {
                surface.release();
            }
        }
    }
}
