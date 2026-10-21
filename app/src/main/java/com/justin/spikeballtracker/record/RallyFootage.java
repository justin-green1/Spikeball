package com.justin.spikeballtracker.record;

import android.media.MediaMuxer;

import java.io.IOException;

/**
 * One confirmed rally's video, held until it is saved or replaced by the
 * next rally.
 */
public interface RallyFootage {

    /** Footage kept before the detected rally start. */
    long BEFORE_US = 1_000_000L;

    /** Footage kept after the detected rally end. */
    long AFTER_US = 3_000_000L;

    /**
     * Adds a video track to {@code muxer}, starts it, writes every sample and
     * stops it.
     *
     * @return the length written, first to last frame, in microseconds
     */
    long writeTo(MediaMuxer muxer) throws IOException;
}
