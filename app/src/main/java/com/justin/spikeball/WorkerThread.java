package com.justin.spikeball;

import android.os.Handler;
import android.os.HandlerThread;

import java.util.concurrent.Executor;

/**
 * One named background thread that runs posted jobs one at a time, in order.
 *
 * <p>The app has three, plus the UI thread:
 * <ul>
 *   <li><b>Analysis</b> (created by the activity): camera frames and video
 *       decoding, each through the detection pipeline. The camera and a video
 *       never run at the same time.</li>
 *   <li><b>Recorder</b> (owned by {@code ReplayBuffer}): the replay encoder's
 *       callbacks and the ring of recorded footage.</li>
 *   <li><b>Saver</b> (owned by {@code RallySaver}): the last rally's footage
 *       and writing it to an MP4.</li>
 * </ul>
 *
 */
public final class WorkerThread {

    private final HandlerThread thread;
    private final Handler handler;
    private final Executor executor;

    public WorkerThread(String name) {
        thread = new HandlerThread(name);
        thread.start();
        handler = new Handler(thread.getLooper());
        executor = handler::post;
    }

    /** For APIs that take a Handler (MediaCodec callbacks). */
    public Handler handler() {
        return handler;
    }

    /** For APIs that take an Executor (CameraX). */
    public Executor executor() {
        return executor;
    }

    public void post(Runnable job) {
        handler.post(job);
    }

    /** Runs the jobs already posted, then stops the thread. Later posts are dropped. */
    public void quit() {
        thread.quitSafely();
    }
}
