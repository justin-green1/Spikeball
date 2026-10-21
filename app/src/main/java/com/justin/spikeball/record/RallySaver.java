package com.justin.spikeball.record;

import android.content.Context;
import android.media.MediaMuxer;

import com.justin.spikeball.WorkerThread;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/**
 * Holds the last rally's footage, from either source, and saves it to
 * {@code DCIM/Spikeball Videos/} on request.
 *
 * <p>The footage is replaced only when a newer rally is kept; saving does not
 * clear it, so it can be saved more than once. It lives in RAM, so it is gone
 * when the app closes.
 */
public final class RallySaver {

    public interface SaveCallback {
        /** Called on the Saver thread. */
        void onSaved(String displayName, double seconds);

        /** Called on the Saver thread. */
        void onFailed(String reason);
    }

    private final Context appContext;
    private final WorkerThread saver = new WorkerThread("Saver");

    private RallyFootage last = null;   // Saver thread only

    public RallySaver(Context context) {
        this.appContext = context.getApplicationContext();
    }

    /** Makes {@code footage} the last rally. A null (footage not captured) keeps the previous one. */
    public void keep(RallyFootage footage) {
        if (footage != null) {
            saver.post(() -> last = footage);
        }
    }

    /** Saves the last rally as {@code rally_<date-time>.mp4}. Returns at once. */
    public void saveLast(SaveCallback callback) {
        saver.post(() -> {
            RallyFootage footage = last;
            if (footage == null) {
                callback.onFailed("no rally yet");
                return;
            }
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
            String name = "rally_" + stamp + ".mp4";
            try {
                long durationUs = write(footage, name);
                callback.onSaved(name, durationUs / 1_000_000.0);
            } catch (Exception e) {
                callback.onFailed(e.getMessage() != null ? e.getMessage() : e.toString());
            }
        });
    }

    /** Lets a save already asked for finish, then stops the Saver thread. */
    public void release() {
        saver.quit();
    }

    private long write(RallyFootage footage, String name) throws Exception {
        Mp4File out = Mp4File.open(appContext, name);
        MediaMuxer muxer = null;
        boolean ok = false;
        try {
            muxer = out.newMuxer();
            long durationUs = footage.writeTo(muxer);
            ok = true;
            return durationUs;
        } finally {
            if (muxer != null) {
                try {
                    muxer.release();
                } catch (IllegalStateException ignored) {
                    // The write already failed; the partial file is deleted below.
                }
            }
            out.finish(ok);
        }
    }
}
