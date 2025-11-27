#!/usr/bin/env python3
import sys
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib

# --- Config ---
RTSP_PORT = "8554"
LISTEN_UDP_PORT = 5400  # Must be consistent with the producer's target port
MOUNT_POINT = "/live"


class MyRTSPServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(MyRTSPServer, self).__init__(**properties)

        # 1. Create Media Factory
        self.factory = GstRtspServer.RTSPMediaFactory()
        self.factory.set_shared(
            True
        )  # Allows multiple clients to share the same stream

        # 2. Define Launch String
        # This is a GStreamer pipeline description string
        # Key points:
        # - udpsrc: Reads data from the local port
        # - caps: The RTP format must be explicitly defined, otherwise udpsrc cannot understand the data
        # - name=pay0: The RTSP server looks for an element named pay0 as the data output.
        launch_str = (
            f"( udpsrc name=pay0 port={LISTEN_UDP_PORT} "
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" )'
        )

        print(f"📦 Factory Launch String: {launch_str}")
        self.factory.set_launch(launch_str)

        # 3. Mount path
        self.get_mount_points().add_factory(MOUNT_POINT, self.factory)
        self.attach(None)


def main():
    Gst.init(None)

    # Start the server
    server = MyRTSPServer()
    server.set_service(RTSP_PORT)

    print(f"\n🎥 RTSP Server is running...")
    print(f"🌍 Streaming address: rtsp://<local IP>:{RTSP_PORT}{MOUNT_POINT}")
    print(f"👂 Listening for internal data: UDP {LISTEN_UDP_PORT}")
    print(f"⏳ Waiting for deepstream_producer.py to start sending data...\n")

    # Enter the main loop
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
