#!/usr/bin/env python3
import sys
import gi
import time

import pyds

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# --- Configuration ---
# Multiple input V4L2 devices for multi-stream testing
INPUT_DEVICES = [
    "/dev/video0",
    "/dev/video1",
    "/dev/video2",
]
NUM_SOURCES = len(INPUT_DEVICES)
UDP_IP = "127.0.0.1"
UDP_PORT = 5400
PGIE_CONFIG = "dstest1_pgie_config.txt"  # Primary GIE config file path


def bus_call(bus, message, loop):
    """Callback function for the GStreamer message bus."""
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write(f"Error: {err}: {debug}\n")
        loop.quit()
    return True


def create_source_bin(index, device_path):
    """Creates a Gst.Bin for V4L2 device source."""
    print(f"Creating source bin for device {device_path}")

    # Create the Gst Bin
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)

    # Elements for V4L2 Source
    # 1. V4L2 Source element
    source = Gst.ElementFactory.make("v4l2src", f"v4l2-source-{index}")
    # 2. Capsfilter to set resolution and framerate
    capsfilter = Gst.ElementFactory.make("capsfilter", f"caps-filter-{index}")
    # 3. Nvvideoconvert to convert raw V4L2 format (e.g., YUYV) to NVMM
    nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", f"nvvidconv-src-{index}")

    if not all([source, capsfilter, nvvidconv_src]):
        sys.stderr.write("One or more elements in source bin failed to create\n")
        return None

    # Set device path
    source.set_property("device", device_path)

    # Set desired camera properties (adjust as needed for your camera)
    caps = Gst.Caps.from_string(
        "video/x-raw, format=YUY2, width=1280, height=720, framerate=30/1"
    )
    capsfilter.set_property("caps", caps)

    # Link elements within the bin
    nbin.add(source)
    nbin.add(capsfilter)
    nbin.add(nvvidconv_src)

    source.link(capsfilter)
    capsfilter.link(nvvidconv_src)

    # Add a Ghost Pad to expose the nvvidconv's source pad (output)
    pad = nvvidconv_src.get_static_pad("src")
    ghost_pad = Gst.GhostPad.new("src", pad)
    nbin.add_pad(ghost_pad)

    return nbin


def main():
    Gst.init(None)

    print(f"🚀 Launching DeepStream Producer...")
    print(f"📡 Target UDP: {UDP_IP}:{UDP_PORT}")
    print(f"🔢 Number of input streams: {NUM_SOURCES}")

    # 1. Create Pipeline
    pipeline = Gst.Pipeline.new("multi-stream-pipeline")

    # 2. Create Core Elements
    # DeepStream Muxer: must be before all DeepStream elements
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreen-display")

    # TEE Splitter: splits the batched OSD output
    tee = Gst.ElementFactory.make("tee", "osd-tee")

    # Stream Demuxer: unbatches the stream if multiple outputs are needed
    demuxer = Gst.ElementFactory.make("nvstreamdemux", "stream-demuxer")

    if not all([streammux, pgie, nvvidconv, nvosd, tee, demuxer]):
        sys.stderr.write("One or more core elements failed to create\n")
        return

    # 3. Configure Properties
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    # Set batch size to the number of inputs
    streammux.set_property("batch-size", NUM_SOURCES)
    pgie.set_property("config-file-path", PGIE_CONFIG)

    # 4. Add Elements to Pipeline
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(tee)
    pipeline.add(demuxer)

    # 5. Connect Input Sources to Muxer
    # Use INPUT_DEVICES list
    for i, device_path in enumerate(INPUT_DEVICES):
        source_bin = create_source_bin(i, device_path)
        if not source_bin:
            return
        pipeline.add(source_bin)

        # Link source bin to nvstreammux sink pad
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        srcpad = source_bin.get_static_pad("src")

        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            sys.stderr.write(f"ERROR: Source {i} link to StreamMuxer failed\n")
            return

    # 6. Link Core DeepStream Pipeline (Batched Processing)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)

    # OSD Output -> TEE Splitter
    nvosd.link(tee)

    # 7. TEE -> Stream Demuxer (Unbatching)
    # Link the TEE to the Demuxer input
    tee_demux_pad = tee.get_request_pad("src_%u")
    demuxer_sink_pad = demuxer.get_static_pad("sink")
    if tee_demux_pad.link(demuxer_sink_pad) != Gst.PadLinkReturn.OK:
        sys.stderr.write("ERROR: TEE link to Demuxer failed\n")
        return

    # ==============================
    # 🌊 Branch A: Local Null Sink (Inference Path)
    # Used for ensuring the batched pipeline runs
    # ==============================
    print("Building Branch A: Local Null Sink (Inference Path)")
    queue_local = Gst.ElementFactory.make("queue", "queue_local")
    fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
    fakesink.set_property("sync", 0)

    pipeline.add(queue_local)
    pipeline.add(fakesink)
    queue_local.link(fakesink)

    # TEE links to Local Sink
    tee_pad_local = tee.get_request_pad("src_%u")
    queue_pad_local = queue_local.get_static_pad("sink")
    if tee_pad_local.link(queue_pad_local) != Gst.PadLinkReturn.OK:
        sys.stderr.write("ERROR: TEE link to Local Queue failed\n")
        return

    # ==============================
    # 🌊 Branch B: UDP Transmission (RTSP Path - Stream 0 only)
    # We take the unbatched output of Stream 0 from the Demuxer
    # ==============================
    print("Building Branch B: UDP Transmission (RTSP for Stream 0)")
    # Request the Demuxer pad for the first stream (Stream 0)
    demuxer_src_pad = demuxer.get_request_pad("src_0")
    if not demuxer_src_pad:
        sys.stderr.write("ERROR: Demuxer source pad for stream 0 failed to obtain\n")
        return

    # Elements for encoding and transmission
    queue_udp = Gst.ElementFactory.make("queue", "queue_udp")
    # Must convert back to format needed by encoder
    nvvidconv_udp = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv_udp")
    encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
    rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
    udpsink = Gst.ElementFactory.make("udpsink", "udpsink")

    # Encoder Configuration (Hardware NVENC)
    encoder.set_property("bitrate", 4000000)
    encoder.set_property("iframeinterval", 30)

    # UDPSink Configuration (Push to local port 5400)
    udpsink.set_property("host", UDP_IP)
    udpsink.set_property("port", UDP_PORT)
    udpsink.set_property("async", False)
    udpsink.set_property("sync", 1)

    pipeline.add(queue_udp)
    pipeline.add(nvvidconv_udp)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(udpsink)

    # Link UDP Branch elements
    # Demuxer Pad 0 -> Queue -> Convert -> Encoder -> RTP Payloader -> UDPSink
    if demuxer_src_pad.link(queue_udp.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
        sys.stderr.write("ERROR: Demuxer link to UDP Queue failed\n")
        return

    queue_udp.link(nvvidconv_udp)
    nvvidconv_udp.link(encoder)
    encoder.link(rtppay)
    rtppay.link(udpsink)

    # ==============================
    # Run Loop
    # ==============================
    loop = GLib.MainLoop.new(None, False)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print("✅ Pipeline ready. Starting execution...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        pass

    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped.")


if __name__ == "__main__":
    main()
