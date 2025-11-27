#!/usr/bin/env python3
import sys
import asyncio
import json
import websockets
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst, GstWebRTC, GstSdp

# --- Config ---
UDP_PORT = 5400
WS_PORT = 8888
STUN_SERVER = "stun://stun.l.google.com:19302"  # Used for NAT traversal


class WebRTCClient:
    def __init__(self, websocket):
        self.websocket = websocket
        self.pipeline = None
        self.webrtcbin = None

    async def send_sdp_offer(self, offer):
        text = offer.sdp.as_text()
        msg = json.dumps({"type": "offer", "sdp": text})
        await self.websocket.send(msg)

    def on_offer_created(self, promise, _, __):
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer")

        # Set local description
        promise = Gst.Promise.new()
        self.webrtcbin.emit("set-local-description", offer, promise)
        promise.interrupt()

        # Send to browser
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.run_coroutine_threadsafe(self.send_sdp_offer(offer), self.loop)

    def on_negotiation_needed(self, element):
        # Create Offer
        promise = Gst.Promise.new_with_change_func(self.on_offer_created, element, None)
        element.emit("create-offer", None, promise)

    def on_ice_candidate(self, element, mlineindex, candidate):
        # Send ICE Candidate to the browser
        msg = json.dumps(
            {"type": "ice", "candidate": candidate, "sdpMLineIndex": mlineindex}
        )
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(self.websocket.send(msg), loop)

    def start_pipeline(self):
        # Core Pipeline:
        # 1. udpsrc: Receives UDP data from DeepStream
        # 2. rtph264depay: Unpacks DeepStream's RTP packets and restores them to H.264 stream
        # 3. rtph264pay: Repackages the data into WebRTC-compatible RTP (config-interval=-1 is crucial for fast image generation)
        # 4. webrtcbin: WebRTC core plugin
        pipeline_str = (
            f"webrtcbin name=sendrecv bundle-policy=max-bundle stun-server={STUN_SERVER} "
            f'udpsrc port={UDP_PORT} caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            "rtpjitterbuffer latency=0 ! rtph264depay ! "
            "rtph264pay config-interval=-1 name=payloader ! "
            "application/x-rtp,media=video,encoding-name=H264,payload=96 ! sendrecv. "
        )

        print(f"Starting the WebRTC Pipeline...")
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.webrtcbin = self.pipeline.get_by_name("sendrecv")

        # Signal binding
        self.webrtcbin.connect("on-negotiation-needed", self.on_negotiation_needed)
        self.webrtcbin.connect("on-ice-candidate", self.on_ice_candidate)

        self.pipeline.set_state(Gst.State.PLAYING)

    async def loop_handler(self):
        self.loop = asyncio.get_running_loop()
        self.start_pipeline()

        try:
            async for message in self.websocket:
                data = json.loads(message)

                if data["type"] == "answer":
                    # Received the browser's answer
                    print("Received Answer, set Remote Description")
                    _, sdpmsg = GstSdp.SDPMessage.new()
                    GstSdp.sdp_message_parse_buffer(bytes(data["sdp"], "utf-8"), sdpmsg)
                    answer = GstWebRTC.WebRTCSessionDescription.new(
                        GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
                    )
                    promise = Gst.Promise.new()
                    self.webrtcbin.emit("set-remote-description", answer, promise)
                    promise.interrupt()

                elif data["type"] == "ice":
                    # Received ICE Candidate from browser
                    candidate = data["candidate"]
                    mlineindex = data["sdpMLineIndex"]
                    self.webrtcbin.emit("add-ice-candidate", mlineindex, candidate)

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnects")
        finally:
            self.pipeline.set_state(Gst.State.NULL)


async def handler(websocket, path):
    print("New client connection...")
    client = WebRTCClient(websocket)
    await client.loop_handler()


def main():
    Gst.init(None)
    print(f"🌐 WebRTC Gateway listening on WebSocket: 0.0.0.0:{WS_PORT}")
    print(f"👂 Listening for DeepStream UDP data:{UDP_PORT}")

    start_server = websockets.serve(handler, "0.0.0.0", WS_PORT)

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    main()
