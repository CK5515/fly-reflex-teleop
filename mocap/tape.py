"""Record the Rokoko SDK's TCP stream to a file and play it back later.

Glove time is shared, so everything downstream is developed against a tape.
The tape stores the raw bytes with their arrival times and knows nothing
about the protocol: on playback this script acts as the SDK server, so any
client (including Rokoko's own rgmp_stream.py) works unchanged.

    python -m mocap.tape record data/tapes/wave.tape      # Ctrl-C to stop
    python -m mocap.tape replay data/tapes/wave.tape      # serves on 12276

Stop the real rokoko-sdk before replaying, or pass --port to use another one.
A client must reconnect to see the tape again after it ends.
"""

import argparse
import socket
import struct
import time
from pathlib import Path

MAGIC = b"RTAPE1\n"
CHUNK_HEADER = struct.Struct("<dI")  # seconds since recording start, byte count
SDK_PORT = 12276


def record(path: Path, host: str, port: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with socket.create_connection((host, port)) as sock, path.open("wb") as tape:
        tape.write(MAGIC)
        print(f"recording {host}:{port} -> {path}  (Ctrl-C to stop)")
        start = time.monotonic()
        try:
            while True:
                data = sock.recv(65536)
                if not data:
                    print("server closed the connection")
                    break
                tape.write(CHUNK_HEADER.pack(time.monotonic() - start, len(data)))
                tape.write(data)
                total += len(data)
        except KeyboardInterrupt:
            pass
    print(f"saved {total} bytes, {time.monotonic() - start:.1f} s")


def read_chunks(path: Path) -> list[tuple[float, bytes]]:
    raw = path.read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError(f"{path} is not a tape file")
    chunks = []
    offset = len(MAGIC)
    while offset + CHUNK_HEADER.size <= len(raw):
        t, n = CHUNK_HEADER.unpack_from(raw, offset)
        offset += CHUNK_HEADER.size
        chunks.append((t, raw[offset : offset + n]))
        offset += n
    return chunks


def replay(path: Path, host: str, port: int, speed: float, once: bool) -> None:
    chunks = read_chunks(path)
    duration = chunks[-1][0] if chunks else 0.0
    with socket.create_server((host, port)) as server:
        print(f"serving {path} ({duration:.1f} s) on {host}:{port}")
        while True:
            client, addr = server.accept()
            print(f"client {addr[0]}:{addr[1]} connected")
            with client:
                start = time.monotonic()
                try:
                    for t, data in chunks:
                        wait = t / speed - (time.monotonic() - start)
                        if wait > 0:
                            time.sleep(wait)
                        client.sendall(data)
                except (BrokenPipeError, ConnectionResetError):
                    print("client left")
            if once:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="save the live stream to a tape")
    rec.add_argument("tape", type=Path)
    rec.add_argument("--host", default="127.0.0.1")
    rec.add_argument("--port", type=int, default=SDK_PORT)

    rep = sub.add_parser("replay", help="serve a tape as if it were the SDK")
    rep.add_argument("tape", type=Path)
    rep.add_argument("--host", default="127.0.0.1")
    rep.add_argument("--port", type=int, default=SDK_PORT)
    rep.add_argument("--speed", type=float, default=1.0)
    rep.add_argument("--once", action="store_true", help="exit after one client")

    args = parser.parse_args()
    if args.command == "record":
        record(args.tape, args.host, args.port)
    else:
        replay(args.tape, args.host, args.port, args.speed, args.once)


if __name__ == "__main__":
    main()
