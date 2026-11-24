"""One shared SAM3 process for all rsi.dev workers (unix socket, one thread per client, inference under a lock).
Run in a Python environment with SAM3 installed, with PYTHONPATH=<sam3 checkout>:<this repository>.
Protocol: client sends (rgb uint8 HxWx3, [prompt, ...]); server replies [(masks, scores), ...] one per prompt."""
import os, threading, torch
from multiprocessing.connection import Listener
from PIL import Image
from rsi import perception_sam3 as S

ADDR = os.environ.get("RSI_SAM3_ADDR", __import__('tempfile').gettempdir() + "/sam3.sock")
KEY = b"rsi-sam3"
LOCK = threading.Lock()


def serve(conn, proc):
    try:
        while True:
            rgb, prompts = conn.recv()
            with LOCK, torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                state = proc.set_image(Image.fromarray(rgb))
                out = [S._prompt(proc, state, p) for p in prompts]
            conn.send(out)
    except EOFError:
        pass
    finally:
        conn.close()


def main():
    proc = S.build()
    if os.path.exists(ADDR):
        os.unlink(ADDR)
    with Listener(ADDR, "AF_UNIX", authkey=KEY) as ls:
        print("SAM3_SERVICE ready", ADDR, flush=True)
        while True:
            threading.Thread(target=serve, args=(ls.accept(), proc), daemon=True).start()


if __name__ == "__main__":
    main()
