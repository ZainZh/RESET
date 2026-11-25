# SAM3 perception service

The runtime programs segment objects with text prompts through SAM3. They do not load SAM3 themselves: every episode
connects to one shared SAM3 server over a unix socket. The socket path is taken from `RSI_SAM3_ADDR`; the connection
authkey is fixed in code (`b"rsi-sam3"`, see `rsi/sam3_service.py` and `rsi/perception_sam3.py`). SAM3 code and
weights are **not** redistributed here.

## Setting up the provided server

1. Create a separate environment for the server (it needs PyTorch with CUDA; versions we used are in
   `requirements-sam3.txt`):
   ```bash
   python3.12 -m venv .venv-sam3
   .venv-sam3/bin/pip install -r requirements-sam3.txt
   ```
2. Install SAM3 from the official repository, https://github.com/facebookresearch/sam3 (we used commit
   `0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc`), following its instructions, e.g.
   `git clone https://github.com/facebookresearch/sam3 && .venv-sam3/bin/pip install -e ./sam3`.
3. Download the checkpoint. It is gated: request access at https://huggingface.co/facebook/sam3 , then
   `hf auth login` and `hf download facebook/sam3 sam3.pt --local-dir <dir>`; set `SAM3_CHECKPOINT=<dir>/sam3.pt`.
4. Start the server from the repository root (it imports `rsi.perception_sam3`, which also needs numpy and OpenCV):
   ```bash
   SAM3_CHECKPOINT=$SAM3_CHECKPOINT RSI_SAM3_ADDR=<sam3 socket> PYTHONPATH=. \
       .venv-sam3/bin/python -m rsi.sam3_service
   ```
   It prints `SAM3_SERVICE ready <sam3 socket>` when it accepts connections. Use the same `<sam3 socket>` value
   for `eval.py` (both default to `<system temp dir>/sam3.sock`; setting `RSI_SAM3_ADDR` explicitly avoids surprises).

The server builds the SAM3 image model (`build_sam3_image_model`, `Sam3Processor` with confidence threshold 0.3),
handles each client connection in its own thread, and runs inference under one lock with bfloat16 autocast.

## Replacing it

Any server that speaks the same protocol can be used instead (e.g. another segmentation model, or a server on a
different machine behind a forwarded unix socket):

- Transport: `multiprocessing.connection.Listener(<sam3 socket>, "AF_UNIX", authkey=b"rsi-sam3")`; messages are
  Python objects pickled by `Connection.send` / `Connection.recv`. One persistent connection per episode process.
- Request: a tuple `(rgb, prompts)`; `rgb` is a `numpy.uint8` array of shape `H x W x 3` (the rendered camera image),
  `prompts` a list of text prompts (the current client sends one prompt per request, e.g. `"apple"`, `"bowl"`,
  `"small cube"`).
- Response: a list with one entry per prompt, each a tuple `(masks, scores)`: `masks` is a boolean array
  `N x H x W` at the full image resolution (one instance mask per detection), `scores` a float array of length `N`.
  With no detection, return `(np.zeros((0, 1, 1), bool), np.zeros(0))`. The reference server returns every
  instance with score >= 0.3; the programs apply their own, higher thresholds on top.

Results obtained with a different segmenter are of course not the paper's numbers.
