# facemesh fixtures — real faces, real video

## face-video.mp4

640x480, 8 fps, 48 frames, 6.000 s, h264. Two 3-second
segments, each a slow Ken Burns zoom over one portrait:

- Segment A (frames 0-23): a big smile, jaw open, eyes open
  (MediaPipe: mouthSmileLeft 0.927, mouthSmileRight 0.950,
  jawOpen 0.218).
- Segment B (frames 24-47): neutral, lips pressed, a slight
  pucker (MediaPipe: smile 0.0, mouthPucker 0.375).

Built from the two Unsplash portraits below with ffmpeg
`zoompan` (1.0 -> 1.12). The video test runs the whole
pipeline on it: decode -> detect -> map -> take. One frame
is skipped by design: frame 24 (t=3.0, the hard cut to
segment B) has a face in IMAGE mode, but VIDEO mode's
tracker loses it for exactly that one frame at the cut and
re-acquires on frame 25 — so the take lands 47 frames from
48 read, 1 skipped, and the tests pin exactly that.

## landmarks/*.json

Real MediaPipe Face Landmarker output for five portraits:
478 landmarks (468 + 10 iris) plus MediaPipe's own 52
blendshape scores, which the mapping tests use as the
ground-truth cross-check (the organ's weights come from
landmark geometry, never from these scores).

- u-woman1.json — the video's segment A face (big smile).
- u-woman2.json — the video's segment B face (neutral).
- men-3.json — smile 0.81/0.92, jawOpen 0.56.
- men-6.json — mouthPucker 0.99.
- women-8.json — smile 0.98/0.98.

## Sources

- u-woman1 / u-woman2: Unsplash portraits (free license),
  `images.unsplash.com` photo IDs
  `photo-1494790108377-be9c29b29330` and
  `photo-1534528741775-53994a69daeb`.
- men-*/women-*: `randomuser.me` portrait thumbnails
  (128x128), free for demo use.

The landmark index map these fixtures pin was verified by
eye on annotated portraits (see
`~/workspace/facemesh-verify/annotated-*.png`): the
subject's left is on the image right, so landmark 291 is
the subject's LEFT mouth corner and 263 the subject's LEFT
eye — confirmed on three asymmetric smiles in the corpus.
