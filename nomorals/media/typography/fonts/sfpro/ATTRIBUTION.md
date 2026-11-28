# SF Pro Text — attribution

SF Pro is the system typeface of iOS. It is proprietary software
owned by Apple Inc. and is licensed by Apple for use in Apple
products and in the design of interfaces for them; it is not open
source and it is not under the SIL Open Font License like the rest
of this directory.

These four weights (Regular 400, Medium 500, Semibold 600, Bold
700) are vendored here for one reason only: Devon's fake-message
mock screens render iPhone screens, and an iPhone screen set in
any other face is visibly not an iPhone screen — the owner's
standing order is that the mock screens be pixel-identical to the
real ones, and the typeface is the largest remaining delta after
colours and geometry. The files are used only as the rendering
face for the iPhone style of the mock-screen renderer
(`nomorals/social/chat/mock_screens/`); they are not re-licensed,
re-sold, or claimed as Devon's own work.

Source: the SF Pro Text TTFs as distributed by the sfwin project
(https://github.com/blaisck/sfwin), which packages Apple's SF
fonts for Windows. The files were fetched from that project's
raw file hosting and are byte-identical to the copies recorded
in `../MANIFEST.json` (sha256 there).

If Apple ever asks for their removal, delete this directory and
the "SF Pro Text" entry in `../MANIFEST.json`; the renderer falls
back to Inter for the iPhone style (see `_Canvas.font`).
