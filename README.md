# Omniscient

A desktop layer that lets a person operate a Windows PC using only their eyes and the
webcam already built into it.

---

## The problem

For someone with no usable movement below the neck, no head movement, and no use of their
mouth, the standard assistive options are closed. Sip-and-puff needs a mouth. Head pointing
needs a neck. Switch scanning needs a muscle. Dedicated eye trackers work well but cost
several hundred dollars and are not present on the machine someone already owns.

Omniscient targets the case where the eyes are the only remaining input channel and the
hardware is whatever the PC already has.

## Why this is hard

At a typical viewing distance, one degree of gaze error covers roughly 110 pixels on
screen. Webcam gaze estimation is accurate to a few degrees at best, so the uncertainty is
on the order of 200 pixels. An ordinary button is about 100 x 30 pixels.

Looking directly at a control and clicking it is not difficult under these conditions. It
is impossible, and no amount of filtering recovers information the camera never captured.

Omniscient spends interaction steps to buy precision instead. A deliberate long blink
freezes the screen and overlays a coarse grid whose cells are comfortably larger than the
error. Selecting a cell magnifies it to fill the display, which divides the error by the
magnification factor and brings ordinary controls within reach. A second magnification is
available for fine work.

Because this is purely geometric, it needs to know nothing about what is on screen. It
works the same in a browser, a native application, a game, or a remote desktop session.

## Approach

- **An OS-level layer, not a browser extension.** An extension cannot launch the browser,
  switch windows, dismiss a system dialog, or recover from a crash — any of which would
  leave an eyes-only user locked out of their own machine.
- **Coarse-to-fine magnification** as the universal targeting mechanism.
- **A single radial menu** for every pointer verb — left, right, middle, double click and
  drag — so no action needs its own gesture.
- **Three gestures, chosen for safety.** A long blink to engage, distinguishable from a
  natural blink by a per-user calibrated duration. An off-screen glance to cancel, which
  cannot occur while attending to the screen. A corner dwell for the system menu, which
  remains reachable even when calibration has drifted badly enough that nothing else is.
- **Calibration that reports its own accuracy in degrees**, and refuses to store a profile
  that fails validation. A silently bad calibration produces an unusable session and no way
  for the user to understand why.
- **Continuous correction.** Every successful activation is a labeled sample — the target
  is known and the gaze at that moment is known — and is fed back to counter drift.

## Status

**Pre-alpha. Design complete, implementation not started.**

No application code exists in this repository yet. The MVP is specified as eight sequential
scopes, from gaze telemetry through calibration, signal conditioning, the overlay and OS
input path, coarse-to-fine targeting, activation, recovery, and text entry.

Target platform for the MVP is Windows.

## Non-goals for the MVP

Recorded here so they are not mistaken for oversights: no browser extension, no
accessibility-tree semantic targeting, no macOS or Linux support, no autostart or
lock-screen support, no multi-monitor, and no support for hardware eye trackers — though
the gaze source sits behind an interface so one can be added without changes elsewhere.

## License

Not yet determined.
