# Omniscient

Operate a  PC using only your eyes and the webcam already built into it.

Omniscient is a desktop layer for people who have no usable movement below the neck, no
head movement, and no use of their mouth. Where the standard assistive options all require
something the user does not have — a mouth for sip-and-puff, a neck for head pointing, a
muscle for switch scanning — Omniscient requires only the eyes and hardware the machine
already has.

## Overview

Dedicated eye trackers work well and cost several hundred dollars. Omniscient targets the
webcam that is already in the laptop, which is roughly an order of magnitude less accurate,
and closes that gap in the interaction design rather than in the sensor.

At a typical viewing distance one degree of gaze error covers about 110 pixels on screen.
Webcam gaze estimation is accurate to a few degrees, so the uncertainty is on the order of
200 pixels, while an ordinary button is about 100 × 30 pixels. Looking straight at a control
and clicking it is not merely unreliable under these conditions — it is impossible, and no
amount of filtering recovers information the camera never captured.

Omniscient spends interaction steps to buy precision instead.

## How it works

A deliberate long blink freezes the screen and overlays a coarse grid whose cells are
comfortably larger than the gaze error. Selecting a cell magnifies it to fill the display,
which divides the error by the magnification factor and brings ordinary controls within
reach. A second magnification is available for fine work, and a dwell on the resolved point
clicks it.

Because this is purely geometric, it needs to know nothing about what is on screen. It
behaves identically in a browser, a native application, a game, or a remote desktop
session.

## Capabilities

- **Coarse-to-fine magnification** as a universal targeting mechanism, requiring no
  knowledge of the underlying application.
- **A single radial menu** covering every pointer action — left, right, middle, double
  click and drag — so no action needs a gesture of its own.
- **Three gestures, chosen for safety.** A long blink to engage, distinguished from a
  natural blink by a per-user calibrated duration. An off-screen glance to cancel, which
  cannot occur while attending to the screen. A corner dwell for the system menu, which
  stays reachable even when calibration has drifted far enough that nothing else is.
- **Calibration that reports its own accuracy** in degrees of visual angle and refuses to
  store a profile that fails validation, so a bad calibration is never allowed to become an
  unusable session the user cannot diagnose.
- **Continuous correction.** Every successful activation is a labelled sample, since both
  the target and the gaze at that moment are known, and is fed back to counter drift.
- **Gaze-rate scrolling** at the screen edges, armed on a delay so that reading toward the
  bottom of a page does not trigger it.
- **On-screen keyboard** driven by dwell, with word prediction.

## Requirements

Windows, and any standard webcam. No additional hardware, no infrared illuminator, and no
dedicated eye tracker.

Hardware eye trackers are not required, but the gaze source sits behind an interface so one
can be added without changes elsewhere in the system.

## Status

Pre-alpha. The design is complete and implementation has not yet started, so there is
nothing to install today.

Development proceeds in eight sequential stages: gaze telemetry, calibration and accuracy
measurement, signal conditioning, the overlay and input path, coarse-to-fine targeting,
activation and the radial menu, scrolling and recovery, and text entry.

## Contributing

Issues and pull requests are welcome. The project favours small, focused modules with a
single clear responsibility, and treats measured accuracy as the metric that decides
whether a change is an improvement.

## License

Released under the MIT License. See [LICENSE](LICENSE).
