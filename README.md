# fly-reflex-teleop

Junction 2026, Rokoko teleoperation track.

A human teleoperates a simulated robot arm through Rokoko motion capture, and a fly-brain model runs alongside it
as a reflex layer: events in the scene (an object approaching fast, contact) become sensory input to a spectral
mixture-of-experts model of the Drosophila brain, and the experts it routes to (escape, grooming) override the
operator with a reflex. A live view shows which expert fired and why the arm moved.

