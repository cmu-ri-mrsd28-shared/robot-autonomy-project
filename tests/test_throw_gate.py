import numpy as np
import pytest
from catchbox.hardware.throw_gate import ThrowGate


def test_stationary_ball_does_not_start():
    gate=ThrowGate()
    for i in range(8):
        assert gate.observe([.5+(i%2)*.002,0,.5],i*.03)
        assert not gate.has_motion()


def test_moving_ball_needs_multiple_frames_and_duplicate_samples_do_not_count():
    gate=ThrowGate()
    assert gate.observe([0,0,.5],1.0)
    assert gate.observe([0,0,.5],1.0)
    assert gate.observe([.025,0,.5],1.05)
    assert not gate.has_motion()
    assert gate.observe([.05,0,.5],1.10)
    assert gate.has_motion()
    gate.active=True;gate.reset()
    assert not gate.active and not gate.has_motion()


def test_large_jump_is_rejected_and_does_not_poison_next_reading():
    gate=ThrowGate();assert gate.observe([0,0,.5],1.)
    assert not gate.observe([1,0,.5],1.03)
    assert not gate.observe([1,0,.5],1.03)
    assert gate.observe([.01,0,.5],1.06)
    assert len(gate.history)==1 and not gate.has_motion()
    for i in range(1,5):
        assert gate.observe([.01+i*.015,0,.5],1.06+i*.03)
        if i<4:assert not gate.has_motion()
    assert gate.has_motion()


def test_old_observations_are_not_motion_evidence():
    gate=ThrowGate();gate.observe([0,0,.5],0.);gate.observe([.02,0,.5],.04)
    gate.observe([.1,0,.5],.5)
    assert len(gate.history)==1 and not gate.has_motion()
