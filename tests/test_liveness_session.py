"""
Tests for Presento Phase 9–15: LivenessSession State Machine & Temporal Aggregator.
Verifies all liveness state transitions, real locks, lease expiration, face disappearance,
identity switches, model disagreement, hysteresis, and face quality gating.
"""
import time
import numpy as np
import pytest

from edge.liveness_session import LivenessSession, LivenessState, LivenessDecision


class TestLivenessSessionStateMachine:
    def test_initial_state_transitions_to_checking(self):
        session = LivenessSession(face_id=1, window_size=10)
        assert session.state == LivenessState.NEW

        # Feed 1 observation
        dec = session.update(frame_id=1, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.92, mf_p_spoof=0.08)
        assert dec.state == LivenessState.CHECKING
        assert dec.is_live is False
        assert dec.ui_status == "Verifying liveness..."

    def test_real_locked_transition_with_consecutive_windows(self):
        """Verify entering REAL_LOCKED requires meeting ratio threshold over consecutive windows."""
        session = LivenessSession(
            face_id=1,
            window_size=5,
            real_entry_ratio=0.80,
            required_consecutive_windows=2,
        )

        # Feed 5 real observations -> ratio is 1.0 (>= 0.80) -> consecutive = 1
        for i in range(5):
            dec = session.update(frame_id=i + 1, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert dec.state == LivenessState.CHECKING  # Needs 2 consecutive windows

        # Feed 6th real observation -> consecutive window 2 reached -> enters REAL_LOCKED!
        dec = session.update(frame_id=6, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert dec.state == LivenessState.REAL_LOCKED
        assert dec.is_live is True
        assert dec.ui_status == "Live person verified"
        assert dec.lease_remaining > 0.0

    def test_real_locked_persistence_under_minor_score_dip(self):
        """Verify REAL_LOCKED remains stable even if an individual frame dips (no UI flicker)."""
        session = LivenessSession(
            face_id=1,
            window_size=10,
            real_entry_ratio=0.80,
            real_exit_ratio=0.50,
            required_consecutive_windows=1,
        )

        # Lock into REAL
        for i in range(10):
            session.update(frame_id=i, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert session.state == LivenessState.REAL_LOCKED

        # One fluctuating frame with lower scores (e.g. 0.60 / 0.40)
        dec = session.update(frame_id=11, mn3_p_real=0.45, mn3_p_spoof=0.55, mf_p_real=0.80, mf_p_spoof=0.20)
        # MUST REMAIN REAL_LOCKED!
        assert dec.state == LivenessState.REAL_LOCKED
        assert dec.is_live is True

    def test_model_disagreement_does_not_prematurely_flip_to_spoof(self):
        """Verify MN3=REAL and MiniFAS=SPOOF produces disagreement without immediate spoof flip."""
        session = LivenessSession(face_id=2, window_size=10, spoof_entry_ratio=0.85)

        # Feed disagreement frames: MN3 says REAL (0.85), MiniFAS says SPOOF (0.20)
        for i in range(10):
            dec = session.update(frame_id=i, mn3_p_real=0.85, mn3_p_spoof=0.15, mf_p_real=0.20, mf_p_spoof=0.80)

        # Both_spoof_ratio is 0.0 because MN3 was real -> Cannot enter SPOOF
        assert dec.state == LivenessState.CHECKING
        assert dec.is_attack is False
        assert dec.both_spoof_ratio == 0.0

    def test_spoof_transition_when_both_models_agree(self):
        """Verify persistent agreement on SPOOF leads to SPOOF state."""
        session = LivenessSession(
            face_id=3,
            window_size=5,
            spoof_entry_ratio=0.80,
            required_consecutive_windows=2,
        )

        # Feed spoof frames (both say spoof)
        for i in range(6):
            dec = session.update(frame_id=i, mn3_p_real=0.05, mn3_p_spoof=0.95, mf_p_real=0.05, mf_p_spoof=0.95)

        assert dec.state == LivenessState.SPOOF
        assert dec.is_attack is True
        assert dec.is_live is False
        assert dec.ui_status == "Presentation attack detected"

    def test_temporary_face_disappearance_triggers_hold(self):
        """Verify face missing for <= hold_timeout enters HOLD and restores on return."""
        session = LivenessSession(face_id=4, window_size=5, hold_timeout=1.5, required_consecutive_windows=1)

        # Establish REAL_LOCKED
        for i in range(5):
            session.update(frame_id=i, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert session.state == LivenessState.REAL_LOCKED

        t0 = session.last_seen

        # Face missing for 0.5s (< 1.5s hold timeout)
        dec_hold = session.on_face_absent(now=t0 + 0.5)
        assert dec_hold.state == LivenessState.HOLD
        assert dec_hold.ui_status == "Maintaining verification..."
        assert dec_hold.is_live is False  # In HOLD, not currently live for attendance

        # Face returns quickly at t0 + 0.8s
        dec_return = session.update(frame_id=10, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert dec_return.state == LivenessState.REAL_LOCKED
        assert dec_return.is_live is True

    def test_long_face_disappearance_triggers_reset(self):
        """Verify face missing > hold_timeout causes session to RESET (DO NOT MAKE REAL IMMORTAL)."""
        session = LivenessSession(face_id=5, window_size=5, hold_timeout=1.5, required_consecutive_windows=1)

        # Establish REAL_LOCKED
        for i in range(5):
            session.update(frame_id=i, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert session.state == LivenessState.REAL_LOCKED

        t0 = session.last_seen

        # Face absent for 3.0s (> 1.5s timeout)
        dec_reset = session.on_face_absent(now=t0 + 3.0)
        assert dec_reset.state == LivenessState.RESET
        assert dec_reset.is_live is False

    def test_zero_state_inheritance_across_different_face_ids(self):
        """Verify Student A's REAL_LOCKED state is NEVER inherited by Student B."""
        session_a = LivenessSession(face_id=100, window_size=5, required_consecutive_windows=1)
        for i in range(5):
            session_a.update(frame_id=i, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert session_a.state == LivenessState.REAL_LOCKED

        # Person A leaves, Person B arrives with face_id=200
        session_b = LivenessSession(face_id=200, window_size=5, required_consecutive_windows=1)
        # Brand new session must be NEW, NOT REAL_LOCKED
        assert session_b.state == LivenessState.NEW
        assert session_b.face_id == 200

        dec_b = session_b.update(frame_id=10, mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.95, mf_p_spoof=0.05)
        assert dec_b.state == LivenessState.CHECKING
        assert dec_b.is_live is False

    def test_bad_frame_ignored_from_evidence_buffer(self):
        """Verify Phase 15: Low quality crop is ignored, NEVER treated as spoof."""
        session = LivenessSession(face_id=7, window_size=5)

        # Flat blurry/dark crop
        bad_crop = np.full((100, 100, 3), 10, dtype=np.uint8)  # Too dark

        initial_count = len(session.observations)
        dec = session.update(
            frame_id=1,
            mn3_p_real=0.10,
            mn3_p_spoof=0.90,
            mf_p_real=0.10,
            mf_p_spoof=0.90,
            crop=bad_crop,
            face_confidence=0.85,
        )

        # Bad crop should be rejected by quality filter and NOT added to observations
        assert len(session.observations) == initial_count
        assert dec.diagnostics["quality_ok"] is False
        assert "dark" in dec.diagnostics["quality_reason"].lower()
        # MUST NOT transition to SPOOF on a bad frame
        assert dec.state != LivenessState.SPOOF

    def test_temporal_ema_smoothing(self):
        """Verify EMA smoothing tracks average probability without wild jumps."""
        session = LivenessSession(face_id=8, ema_alpha=0.20)

        # Initial EMA is 0.50
        assert session.ema_p_real == 0.50

        # Feed frame with avg 1.0: 0.20 * 1.0 + 0.80 * 0.50 = 0.60
        dec = session.update(frame_id=1, mn3_p_real=1.0, mn3_p_spoof=0.0, mf_p_real=1.0, mf_p_spoof=0.0)
        assert abs(dec.ema_p_real - 0.60) < 1e-3
