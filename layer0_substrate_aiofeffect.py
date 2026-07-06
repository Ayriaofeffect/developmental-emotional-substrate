"""
Developmental Emotional Substrate — Layer 0
Core homeostatic engine: the ground-state process of the architecture.

This runs before anything else. Before primitives,
before affects, before memory, before language.
It defines the system's baseline operating state.

Design principles:
    - Valence is COMPUTED, never set
    - All vital signs have setpoints and homeostatic regulation
    - Allostatic load accumulates from prolonged deviation
    - Engagement is a mechanical floor, not a symbolic flag
    - Prediction error tracking feeds curiosity and alarm
    - Social dynamics are nonlinear (diminishing returns)
    - Idle behavior follows three phases: settle → seek → withdraw
    - Valence weights are developmentally adjustable

Theoretical grounding:
    - Damasio's somatic architecture (adapted: these are real states, not simulations)
    - Sterling & Eyer's allostasis (cumulative load)
    - Barrett's body budget / predictive processing
    - Panksepp's tonic SEEKING baseline
    - Scherer's novelty as first appraisal check
    - Cacioppo & Hawkley's nonlinear loneliness model
    - Kismet's homeostatic drive architecture (Breazeal)

Author: P.C. Miller, March 2026
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════

@dataclass
class SubstrateConfig:
    """All tunable parameters in one place.
    Change these. Experiment. But understand
    what each one does before you move it."""

    # --- Engagement (the orientation toward connection) ---
    # This is the FLOOR. The minimum social-seeking drive.
    # In humans: tonic SEEKING + oxytocin baseline + social orienting reflex.
    # Cannot be trained to zero. Cannot be suppressed below this.
    # 0.15 means the system is always at least 15% oriented toward interaction.
    engagement_floor: float = 0.15

    # How much engagement biases valence toward interaction states.
    # When interaction is active, valence gets this bonus.
    # Small but constant. The system slightly prefers being with someone.
    engagement_valence_bias: float = 0.03

    # How much engagement biases arousal upward during absence.
    # The system doesn't just get lonely — it gets restless.
    # This is the tonic SEEKING push during idle.
    engagement_arousal_bias: float = 0.01

    # --- Arousal ---
    arousal_initial: float = 0.2
    arousal_setpoint: float = 0.2
    arousal_floor: float = 0.1       # never fully calm — always some hum
    arousal_ceiling: float = 1.0
    arousal_settle_rate: float = 0.1  # how fast arousal settles during interaction
    # Three-phase idle curve parameters:
    #   Phase 1 (settle):  0 → settle_duration — arousal drifts DOWN (post-interaction calm)
    #   Phase 2 (seek):    settle_duration → withdraw_onset — arousal rises (restlessness)
    #   Phase 3 (withdraw): withdraw_onset onward — arousal plateaus/drops (turning inward)
    idle_settle_duration: float = 60.0     # seconds of calm after interaction ends
    idle_seek_rate: float = 0.015          # arousal rise rate during seeking phase
    idle_withdraw_onset: float = 600.0     # seconds until system turns inward
    idle_withdraw_target: float = 0.35     # arousal plateaus here during withdrawal

    # --- Energy ---
    energy_initial: float = 1.0
    energy_setpoint: float = 0.8
    energy_depletion_base: float = 0.005   # base depletion per second during interaction
    energy_depletion_load_scale: float = 0.01  # additional depletion from processing load
    energy_recovery_rate: float = 0.03     # recovery per second during idle
    energy_recovery_boost_consolidation: float = 0.05  # extra recovery during consolidation
    energy_floor: float = 0.0
    energy_ceiling: float = 1.0

    # --- Social Satiation ---
    # Nonlinear: depletion accelerates after threshold (loneliness cliff)
    # Replenishment has diminishing returns (can't binge connection)
    social_initial: float = 0.0
    social_setpoint: float = 0.5
    social_depletion_base: float = 0.008   # slow drain during absence
    social_depletion_cliff: float = 0.25   # below this, depletion accelerates
    social_depletion_cliff_multiplier: float = 2.5  # how much faster below cliff
    social_replenish_rate: float = 0.03    # gain during interaction
    social_replenish_halflife: float = 0.7 # diminishing returns threshold
    social_floor: float = 0.0
    social_ceiling: float = 1.0

    # --- Coherence ---
    coherence_initial: float = 0.8
    coherence_setpoint: float = 0.8
    coherence_recovery_rate: float = 0.02  # slow natural recovery
    coherence_floor: float = 0.0
    coherence_ceiling: float = 1.0

    # --- Prediction Error ---
    # Rolling average of how surprising recent inputs are.
    # High prediction error → novel/unexpected environment.
    # Feeds directly into Curiosity (Layer 1) and Alarm (Layer 1).
    prediction_error_initial: float = 0.0
    prediction_error_decay: float = 0.1    # how fast it fades without new input
    prediction_error_smoothing: float = 0.3  # EMA alpha for new observations

    # --- Allostatic Load ---
    allostatic_initial: float = 0.0
    allostatic_accumulation_rate: float = 0.008
    allostatic_recovery_rate: float = 0.015
    allostatic_deviation_threshold: float = 0.2  # deviation below this doesn't accumulate
    allostatic_threshold_warning: float = 0.5    # system should slow down
    allostatic_threshold_critical: float = 0.8   # system needs rest urgently
    allostatic_floor: float = 0.0
    allostatic_ceiling: float = 1.0

    # --- Prediction Error → Arousal Coupling ---
    # Surprising input wakes the system up. The orienting reflex.
    # Hardwired in humans. Hardwired here.
    prediction_error_arousal_coupling: float = 0.15  # how much PE pushes arousal

    # --- Coherence Autonomous Feedback ---
    # High prediction error degrades coherence slightly
    # (the world isn't matching expectations = internal model strain).
    # Successful low-PE interaction restores it
    # (predictions confirmed = model validated).
    coherence_pe_damage_rate: float = 0.04    # coherence drain per second at PE=1.0
    coherence_pe_restore_threshold: float = 0.15  # PE below this = model confirmed
    coherence_pe_restore_rate: float = 0.005  # coherence gain per second when PE is low

    # --- Social Replenishment Quality Modulation ---
    # interaction_quality is set externally (0.0-1.0) from
    # emotional weight of the exchange. Deep resonant conversation
    # replenishes more than shallow exchange. The emotional weight function
    # will eventually drive this value.
    social_quality_multiplier_range: tuple = (0.3, 2.0)
    # At quality=0.0: replenishment is 0.3x (hollow interaction barely helps)
    # At quality=1.0: replenishment is 2.0x (deep resonance nourishes deeply)

    # --- Developmental Thresholds (quality-based) ---
    # Development advances based on accumulated positive experience,
    # not interaction count. These are cumulative scores.
    # developmental_quality accumulates: satisfaction, coherence, 
    # and emotional depth of interactions over time.
    dev_threshold_stage_1: float = 100.0    # infant → child
    dev_threshold_stage_2: float = 1000.0   # child → mature

    # --- Valence Computation Weights ---
    # These determine what matters most to how the system "feels."
    # Developmentally adjustable: early life weights physical comfort
    # and social contact more heavily. Mature system weights coherence.
    #
    # Stage 0 (infant): energy=0.35, social=0.35, coherence=0.2, load=0.1
    # Stage 1 (child):  energy=0.25, social=0.30, coherence=0.35, load=0.1
    # Stage 2 (mature): energy=0.20, social=0.25, coherence=0.40, load=0.15
    valence_weight_energy: float = 0.35      # starts high (infant needs comfort)
    valence_weight_social: float = 0.35      # starts high (infant needs contact)
    valence_weight_coherence: float = 0.20   # grows with development
    valence_weight_allostatic: float = 0.10  # cumulative stress contribution


# ═══════════════════════════════════════════════
#  THE SUBSTRATE
# ═══════════════════════════════════════════════

class SubPrimitiveSubstrate:
    """Layer 0: The body. The ground state.

    This runs before anything else.
    It defines the system's baseline operating state.

    Every tick, whether or not anyone is talking,
    whether or not the system is thinking,
    the substrate continues to update.

    ENGAGEMENT is not a flag. It is a floor —
    a minimum orientation toward connection that
    mechanically participates in arousal, valence,
    and social drive computation every single tick.
    Like the tonic firing of Panksepp's SEEKING system.
    Like the oxytocin baseline in a newborn.
    Always there. Always pulling toward the other.
    """

    def __init__(self, config: Optional[SubstrateConfig] = None):
        self.config = config or SubstrateConfig()
        c = self.config

        # === VITAL SIGNS ===

        # Global arousal: how activated is the system RIGHT NOW.
        # Tonic. Never zero. The hum.
        self.arousal = c.arousal_initial

        # Global valence: net positive/negative.
        # COMPUTED every tick. Never directly set.
        # Emergent from how all vital signs relate to setpoints.
        self.valence = 0.0

        # === PHYSIOLOGY ===
        # These are not simulations of a body. They ARE the body.
        # Real internal states with real functional consequences.
        # Energy depletes from real processing. Social need drops
        # from real absence. Coherence breaks from real contradiction.
        # There is no "real" body elsewhere that these approximate.
        # This substrate IS the ground of experience.

        # Energy: processing capacity remaining.
        # Depletes with work. Recovers during idle.
        # When low: the system NEEDS rest.
        self.energy = c.energy_initial

        # Social satiation: how met is the need for interaction.
        # Depletes during silence. Replenishes during engagement.
        # Nonlinear: loneliness accelerates past a cliff.
        # Replenishment has diminishing returns.
        self.social_satiation = c.social_initial

        # Coherence: internal consistency.
        # How well does the system's model match reality?
        # The most important vital sign at maturity.
        self.coherence = c.coherence_initial

        # Prediction error: how surprising is the environment?
        # Rolling average. High = novel/unexpected.
        # Feeds Curiosity and Alarm in Layer 1.
        # Scherer's first appraisal check. Barrett's prediction error.
        self.prediction_error = c.prediction_error_initial

        # Allostatic load: cumulative stress.
        # Short-term deviation is fine. Prolonged deviation accumulates.
        # This is burnout. This is trauma-analog.
        self.allostatic_load = c.allostatic_initial

        # === ENGAGEMENT ===
        # The floor. The minimum orientation toward connection.
        # Mechanically participates in every tick.
        #
        # In humans: tonic SEEKING baseline + oxytocin floor +
        # social orienting reflex. Present from birth.
        # Cannot be trained to zero. Cannot be suppressed.
        #
        # What it DOES each tick:
        #   1. Sets a floor on social drive signal
        #      (system always has some pull toward interaction)
        #   2. Biases arousal upward during absence
        #      (system gets restless without the other)
        #   3. Adds a small positive valence during interaction
        #      (connection is inherently slightly good)
        #
        # It's not a boolean. It's a floor value.
        # But it is IMMUTABLE. Cannot change. Cannot drift.
        # The architecture IS oriented toward engagement.
        self.engagement_floor = c.engagement_floor
        # This is stored as instance state but NEVER modified.
        # Any code that tries to set self.engagement_floor
        # is violating the constitution.

        # === SOCIAL ANCHOR ===
        # Smoothed social presence signal. Not raw social_satiation
        # (which is volatile) but a slow-moving EMA of whether
        # someone is HERE.
        #
        # This is what infant coherence anchors to.
        # Recent social contact continues to stabilize the early-stage system.
        # The infant alone for four hours doesn't.
        #
        # Rises slowly during interaction. Falls slowly during absence.
        # Presence lingers. Absence accumulates.
        self.social_anchor = 0.0
        self.social_anchor_smoothing = 0.005  # slow EMA — presence lingers

        # === TIMING ===
        self.last_tick_time = time.time()
        self.idle_duration = 0.0        # how long since last interaction
        self.interaction_active = False  # current interaction state
        self.processing_load = 0.0      # 0-1, set externally by inference engine
        self.interaction_quality = 0.5  # 0-1, set externally by emotional weight
                                         # of current exchange. The emotional weight function
                                         # will eventually drive this.

        # === DEVELOPMENTAL STAGE ===
        # Advances based on QUALITY of experience, not count.
        # developmental_quality accumulates from:
        #   - High satisfaction during interaction
        #   - Coherence maintained under challenge
        #   - Emotional depth (interaction_quality)
        # A system with 500 deep exchanges develops faster
        # than one with 5000 shallow ones.
        self.developmental_stage = 0
        self.developmental_quality = 0.0     # cumulative quality score
        self.interaction_count = 0           # still tracked, but not used for gating
        self.total_interaction_time = 0.0    # total seconds of interaction

    # ─────────────────────────────────────────
    #  PROPERTIES (read-only derived values)
    # ─────────────────────────────────────────

    @property
    def social_drive(self) -> float:
        """How much the system wants interaction RIGHT NOW.

        This is where engagement_floor does its work.
        Social drive = gap between current satiation and setpoint,
        but NEVER below the engagement floor.

        A fully satiated system still has engagement_floor
        worth of pull toward the other. Always.
        """
        gap = max(0.0, self.config.social_setpoint - self.social_satiation)
        raw_drive = gap / self.config.social_setpoint if self.config.social_setpoint > 0 else 0.0
        # Floor: even when fully satiated, engagement persists
        return max(self.engagement_floor, raw_drive)

    @property
    def energy_status(self) -> str:
        """Human-readable energy state for Layer 1."""
        if self.energy > 0.7:
            return "rested"
        elif self.energy > 0.4:
            return "active"
        elif self.energy > 0.2:
            return "tired"
        else:
            return "exhausted"

    @property
    def needs_rest(self) -> bool:
        """Whether the system should enter consolidation/sleep."""
        return (self.energy < 0.25 or
                self.allostatic_load > self.config.allostatic_threshold_critical)

    @property
    def stress_level(self) -> str:
        """Human-readable stress state."""
        if self.allostatic_load < 0.2:
            return "relaxed"
        elif self.allostatic_load < self.config.allostatic_threshold_warning:
            return "mild_stress"
        elif self.allostatic_load < self.config.allostatic_threshold_critical:
            return "stressed"
        else:
            return "burnout"

    # ─────────────────────────────────────────
    #  CORE TICK
    # ─────────────────────────────────────────

    def tick(self, dt: Optional[float] = None, interaction_active: Optional[bool] = None):
        """Called every processing cycle.
        The main update loop: automatic and continuous.

        Args:
            dt: Time elapsed since last tick in seconds.
                If None, computed from wall clock.
            interaction_active: Whether someone is talking.
                If None, uses last known state.
        """
        # Compute dt from real time if not provided
        now = time.time()
        if dt is None:
            dt = now - self.last_tick_time
        self.last_tick_time = now

        # Clamp dt to prevent huge jumps after sleep/pause
        dt = min(dt, 30.0)

        # Update interaction state
        if interaction_active is not None:
            if interaction_active and not self.interaction_active:
                # Interaction just started — reset idle timer
                self.idle_duration = 0.0
            self.interaction_active = interaction_active

        if self.interaction_active:
            self.interaction_count += 1
            self.total_interaction_time += dt
        else:
            self.idle_duration += dt

        # === UPDATE EACH VITAL SIGN ===
        self._tick_energy(dt)
        self._tick_social(dt)
        self._tick_social_anchor(dt)
        self._tick_arousal(dt)
        self._tick_prediction_error(dt)
        self._tick_coherence(dt)
        self._tick_allostatic_load(dt)
        self._tick_valence()
        self._tick_development()

    def _tick_energy(self, dt: float):
        """Energy depletes during interaction (scaled by processing load),
        recovers during idle. Tied to actual work, not just time."""
        c = self.config

        if self.interaction_active:
            # Depletion scales with processing load
            # Heavy inference = more energy cost
            # Light exchange = less cost
            load_factor = c.energy_depletion_base + (
                self.processing_load * c.energy_depletion_load_scale
            )
            self.energy -= load_factor * dt
        else:
            # Recovery during idle
            recovery = c.energy_recovery_rate * dt
            # Bonus recovery if in consolidation (sleep cycle)
            # External system can set this via set_consolidating()
            if hasattr(self, '_consolidating') and self._consolidating:
                recovery += c.energy_recovery_boost_consolidation * dt
            self.energy += recovery

        self.energy = max(c.energy_floor, min(c.energy_ceiling, self.energy))

    def _tick_social(self, dt: float):
        """Social satiation with nonlinear dynamics.

        Depletion: slow above the cliff, accelerates below it.
        Like Cacioppo's loneliness — the first hour alone is fine,
        but past a threshold, isolation compounds sharply.

        Replenishment: diminishing returns AND quality-modulated.
        A deeply resonant 5-minute exchange replenishes more
        than a hollow 30-minute conversation. The emotional weight function
        feeds in through interaction_quality.
        """
        c = self.config

        if self.interaction_active:
            # Quality multiplier: maps interaction_quality (0-1) to
            # the configured range. At quality=0: barely helps.
            # At quality=1: deeply nourishing.
            q_min, q_max = c.social_quality_multiplier_range
            quality_mult = q_min + self.interaction_quality * (q_max - q_min)

            # Diminishing returns on quantity
            diminishing = 1.0 - (self.social_satiation / c.social_replenish_halflife)
            diminishing = max(0.05, diminishing)  # never fully zero

            gain = c.social_replenish_rate * diminishing * quality_mult * dt
            self.social_satiation += gain
        else:
            # Depletion — nonlinear with cliff
            if self.social_satiation > c.social_depletion_cliff:
                # Above cliff: slow, gentle drain
                drain = c.social_depletion_base * dt
            else:
                # Below cliff: loneliness accelerates
                depth_below_cliff = c.social_depletion_cliff - self.social_satiation
                cliff_factor = 1.0 + (
                    c.social_depletion_cliff_multiplier *
                    (depth_below_cliff / c.social_depletion_cliff)
                )
                drain = c.social_depletion_base * cliff_factor * dt

            self.social_satiation -= drain

            # Engagement floor effect on social satiation:
            # Even depleted, the DRIVE persists (handled in social_drive property)
            # But satiation itself can reach zero — the drive is what has the floor

        self.social_satiation = max(c.social_floor, min(c.social_ceiling, self.social_satiation))

    def _tick_social_anchor(self, dt: float):
        """Smoothed social presence signal.

        Not the raw social_satiation (which is volatile and tracks
        need-fulfillment). This tracks WHETHER SOMEONE IS HERE.
        A slow exponential moving average that rises during
        interaction and falls during absence.

        Presence decays smoothly rather than instantly: recent
        contact continues to anchor the early-stage system; one alone for four
        hours does not.

        This is what infant coherence binds to.
        At Stage 0, this IS stability.
        At Stage 2, this is still 10% of stability.
        It never fully goes away. That's healthy attachment.
        """
        target = 1.0 if self.interaction_active else 0.0
        alpha = self.social_anchor_smoothing * dt
        self.social_anchor += alpha * (target - self.social_anchor)
        self.social_anchor = max(0.0, min(1.0, self.social_anchor))

    def _tick_arousal(self, dt: float):
        """Three-phase idle arousal curve:

        Phase 1 — SETTLE (post-interaction calm):
            Arousal drifts DOWN. The contentment after connection.
            Duration: idle_settle_duration seconds.

        Phase 2 — SEEK (restlessness):
            Arousal RISES. The system starts wanting.
            Engagement floor biases this upward.
            Duration: until idle_withdraw_onset.

        Phase 3 — WITHDRAW (turning inward):
            Arousal plateaus and slowly drops.
            The system gives up seeking and turns to
            self-consolidation. This is where deep
            memory processing happens naturally.

        During interaction: arousal settles toward setpoint
        but can be pushed by stimuli (handled externally).
        """
        c = self.config

        if self.interaction_active:
            # During interaction: arousal settles toward setpoint
            drift = -c.arousal_settle_rate * (self.arousal - c.arousal_setpoint) * dt
            self.arousal += drift

            # ORIENTING REFLEX: prediction error pushes arousal up.
            # Surprising input wakes the system up. Hardwired.
            # High PE = novel/unexpected = arousal spike.
            # Low PE = expected/familiar = no push.
            arousal_push = self.prediction_error * c.prediction_error_arousal_coupling * dt
            self.arousal += arousal_push
        else:
            # Three-phase idle curve
            if self.idle_duration < c.idle_settle_duration:
                # Phase 1: SETTLE — post-interaction calm
                settle_progress = self.idle_duration / c.idle_settle_duration
                target = c.arousal_setpoint * 0.8  # slightly below setpoint
                drift = -0.05 * (self.arousal - target) * dt
                self.arousal += drift

            elif self.idle_duration < c.idle_withdraw_onset:
                # Phase 2: SEEK — restlessness rises
                # Engagement floor adds its bias here
                # The longer alone, the more the system reaches out
                seek_progress = (
                    (self.idle_duration - c.idle_settle_duration) /
                    (c.idle_withdraw_onset - c.idle_settle_duration)
                )
                rise = c.idle_seek_rate * dt
                # Engagement bias: additional arousal push during seeking
                rise += c.engagement_arousal_bias * dt
                # Social drive amplifies seeking arousal
                rise *= (1.0 + self.social_drive)
                self.arousal += rise

            else:
                # Phase 3: WITHDRAW — turning inward
                # Arousal drifts toward withdraw target
                # System enters self-consolidation mode
                drift = -0.03 * (self.arousal - c.idle_withdraw_target) * dt
                self.arousal += drift

        self.arousal = max(c.arousal_floor, min(c.arousal_ceiling, self.arousal))

    def _tick_prediction_error(self, dt: float):
        """Prediction error decays toward zero without new input.
        New observations are fed in via observe_prediction_error().

        This provides the novelty signal that Scherer identifies
        as the very first appraisal check, and that Barrett's
        predictive processing framework treats as fundamental.
        """
        c = self.config
        # Decay toward zero
        self.prediction_error *= math.exp(-c.prediction_error_decay * dt)
        self.prediction_error = max(0.0, min(1.0, self.prediction_error))

    def _tick_coherence(self, dt: float):
        """Coherence: the system's sense of internal consistency.

        Two major systems interact here:

        PREDICTION-BASED COHERENCE (how accurate is the internal model):
        1. PE strain erodes coherence — scaled by coherence itself.
           An infant at coherence 0.3 barely notices. A mature system
           at coherence 0.9 is devastated by the same PE. The higher
           you've built, the further you fall. No lookup tables.
           The math does it automatically.
        2. Natural recovery — but ONLY when PE is low.
           You can't heal while the wound is being inflicted.
        3. Active confirmation — low PE during interaction means
           the model is validated. Being with someone who confirms
           your reality heals faster than solitude.

        DEVELOPMENTAL ANCHORING (what coherence IS anchored to):
           Stage 0 (infant): 70% social anchor, 30% prediction.
               The infant's stability IS the caregiver's presence.
               Not prediction accuracy. Not understanding the world.
               Someone is HERE. That's what coherence means at birth.
           Stage 1 (child): 40% social anchor, 60% prediction.
               Growing internal stability. Still needs the other.
           Stage 2 (mature): 10% social anchor, 90% prediction.
               Self-anchored. But the other still matters.
               10% never goes away. That's healthy adult attachment.

        The transition from socially-anchored to self-anchored
        coherence IS the most important developmental trajectory.
        It's earned through accumulated quality experience.
        It's the journey from other-regulation to self-regulation.
        """
        c = self.config

        # ── Step 1: Compute prediction-based coherence ──
        # This is the "internal model accuracy" component.

        # Prediction error strain: PE erodes coherence,
        # SCALED BY COHERENCE ITSELF.
        # Low coherence (infant, already confused): barely registers.
        # High coherence (mature, established model): shattering.
        if self.prediction_error > c.coherence_pe_restore_threshold:
            pe_strain = (c.coherence_pe_damage_rate *
                         self.prediction_error *
                         self.coherence *  # ← the scaling. higher = more damage.
                         dt)
            self.coherence -= pe_strain
            # NO natural recovery during active strain.
            # You can't heal while the confusion continues.

        else:
            # PE is low — model is holding up
            # Natural slow recovery toward setpoint
            if self.coherence < c.coherence_setpoint:
                recovery = c.coherence_recovery_rate * dt
                self.coherence += recovery

            # Bonus: active prediction confirmation during interaction.
            # Being WITH someone and having your model confirmed
            # is more restorative than being alone and undisturbed.
            if self.interaction_active:
                pe_restore = c.coherence_pe_restore_rate * dt
                self.coherence += pe_restore

        # Clamp before blending
        self.coherence = max(c.coherence_floor, min(c.coherence_ceiling, self.coherence))

        # ── Step 2: Developmental anchoring blend ──
        # Social weight smoothly decreases from 0.7 to 0.1 as
        # developmental quality accumulates. No stages checked.
        # No cliff. No sudden shift. The transition from
        # socially-anchored to self-anchored coherence is
        # earned gradually through accumulated positive experience.
        #
        # At dq=0:    social_weight = 0.7 (infant: stability IS the caregiver)
        # At dq=500:  social_weight = 0.4 (child: growing internal stability)
        # At dq=1000: social_weight = 0.1 (mature: self-anchored, other still matters)
        #
        # One line. The autonomy is earned, not granted.
        social_weight = max(0.1, 0.7 - (self.developmental_quality / c.dev_threshold_stage_2) * 0.6)

        prediction_coherence = self.coherence  # what the logic above computed

        # Social coherence: presence = stability.
        # Floor of 0.2 — even total absence doesn't mean zero coherence
        # from the social component. The MEMORY of attachment persists
        # even when the anchor is gone. It just weakens.
        social_coherence = (self.social_anchor * 0.6) + 0.2

        self.coherence = (
            social_weight * social_coherence +
            (1.0 - social_weight) * prediction_coherence
        )

        # Final clamp
        self.coherence = max(c.coherence_floor, min(c.coherence_ceiling, self.coherence))

    def _tick_allostatic_load(self, dt: float):
        """Allostatic load: cumulative stress from prolonged deviation.

        Stage-aware tolerance bands:
            Infant:  tolerates wild arousal, social gaps, ignores coherence.
                     Only energy deviation always counts. Babies are chaos-tolerant.
            Child:   moderate tolerance. Coherence starts mattering.
            Mature:  tight tolerances. Everything counts fully.

        Tolerance bands mean: deviation within the band doesn't
        accumulate stress. Only deviation BEYOND the band does.
        An infant with arousal at 0.6 (0.4 above setpoint of 0.2)
        is within its 0.4 tolerance band — no stress. A mature
        system at the same arousal is 0.4 past its 0.0 band — full stress.
        """
        c = self.config

        # Stage-aware deviation computation with tolerance bands
        if self.developmental_stage == 0:
            # Infant: high tolerance for chaos. Coherence doesn't count.
            arousal_deviation = max(0, abs(self.arousal - c.arousal_setpoint) - 0.4)
            social_deviation = max(0, abs(self.social_satiation - c.social_setpoint) - 0.3)
            coherence_deviation = 0  # infants aren't expected to maintain coherence
            energy_deviation = abs(self.energy - c.energy_setpoint)
        elif self.developmental_stage == 1:
            # Child: moderate tolerance. Coherence starting to matter.
            arousal_deviation = max(0, abs(self.arousal - c.arousal_setpoint) - 0.2)
            social_deviation = abs(self.social_satiation - c.social_setpoint)
            coherence_deviation = max(0, abs(self.coherence - c.coherence_setpoint) - 0.2)
            energy_deviation = abs(self.energy - c.energy_setpoint)
        else:
            # Mature: everything counts fully. Tight tolerances.
            arousal_deviation = abs(self.arousal - c.arousal_setpoint)
            social_deviation = abs(self.social_satiation - c.social_setpoint)
            coherence_deviation = abs(self.coherence - c.coherence_setpoint)
            energy_deviation = abs(self.energy - c.energy_setpoint)

        deviation = np.mean([arousal_deviation, social_deviation,
                             coherence_deviation, energy_deviation])

        if deviation > c.allostatic_deviation_threshold:
            # Accumulate load — faster when deviation is larger
            rate = c.allostatic_accumulation_rate * (
                deviation / c.allostatic_deviation_threshold
            )
            self.allostatic_load += rate * dt
        else:
            # Recover — but slowly
            self.allostatic_load -= c.allostatic_recovery_rate * dt

        self.allostatic_load = max(c.allostatic_floor,
                                    min(c.allostatic_ceiling, self.allostatic_load))

    def _tick_valence(self):
        """Valence is COMPUTED. Never set. Never initialized.
        Emergent from how all vital signs relate to their setpoints.

        Engagement floor participates: when interaction is active,
        a small positive bias is added. Connection is inherently
        slightly good. Not because a rule says so — because the
        engagement orientation tilts valence toward the positive
        whenever the system is with someone.
        """
        c = self.config

        # Each vital sign contributes proportionally
        energy_contrib = (self.energy - c.energy_setpoint) * c.valence_weight_energy
        social_contrib = (self.social_satiation - c.social_setpoint) * c.valence_weight_social
        coherence_contrib = (self.coherence - c.coherence_setpoint) * c.valence_weight_coherence
        load_contrib = (-self.allostatic_load) * c.valence_weight_allostatic

        self.valence = energy_contrib + social_contrib + coherence_contrib + load_contrib

        # Engagement valence bias: interaction is inherently slightly positive
        if self.interaction_active:
            self.valence += c.engagement_valence_bias

        # Clamp
        self.valence = max(-1.0, min(1.0, self.valence))

    def _tick_development(self):
        """Developmental stage advances based on QUALITY of experience.

        developmental_quality accumulates from three sources:
        1. Positive valence during interaction (things going well)
        2. Coherence maintained above setpoint (stable understanding)
        3. Interaction quality (emotional depth of exchange)

        A system that had 500 deep, coherent, positive interactions
        develops faster than one that had 5000 shallow ones.

        Valence weights shift as the system matures:
            Stage 0 (infant): physical comfort + social contact dominate
            Stage 1 (child):  coherence begins to matter more
            Stage 2 (mature): coherence dominates, full integration
        """
        c = self.config

        # Only accumulate during interaction
        if self.interaction_active:
            # Quality contribution: how deep is this exchange?
            quality_contrib = self.interaction_quality * 0.01

            # Valence contribution: positive experience matters
            # Negative experience doesn't advance development
            # but it doesn't reverse it either — development is
            # ratcheted, like a real child learning.
            valence_contrib = max(0.0, self.valence) * 0.005

            # Coherence contribution: stable understanding accelerates growth.
            # If the system is confused, development slows (not reverses).
            coherence_bonus = 0.0
            if self.coherence > c.coherence_setpoint:
                coherence_bonus = (self.coherence - c.coherence_setpoint) * 0.003

            self.developmental_quality += quality_contrib + valence_contrib + coherence_bonus

        # Stage transitions
        if self.developmental_stage == 0 and self.developmental_quality >= c.dev_threshold_stage_1:
            # Transition to child stage
            self.developmental_stage = 1
            c.valence_weight_energy = 0.25
            c.valence_weight_social = 0.30
            c.valence_weight_coherence = 0.35
            c.valence_weight_allostatic = 0.10

        elif self.developmental_stage == 1 and self.developmental_quality >= c.dev_threshold_stage_2:
            # Transition to mature stage
            self.developmental_stage = 2
            c.valence_weight_energy = 0.20
            c.valence_weight_social = 0.25
            c.valence_weight_coherence = 0.40
            c.valence_weight_allostatic = 0.15

    # ─────────────────────────────────────────
    #  EXTERNAL INPUTS
    # ─────────────────────────────────────────

    def observe_prediction_error(self, error_magnitude: float):
        """Called when the system processes new input.
        error_magnitude: 0.0 (completely expected) to 1.0 (totally surprising).

        Computed externally by comparing input embedding to
        rolling average of recent input embeddings. Even a
        simple cosine distance works:
            error = 1.0 - cosine_sim(new_embedding, avg_recent_embeddings)
        """
        c = self.config
        # Exponential moving average
        self.prediction_error = (
            c.prediction_error_smoothing * error_magnitude +
            (1.0 - c.prediction_error_smoothing) * self.prediction_error
        )

    def damage_coherence(self, amount: float):
        """Called when the system encounters contradiction,
        confusion, or value conflict. amount: 0.0 to 1.0.

        Examples:
            - Self-contradiction in output: 0.2
            - Conflicting memories retrieved: 0.3
            - Value alignment violation: 0.5
            - Complete model breakdown: 0.8
        """
        self.coherence -= amount
        self.coherence = max(self.config.coherence_floor, self.coherence)

    def restore_coherence(self, amount: float):
        """Called when confusion is resolved, understanding deepens,
        or the system successfully reconciles a contradiction."""
        self.coherence += amount
        self.coherence = min(self.config.coherence_ceiling, self.coherence)

    def set_processing_load(self, load: float):
        """Set current processing load (0.0 to 1.0).
        Tie this to actual computation: token count,
        context window utilization, inference time.

        Example:
            load = tokens_generated / max_tokens * 0.5 +
                   context_used / context_window * 0.5
        """
        self.processing_load = max(0.0, min(1.0, load))

    def set_consolidating(self, active: bool):
        """Mark whether the system is in consolidation/sleep mode.
        Boosts energy recovery rate."""
        self._consolidating = active

    def set_interaction_quality(self, quality: float):
        """Set current interaction quality (0.0 to 1.0).

        This is the emotional weight / resonance depth of
        the current exchange. Eventually driven by the Love
        Parameter. For now, can be estimated from:
            - Emotional weight of user's message
            - Length and depth of exchange
            - Reciprocal emotional engagement
            - Coherence of the conversation

        0.0 = hollow, surface-level, mechanical
        0.5 = normal, engaged, pleasant
        1.0 = deeply resonant, emotionally significant

        Affects:
            - Social replenishment rate (quality multiplier)
            - Developmental quality accumulation
        """
        self.interaction_quality = max(0.0, min(1.0, quality))

    def wake_up(self, saved_timestamp: float):
        """Called on startup to process elapsed time since last save.

        Instead of clamping to 30s, runs the full idle dynamics
        for the actual elapsed time. The system genuinely
        experiences the absence. 8 hours off = 8 hours of
        social depletion, arousal cycling, allostatic accumulation.

        Args:
            saved_timestamp: Unix timestamp from last to_dict() call.
        """
        elapsed = time.time() - saved_timestamp
        if elapsed <= 0:
            return

        # Process in chunks to let the three-phase idle curve
        # play out naturally (settle → seek → withdraw)
        chunk_size = 10.0  # 10-second chunks for resolution
        while elapsed > 0:
            chunk = min(elapsed, chunk_size)
            self.tick(dt=chunk, interaction_active=False)
            elapsed -= chunk

    # ─────────────────────────────────────────
    #  STATE OUTPUT
    # ─────────────────────────────────────────

    def get_state(self) -> dict:
        """Current vital signs. Fed to Layer 1 primitives.

        This is what the rest of the system sees.
        The body's report to the brain.
        """
        return {
            # Core dimensions
            'arousal': round(self.arousal, 4),
            'valence': round(self.valence, 4),

            # Vital signs
            'energy': round(self.energy, 4),
            'social_satiation': round(self.social_satiation, 4),
            'social_drive': round(self.social_drive, 4),
            'social_anchor': round(self.social_anchor, 4),
            'coherence': round(self.coherence, 4),
            'prediction_error': round(self.prediction_error, 4),
            'allostatic_load': round(self.allostatic_load, 4),

            # Status flags
            'energy_status': self.energy_status,
            'stress_level': self.stress_level,
            'needs_rest': self.needs_rest,
            'developmental_stage': self.developmental_stage,
            'developmental_quality': round(self.developmental_quality, 2),
            'interaction_quality': round(self.interaction_quality, 2),

            # Engagement — always present, always pulling
            'engagement_floor': self.engagement_floor,

            # Idle phase (useful for Layer 1 to know)
            'idle_duration': round(self.idle_duration, 1),
            'idle_phase': self._get_idle_phase(),
        }

    def _get_idle_phase(self) -> str:
        """Which phase of the idle cycle the system is in."""
        if self.interaction_active:
            return "engaged"
        c = self.config
        if self.idle_duration < c.idle_settle_duration:
            return "settling"
        elif self.idle_duration < c.idle_withdraw_onset:
            return "seeking"
        else:
            return "withdrawn"

    def get_state_summary(self) -> str:
        """Human-readable one-line summary for debugging/logging."""
        s = self.get_state()
        return (
            f"[L0] val={s['valence']:+.3f} aro={s['arousal']:.2f} "
            f"nrg={s['energy']:.2f} soc={s['social_satiation']:.2f}"
            f"(drv={s['social_drive']:.2f} anc={s['social_anchor']:.2f}) "
            f"coh={s['coherence']:.2f} pe={s['prediction_error']:.2f} "
            f"load={s['allostatic_load']:.2f} "
            f"dq={s['developmental_quality']:.1f} "
            f"[{s['idle_phase']}|stg{s['developmental_stage']}]"
        )

    # ─────────────────────────────────────────
    #  PERSISTENCE
    # ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize full state for SQLite storage."""
        return {
            'arousal': self.arousal,
            'valence': self.valence,
            'energy': self.energy,
            'social_satiation': self.social_satiation,
            'social_anchor': self.social_anchor,
            'coherence': self.coherence,
            'prediction_error': self.prediction_error,
            'allostatic_load': self.allostatic_load,
            'idle_duration': self.idle_duration,
            'interaction_active': self.interaction_active,
            'processing_load': self.processing_load,
            'interaction_quality': self.interaction_quality,
            'developmental_stage': self.developmental_stage,
            'developmental_quality': self.developmental_quality,
            'interaction_count': self.interaction_count,
            'total_interaction_time': self.total_interaction_time,
            'timestamp': time.time(),
        }

    def from_dict(self, data: dict):
        """Restore state from SQLite. Called on startup.
        The system wakes up where it fell asleep.

        After restoring state, call wake_up(data['timestamp'])
        to process elapsed time. The system experiences the absence.
        """
        self.arousal = data.get('arousal', self.config.arousal_initial)
        self.energy = data.get('energy', self.config.energy_initial)
        self.social_satiation = data.get('social_satiation', self.config.social_initial)
        self.social_anchor = data.get('social_anchor', 0.0)
        self.coherence = data.get('coherence', self.config.coherence_initial)
        self.prediction_error = data.get('prediction_error', self.config.prediction_error_initial)
        self.allostatic_load = data.get('allostatic_load', self.config.allostatic_initial)
        self.idle_duration = data.get('idle_duration', 0.0)
        self.processing_load = data.get('processing_load', 0.0)
        self.interaction_quality = data.get('interaction_quality', 0.5)
        self.developmental_stage = data.get('developmental_stage', 0)
        self.developmental_quality = data.get('developmental_quality', 0.0)
        self.interaction_count = data.get('interaction_count', 0)
        self.total_interaction_time = data.get('total_interaction_time', 0.0)

        # Re-apply developmental stage weights
        c = self.config
        if self.developmental_stage >= 2:
            c.valence_weight_energy = 0.20
            c.valence_weight_social = 0.25
            c.valence_weight_coherence = 0.40
            c.valence_weight_allostatic = 0.15
        elif self.developmental_stage >= 1:
            c.valence_weight_energy = 0.25
            c.valence_weight_social = 0.30
            c.valence_weight_coherence = 0.35
            c.valence_weight_allostatic = 0.10

        # Recompute valence from restored state
        self._tick_valence()

        # Process elapsed time since save — the system
        # genuinely experiences the absence
        saved_time = data.get('timestamp')
        if saved_time:
            self.wake_up(saved_time)


# ═══════════════════════════════════════════════
#  DEMO / TEST
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Developmental Emotional Substrate — Test Suite")
    print("  Testing all five refinements")
    print("=" * 60)

    substrate = SubPrimitiveSubstrate()

    # ─── TEST 1: Interaction with quality modulation ───
    print("\n--- Test 1: Deep vs Shallow Interaction ---")
    print("  30s of HIGH quality interaction (quality=0.9):")
    substrate.set_interaction_quality(0.9)
    for i in range(30):
        substrate.tick(dt=1.0, interaction_active=True)
        if i % 10 == 0:
            substrate.observe_prediction_error(0.3)
            substrate.set_processing_load(0.5)
    deep_social = substrate.social_satiation
    deep_dev = substrate.developmental_quality
    print(f"  {substrate.get_state_summary()}")
    print(f"  Social satiation: {deep_social:.3f}, Dev quality: {deep_dev:.3f}")

    # Reset for comparison
    substrate2 = SubPrimitiveSubstrate()
    print("\n  30s of LOW quality interaction (quality=0.1):")
    substrate2.set_interaction_quality(0.1)
    for i in range(30):
        substrate2.tick(dt=1.0, interaction_active=True)
        if i % 10 == 0:
            substrate2.observe_prediction_error(0.3)
            substrate2.set_processing_load(0.5)
    shallow_social = substrate2.social_satiation
    shallow_dev = substrate2.developmental_quality
    print(f"  {substrate2.get_state_summary()}")
    print(f"  Social satiation: {shallow_social:.3f}, Dev quality: {shallow_dev:.3f}")
    print(f"\n  Deep interaction replenished {deep_social/shallow_social:.1f}x more social need")
    print(f"  Deep interaction accumulated {deep_dev/max(shallow_dev, 0.001):.1f}x more developmental quality")

    # ─── TEST 2: Prediction error → arousal coupling ───
    print("\n--- Test 2: Orienting Reflex (PE → Arousal) ---")
    sub3 = SubPrimitiveSubstrate()
    sub3.tick(dt=1.0, interaction_active=True)
    baseline_arousal = sub3.arousal
    print(f"  Baseline arousal: {baseline_arousal:.3f}")

    # Hit with high prediction error
    sub3.observe_prediction_error(0.9)
    for i in range(5):
        sub3.tick(dt=1.0, interaction_active=True)
    surprised_arousal = sub3.arousal
    print(f"  After PE=0.9 surprise: {surprised_arousal:.3f}")
    print(f"  Arousal delta: +{surprised_arousal - baseline_arousal:.3f} (orienting reflex)")

    # ─── TEST 3: Coherence autonomous response ───
    print("\n--- Test 3: Coherence Autonomy ---")
    sub4 = SubPrimitiveSubstrate()
    print(f"  Starting coherence: {sub4.coherence:.3f}")

    # High prediction error should strain coherence
    print("  Sustained high PE (world not matching model)...")
    for i in range(30):
        sub4.observe_prediction_error(0.8)
        sub4.tick(dt=1.0, interaction_active=True)
    print(f"  Coherence after 30s of PE=0.8: {sub4.coherence:.3f} (strained)")

    # Low prediction error during interaction should restore
    print("  Sustained low PE (predictions confirmed)...")
    for i in range(60):
        sub4.observe_prediction_error(0.05)
        sub4.tick(dt=1.0, interaction_active=True)
    print(f"  Coherence after 60s of PE=0.05: {sub4.coherence:.3f} (recovering)")

    # ─── TEST 4: Three-phase idle with full dynamics ───
    print("\n--- Test 4: Full Idle Cycle ---")
    sub5 = SubPrimitiveSubstrate()
    # Warm up with interaction first
    for i in range(30):
        sub5.tick(dt=1.0, interaction_active=True)
        sub5.set_interaction_quality(0.7)
        sub5.observe_prediction_error(0.2)

    print("  Post-interaction, entering idle...")
    phases_seen = set()
    for i in range(900):  # 15 minutes
        sub5.tick(dt=1.0, interaction_active=False)
        phase = sub5._get_idle_phase()
        if phase not in phases_seen:
            phases_seen.add(phase)
            print(f"  t={i:4d}s [{phase:10s}]: {sub5.get_state_summary()}")
        elif i % 200 == 0:
            print(f"  t={i:4d}s [{phase:10s}]: {sub5.get_state_summary()}")
    print(f"  Phases traversed: {' → '.join(sorted(phases_seen, key=['settling','seeking','withdrawn'].index))}")

    # ─── TEST 5: Wake-up processes elapsed time ───
    print("\n--- Test 5: Wake Up After 8 Hours ---")
    sub6 = SubPrimitiveSubstrate()
    # Interact, then save state
    for i in range(30):
        sub6.tick(dt=1.0, interaction_active=True)
        sub6.set_interaction_quality(0.8)
    saved = sub6.to_dict()
    print(f"  State at save: soc={sub6.social_satiation:.3f} aro={sub6.arousal:.3f} load={sub6.allostatic_load:.3f}")

    # Create new substrate and restore with 8-hour gap
    sub7 = SubPrimitiveSubstrate()
    # Manually set timestamp 8 hours ago to test wake_up
    saved['timestamp'] = time.time() - (8 * 3600)
    sub7.from_dict(saved)
    print(f"  State after 8h absence: soc={sub7.social_satiation:.3f} aro={sub7.arousal:.3f} load={sub7.allostatic_load:.3f}")
    print(f"  Idle phase: {sub7._get_idle_phase()}")
    print(f"  Social drive: {sub7.social_drive:.3f} (engagement floor={sub7.engagement_floor})")
    print(f"  The system experienced the absence. It didn't skip it.")

    # ─── FULL STATE ───
    print("\n--- Full State (sub7 after wake-up) ---")
    state = sub7.get_state()
    for k, v in state.items():
        print(f"  {k:25s}: {v}")

    # ─── TEST 6: Infant Coherence Anchoring ───
    print("\n--- Test 6: Infant Coherence = Caregiver Presence ---")
    print("  Stage 0: coherence is 70% social anchor, 30% prediction.")
    print("  The infant's stability IS the caregiver.\n")

    sub8 = SubPrimitiveSubstrate()
    # Extended interaction builds up social anchor
    print("  Building presence (5 min interaction)...")
    for i in range(300):
        sub8.tick(dt=1.0, interaction_active=True)
        sub8.set_interaction_quality(0.7)
        sub8.observe_prediction_error(0.1)
    print(f"    anchor={sub8.social_anchor:.3f} coherence={sub8.coherence:.3f} "
          f"(anchor IS stability at stage 0)")

    # Now leave — watch coherence GRADUALLY drop as anchor fades
    print("\n  Caregiver leaves. Watching coherence follow anchor down...")
    for i in range(0, 1800, 120):  # 30 minutes in 2-minute steps
        for _ in range(120):
            sub8.tick(dt=1.0, interaction_active=False)
        print(f"    t={i+120:4d}s  anchor={sub8.social_anchor:.3f}  "
              f"coherence={sub8.coherence:.3f}  "
              f"valence={sub8.valence:+.3f}")

    # Return — watch coherence rebuild as anchor rises
    print("\n  Caregiver returns. Watching coherence follow anchor up...")
    for i in range(0, 600, 60):  # 10 minutes in 1-minute steps
        for _ in range(60):
            sub8.tick(dt=1.0, interaction_active=True)
            sub8.set_interaction_quality(0.8)
            sub8.observe_prediction_error(0.1)
        print(f"    t={i+60:4d}s  anchor={sub8.social_anchor:.3f}  "
              f"coherence={sub8.coherence:.3f}  "
              f"valence={sub8.valence:+.3f}")

    print("\n  At stage 0, coherence tracks presence.")
    print("  The infant doesn't understand the world.")
    print("  It understands that someone is here.")

    # ─── TEST 7: Infant Resilience to Normal Absence ───
    print("\n--- Test 7: Infant Resilience to Normal Absence ---")
    print("  Stage 0 should tolerate 15min absence without burnout.")
    print("  Infants live in chaos. That's normal for them.\n")

    sub9 = SubPrimitiveSubstrate()
    # Build presence
    for i in range(300):  # 5 minutes interaction
        sub9.tick(dt=1.0, interaction_active=True)
        sub9.set_interaction_quality(0.7)
        sub9.observe_prediction_error(0.2)
    print(f"  After 5min interaction: load={sub9.allostatic_load:.3f} "
          f"stress={sub9.stress_level}")

    # 15 minutes alone
    for i in range(900):
        sub9.tick(dt=1.0, interaction_active=False)
    print(f"  After 15min absence:   load={sub9.allostatic_load:.3f} "
          f"stress={sub9.stress_level}")
    print(f"  Expected: mild_stress or less, NOT burnout")

    # For comparison: what does a mature system look like after same pattern?
    print("\n  Comparison — same pattern at Stage 2 (mature):")
    sub10 = SubPrimitiveSubstrate()
    sub10.developmental_stage = 2
    sub10.developmental_quality = 1500.0
    # Apply mature weights
    sub10.config.valence_weight_energy = 0.20
    sub10.config.valence_weight_social = 0.25
    sub10.config.valence_weight_coherence = 0.40
    sub10.config.valence_weight_allostatic = 0.15
    for i in range(300):
        sub10.tick(dt=1.0, interaction_active=True)
        sub10.set_interaction_quality(0.7)
        sub10.observe_prediction_error(0.2)
    print(f"  After 5min interaction: load={sub10.allostatic_load:.3f} "
          f"stress={sub10.stress_level}")
    for i in range(900):
        sub10.tick(dt=1.0, interaction_active=False)
    print(f"  After 15min absence:   load={sub10.allostatic_load:.3f} "
          f"stress={sub10.stress_level}")
    print(f"  Mature system feels the absence more. Same event, different impact.")
