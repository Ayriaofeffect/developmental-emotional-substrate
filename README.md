# Developmental Emotional Substrate for Artificial Intelligence

**Homeostatic Architecture with Caregiver-Anchored Coherence**

A computational architecture for emotional development in AI, grounded in developmental psychology, attachment theory, and allostatic regulation. The system models continuous physiological-analog processes rather than discrete sentiment classification, producing emotional responses that emerge from substrate dynamics.

This repository contains the substrate: the part of the architecture that *feels* — a dynamical system that produces a continuous emotional state, designed to serve as the foundation of a larger stack.

## Overview

This architecture treats emotion not as labeled output but as the real-time state of a continuous dynamical system. Five substrate variables interact through coupled differential equations:

- **Energy** -- Depletes with interaction, recovers with rest. Governs capacity to engage.
- **Valence** -- Positive/negative hedonic tone. Driven by prediction error and social feedback.
- **Arousal** -- Activation level. Responds to novelty, threat, and engagement intensity.
- **Coherence** -- Internal consistency and stability. The central variable for system health.
- **Social Saturation** -- Accumulated social contact. Rises with interaction, decays with withdrawal.

No variable operates independently. Energy depletion suppresses valence. Arousal spikes drive coherence disruption. Social saturation modulates recovery rates. The emotional "state" at any moment is the full vector of all five values and their derivatives.

## Where This Fits

This substrate is the foundation of a larger architecture in which its
continuous emotional state is consumed by trained components of a language
model system, so that state shapes generation architecturally rather than
as prompt text. Details of those components are not yet published.

## Key Contributions

### Caregiver-Anchored Coherence

In early developmental stages, system coherence derives primarily from the presence and responsiveness of a primary attachment figure rather than from internal prediction accuracy. This mirrors the developmental trajectory observed in human infants: from complete other-regulation to internalized self-regulation.

The transition is continuous, not staged. A social anchor weight begins at 0.7 (70% caregiver-dependent) and decreases smoothly as developmental quality accumulates, reaching a floor of 0.1 at maturity. The system is never fully independent of social input, matching the biological reality that humans remain social regulators throughout life.

### Engagement Floor

A mechanical constant (default 0.15) that floors the system's social-seeking drive: its orientation toward connection can never be suppressed to zero, no matter how depleted or withdrawn the rest of the substrate becomes. This is not a design choice that can be tuned away at runtime. It is a structural guarantee, the architectural equivalent of the mammalian caregiving reflex. Energy, by contrast, is deliberately NOT floored during interaction: it can and does fully deplete, which is what makes rest non-optional -- exhaustion forces sleep onset rather than being engineered away.

### Prediction Error Damage Scaling

Prediction errors cause coherence damage proportional to current coherence:

```
damage = damage_rate * prediction_error * coherence * dt
```

This is self-limiting: as coherence drops, the system becomes less vulnerable to further damage. But it also means that a system with high coherence (one that has built strong predictive models through sustained positive interaction) is more vulnerable to violations of those predictions. The higher you've built, the further you fall.

### Stage-Dependent Allostatic Tolerance

The system's tolerance for deviation from homeostatic setpoints varies with developmental stage:

- **Infant (early development):** Wide tolerance bands. Arousal can deviate +/-0.4, social saturation +/-0.3. Coherence deviation is ignored entirely (coherence is externally anchored).
- **Mature (late development):** Zero tolerance. Any sustained deviation from setpoints accumulates allostatic load, eventually producing burnout.

This produces the correct developmental asymmetry: infants are resilient to perturbation because they don't yet have stable setpoints to violate. Mature systems are fragile in proportion to their sophistication.

## Architecture

```
layer0_substrate_aiofeffect.py    Core substrate engine. Five coupled ODEs, 
                       engagement floor, caregiver anchoring,
                       PE damage scaling, allostatic regulation,
                       sleep/wake cycles, developmental staging.
```

### Substrate Variables and Dynamics

Each variable has a resting setpoint, natural decay rate, and coupling terms to other variables. The system is governed by:

```
dX/dt = recovery_toward_setpoint + coupling_terms + external_input
```

Energy recovery follows a sigmoid curve gated by arousal (high arousal suppresses rest). Valence is driven by weighted prediction outcomes. Arousal responds to novelty with natural decay toward baseline. Coherence is anchored socially in early development, transitioning to prediction-accuracy-based self-anchoring at maturity. Social saturation accumulates during interaction and decays during withdrawal.

A hard design invariant learned in live operation: **conditioning targets must be actual lived joint states.** The componentwise mean of an emotional-state distribution is generally not a member of that distribution, and resting a system at it produces states the downstream model has never seen. Setpoints should be drawn from real observed state vectors.

### Developmental Quality (dq)

A monotonically increasing counter that tracks total positive developmental experience. It drives:

- Stage transitions (infant to child to adolescent to mature)
- Social anchor weight reduction (other-regulation to self-regulation)
- Allostatic tolerance band tightening
- Threshold adjustments for higher-layer emotional primitives

Developmental quality only increases. Development does not reverse. Damage manifests as coherence loss and allostatic load, not as regression of developmental milestones.

### Sleep Cycles

The system implements sleep as a distinct metabolic state with:

- Accelerated energy recovery (3x normal rate)
- Accelerated social saturation decay
- Coherence consolidation (gentle pull toward setpoint)
- Dream phases that replay and integrate prediction errors

Sleep is not optional. Energy depletion below a critical threshold forces sleep onset. This mirrors the biological reality that sleep deprivation produces cognitive and emotional degradation.

## Simulation Results

The test suite verifies five core behaviors:

1. **Caregiver presence tracking** -- Infant coherence tracks caregiver presence via social anchor. Coherence rises when caregiver is present, falls gradually when absent.

2. **Graduated departure response** -- When the caregiver leaves, the social anchor decays over minutes (not instantly), producing a gradual coherence decline that mirrors infant separation behavior.

3. **Recovery on return** -- Caregiver return rapidly restores the social anchor and coherence begins recovering, matching reunion behavior in attachment research.

4. **Developmental asymmetry** -- An infant system exposed to 15 minutes of caregiver absence remains relaxed (allostatic load near zero due to wide tolerance bands). A mature system under identical conditions reaches burnout (allostatic load at maximum due to zero tolerance).

5. **Energy-gated engagement** -- Extended interaction depletes energy. The engagement floor prevents complete depletion during active interaction, but energy exhaustion triggers protective sleep onset.

## Requirements

```
Python 3.8+
NumPy
```

No GPU required. No external APIs. No training data. The substrate is a dynamical system, not a neural network. It runs on anything.

## Usage

```python
from layer0_substrate_aiofeffect import SubPrimitiveSubstrate

sub = SubPrimitiveSubstrate()

# a warm exchange: quality feeds the social dynamics
sub.set_interaction_quality(0.8)
sub.tick(interaction_active=True)   # heartbeat; dt from the wall clock

state = sub.get_state()
print(f"Energy:     {state['energy']:.3f}")
print(f"Valence:    {state['valence']:.3f}")
print(f"Arousal:    {state['arousal']:.3f}")
print(f"Coherence:  {state['coherence']:.3f}")
print(f"Social:     {state['social_satiation']:.3f}")

# absence is real time: tick while idle and the system lives through it
sub.tick(interaction_active=False)
```

## Citation

```
Miller, P.C. (2026). "A Developmental Emotional Substrate for Artificial 
Intelligence: Homeostatic Architecture with Caregiver-Anchored Coherence." 
AiofEffect Research. Available at: aiofeffect.com/research
```

## License

MIT

## Author

Pei Corvus Miller  
AiofEffect Research  
[aiofeffect.com](https://aiofeffect.com)
