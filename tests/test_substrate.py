"""Test suite for the Developmental Emotional Substrate.

Verifies the five core behaviors described in README 'Simulation Results':
  1. Caregiver presence tracking (infant coherence follows the anchor)
  2. Graduated departure response (minutes, not instant)
  3. Recovery on return
  4. Developmental asymmetry (infant relaxed / mature burns out)
  5. Energy-gated engagement (floor holds; depletion demands rest)

Run:  python3 tests/test_substrate.py   (or pytest tests/)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from layer0_substrate_aiofeffect import SubPrimitiveSubstrate


def simulate(sub, seconds, active, quality=None, step=1.0):
    if quality is not None:
        sub.set_interaction_quality(quality)
    t = 0.0
    while t < seconds:
        sub.tick(dt=step, interaction_active=active)
        t += step


def fresh_infant():
    sub = SubPrimitiveSubstrate()
    assert sub.get_state()['developmental_stage'] == 0
    return sub


def make_mature(sub=None):
    sub = sub or SubPrimitiveSubstrate()
    d = sub.to_dict()
    d['developmental_stage'] = 2
    d['developmental_quality'] = 1e6
    sub.from_dict(d)
    sub.developmental_stage = 2
    return sub


def test_1_caregiver_presence_tracking():
    sub = fresh_infant()
    simulate(sub, 300, active=True, quality=0.8)
    present = sub.get_state()
    assert present['social_anchor'] > 0.5, present['social_anchor']
    coh_present = present['coherence']
    simulate(sub, 1800, active=False)
    gone = sub.get_state()
    assert gone['social_anchor'] < present['social_anchor']
    assert gone['coherence'] < coh_present
    print(f"  1. presence tracking: anchor {present['social_anchor']:.2f}->"
          f"{gone['social_anchor']:.2f}, coherence {coh_present:.2f}->"
          f"{gone['coherence']:.2f}  OK")


def test_2_graduated_departure():
    sub = fresh_infant()
    simulate(sub, 300, active=True, quality=0.8)
    at_leave = sub.get_state()['coherence']
    simulate(sub, 5, active=False)
    after_5s = sub.get_state()
    drop_5s = at_leave - after_5s['coherence']
    assert after_5s['social_anchor'] > 0.3, "anchor must not vanish in 5s"
    simulate(sub, 900, active=False)
    after_15m = sub.get_state()
    drop_15m = at_leave - after_15m['coherence']
    assert drop_5s < 0.15, f"instant coherence crash: {drop_5s}"
    assert drop_15m > drop_5s, "decline must continue over minutes"
    print(f"  2. graduated departure: drop@5s {drop_5s:.3f} < "
          f"drop@15m {drop_15m:.3f}  OK")


def test_3_recovery_on_return():
    sub = fresh_infant()
    simulate(sub, 300, active=True, quality=0.8)
    simulate(sub, 1800, active=False)
    low = sub.get_state()
    simulate(sub, 180, active=True, quality=0.8)
    back = sub.get_state()
    assert back['social_anchor'] > low['social_anchor']
    assert back['coherence'] >= low['coherence'] - 1e-9
    print(f"  3. recovery on return: anchor {low['social_anchor']:.2f}->"
          f"{back['social_anchor']:.2f}, coherence {low['coherence']:.2f}->"
          f"{back['coherence']:.2f}  OK")


def test_4_developmental_asymmetry():
    infant = fresh_infant()
    simulate(infant, 300, active=True, quality=0.8)
    simulate(infant, 900, active=False)
    infant_load = infant.get_state()['allostatic_load']
    mature = make_mature()
    simulate(mature, 300, active=True, quality=0.8)
    simulate(mature, 900, active=False)
    mature_load = mature.get_state()['allostatic_load']
    assert infant_load < 0.2, f"infant should stay relaxed: {infant_load}"
    assert mature_load > infant_load, (
        f"mature ({mature_load}) must exceed infant ({infant_load})")
    print(f"  4. asymmetry: infant load {infant_load:.3f} < "
          f"mature load {mature_load:.3f}  OK")


def test_5_energy_gated_engagement():
    sub = fresh_infant()
    floor_broken, needed_rest = 1.0, False
    sub.set_interaction_quality(0.8)
    for _ in range(4 * 3600):
        sub.tick(dt=1.0, interaction_active=True)
        st = sub.get_state()
        floor_broken = min(floor_broken, st['energy'])
        if st['needs_rest']:
            needed_rest = True
    assert floor_broken >= 0.0, "energy went negative"
    assert needed_rest or floor_broken < 0.35, (
        "4h of interaction should deplete toward the rest threshold")
    print(f"  5. energy gating: min energy {floor_broken:.3f}, "
          f"needs_rest seen: {needed_rest}  OK")


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    print(f"substrate test suite — {len(fns)} behaviors")
    for fn in fns:
        fn()
    print("ALL PASS")
