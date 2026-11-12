import random
import unittest

from rbya.actions import Action
from rbya.agents import ReActAgent, ReconsiderAgent
from rbya.human import SimulatedHuman
from rbya.reasoners import ScriptedReasoner
from rbya.sim import FAULT_KINDS, Fault, Simulator, sample_fault


class TestSimulator(unittest.TestCase):
    def test_correct_fix_resolves_every_fault(self):
        for kind in FAULT_KINDS:
            f = sample_fault(random.Random(1), kind)
            sim = Simulator(f, seed=1)
            for _ in range(3):
                sim.step()
            if f.correct_fix():
                self.assertTrue(sim.apply(f.correct_fix()), kind)
            for _ in range(6):
                sim.step()
            self.assertTrue(sim.resolved(), kind)

    def test_wrong_db_restart_causes_collateral(self):
        sim = Simulator(Fault("bad_deploy", "cart", start=1), seed=0)
        sim.step()
        sim.apply(Action("restart_database", "db"))
        self.assertGreater(sim.collateral, 0)


class TestAgents(unittest.TestCase):
    def test_all_agents_resolve_bad_deploy(self):
        for make in (lambda r: ReActAgent(r), lambda r: ReconsiderAgent(r, SimulatedHuman(0))):
            res = make(ScriptedReasoner(0, slip=0)).run(Simulator(Fault("bad_deploy", "payment", start=1), seed=0))
            self.assertTrue(res.resolved)

    def test_high_risk_goes_to_human(self):
        res = ReconsiderAgent(ScriptedReasoner(0, slip=0), SimulatedHuman(0, accuracy=1.0)).run(
            Simulator(Fault("node_pressure", "node-b", start=1), seed=0))
        self.assertEqual(res.escalations, 1)
        self.assertTrue(res.resolved)


if __name__ == "__main__":
    unittest.main()
