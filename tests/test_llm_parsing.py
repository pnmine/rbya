import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from rbya.reasoners import LLMReasoner, _confidence, _extract_json
from rbya.sim import Fault, Simulator

REPLY = ('<think>The {loudest} alert is frontend but traces point at payment.</think>\n'
         '```json\n{"diagnosis": "bad deploy on payment", "candidates": ['
         '{"action": "rollback_deployment", "target": "payment", "confidence": "85%", "rationale": "new version"},'
         '{"action": "reboot_everything", "target": "all", "confidence": 0.2}]}\n```')


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": REPLY}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestParsing(unittest.TestCase):
    def test_extract_skips_think_and_fences(self):
        data = _extract_json(REPLY)
        self.assertEqual(data["candidates"][0]["target"], "payment")

    def test_confidence(self):
        self.assertAlmostEqual(_confidence("85%"), 0.85)
        self.assertAlmostEqual(_confidence(85), 0.85)
        self.assertAlmostEqual(_confidence("0.7"), 0.7)
        self.assertEqual(_confidence("high"), 0.5)

    def test_round_trip_with_mock_server(self):
        srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        r = LLMReasoner(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="mock", api_key="x")
        sim = Simulator(Fault("shifting", "payment", start=1), seed=0)
        sim.step()
        prop = r.propose(sim.step())
        srv.shutdown()
        self.assertEqual([str(c.action) for c in prop.candidates], ["rollback_deployment(payment)"])
        self.assertAlmostEqual(prop.candidates[0].confidence, 0.85)
        self.assertEqual(r.api_calls, 1)


if __name__ == "__main__":
    unittest.main()


class RepairHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RepairHandler.calls += 1
        if len(req["messages"]) == 2:
            content, finish = "<think>Let me think about the alerts carefully...", "length"
        else:
            content, finish = '{"diagnosis": "d", "candidates": [{"action": "scale_replicas", "target": "cart", "confidence": 0.7}]}', "stop"
        body = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestRepair(unittest.TestCase):
    def test_repair_turn_recovers_json(self):
        srv = HTTPServer(("127.0.0.1", 0), RepairHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        r = LLMReasoner(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="mock", api_key="x")
        sim = Simulator(Fault("traffic_spike", "cart", start=1), seed=0)
        sim.step()
        prop = r.propose(sim.step())
        srv.shutdown()
        self.assertEqual([str(c.action) for c in prop.candidates], ["scale_replicas(cart)"])
        self.assertEqual((r.repairs, r.truncated, r.api_calls), (1, 1, 2))
