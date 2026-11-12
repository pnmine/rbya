"""Action catalog with risk tiers.

Tier meanings used by the agent:
  low     - runs on its own
  medium  - runs, with an undo window
  high    - needs human approval
"""
from dataclasses import dataclass

LOW, MEDIUM, HIGH = "low", "medium", "high"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    tier: str
    target_kind: str      # "service", "node" or "db"
    reversible: bool
    description: str


CATALOG = {
    "restart_pod": ActionSpec("restart_pod", LOW, "service", False,
                              "Restart the pods of one service."),
    "scale_replicas": ActionSpec("scale_replicas", LOW, "service", True,
                                 "Add replicas to one service."),
    "rollback_deployment": ActionSpec("rollback_deployment", MEDIUM, "service", True,
                                      "Roll a service back to its previous version."),
    "revert_config": ActionSpec("revert_config", MEDIUM, "service", True,
                                "Restore the previous configuration of a service."),
    "kill_slow_query": ActionSpec("kill_slow_query", MEDIUM, "db", False,
                                  "Kill long-running queries on the database."),
    "restart_database": ActionSpec("restart_database", HIGH, "db", False,
                                   "Restart the database. Causes a short outage."),
    "drain_node": ActionSpec("drain_node", HIGH, "node", False,
                             "Move all pods off a node and cordon it."),
}


@dataclass(frozen=True)
class Action:
    name: str
    target: str

    @property
    def spec(self) -> ActionSpec:
        return CATALOG[self.name]

    @property
    def tier(self) -> str:
        return self.spec.tier

    def __str__(self) -> str:
        return f"{self.name}({self.target})"


def is_valid(name: str, target: str, services, nodes) -> bool:
    if name not in CATALOG:
        return False
    kind = CATALOG[name].target_kind
    if kind == "service":
        return target in services
    if kind == "node":
        return target in nodes
    if kind == "db":
        return target == "db"
    return False
