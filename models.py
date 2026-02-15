from enum import Enum


# === === Timeline Models === ===
class ActionIntent(int, Enum):
    # Skip
    WAIT = 0			# Do nothing for this turn

    # Combat
    ATTACK = 1			# Deal damage to target(s)
    DEFEND = 2			# Reduce damage taken from target(s)
    DODGE = 3			# Avoid damage from target(s)

    # Move
    MOVE = 4			# Basic movement (walking, ...)
    ADV_MOVE = 5		# Advanced movement (climbing, flying, running, ...)
    SNEAK = 6			# Stealthy movement (crawling, hiding, ...) to avoid detection

    # Social
    PERSUADE = 7		# Persuade an NPC or another player
    INTIMIDATE = 8		# Intimidate an NPC or another player
    ANTAGONIZE = 9		# Antagonize an NPC or another player
    DECEIVE = 10		# Deceive an NPC or another player

    # Usage
    ACTIVATE = 11       # using items/tools/powers/abilities
    USE = 12            # using consumables
    INTERACT = 13       # interacting with NPCs/environment
    INVESTIGATE = 14    # find clues, secrets, traps, ...


class Action:
    def __init__(self, actor: str, content: str):
        self.content: str = content
        self.actor: str = actor
        self.intent: ActionIntent = ActionIntent.WAIT
        self.targets: list[str] = []
        self.tools: list[str] = []
        self.skill_weights: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.requirement: int = 0
        self.outcome: int = 0
