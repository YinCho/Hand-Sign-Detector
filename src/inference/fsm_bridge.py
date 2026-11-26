"""
Translate stable hand-sign commands into the words RoboPet's state machine reads.

The detector names hand POSES (palm, point_down, ...). Nole's fsm_node in
Project_B7 (branch nolef, src/robopet_movement/robopet_movement/fsm_node.py)
listens for ACTIONS on /hand_gesture (std_msgs/String): "stop", "come",
"spin" and "switch_owner". It treats every message as a new request, so each
gesture should be sent once, not on every frame it is held.

To change which pose triggers which action, edit GESTURE_TO_FSM only.
"""

# Pose from the trained model -> word fsm_node expects. A pose missing from
# this table (no_gesture, or None when there is no stable command) sends nothing.
GESTURE_TO_FSM = {
    "palm": "stop",
    "point_down": "come",
    "thumb_side": "spin",
    "point": "switch_owner",
}


class FsmCommandFilter:
    """Turn the pipeline's per-frame command into one message per new gesture.

    GesturePipeline keeps returning "thumb_side" on every frame while the thumb
    is held. Without this filter fsm_node would receive "spin" many times a
    second. A message is sent only when the command changes to a new pose, so
    lowering the hand (command None) and raising it again sends it again.
    """

    def __init__(self):
        self.last = None

    def update(self, command):
        """Give the pipeline's command for this frame. Returns the fsm_node word
        to publish now, or None when nothing should be sent."""
        is_new = command != self.last
        self.last = command
        if is_new and command in GESTURE_TO_FSM:
            return GESTURE_TO_FSM[command]
        return None
