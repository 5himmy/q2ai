/*
 * ai_input.c -- Apply AI actions to client input
 *
 * Converts ai_action_t (from shared memory) into view angle changes
 * and movement commands that flow through the normal Quake 2 input pipeline.
 *
 * Called every client frame (~62Hz) but actions update at server frame rate
 * (10Hz). Yaw/pitch deltas are applied once per new action to avoid
 * accumulating 6x the intended rotation.
 */

#include "client.h"
#include "ai_bridge.h"

static int last_applied_frame = -1;

void AI_Input_Apply(const ai_action_t *action)
{
    // Only apply yaw/pitch delta once per server frame (when action changes)
    if (cl.frame.number != last_applied_frame) {
        last_applied_frame = cl.frame.number;

        cl.viewangles[YAW] += action->yaw_delta;

        // Mild pitch centering: decay 15% toward level each frame
        // Prevents random-walk from getting stuck at sky/floor
        cl.viewangles[PITCH] *= 0.85f;
        cl.viewangles[PITCH] += action->pitch_delta;

        // Clamp pitch to ±89 degrees
        float pitch = SHORT2ANGLE(cl.frame.ps.pmove.delta_angles[PITCH]);
        float angle = cl.viewangles[PITCH] + pitch;

        if (angle < -180)
            angle += 360;
        if (angle > 180)
            angle -= 360;

        angle = Q_clipf(angle, -89, 89);
        cl.viewangles[PITCH] = angle - pitch;
    }

    // Movement is fine to apply every frame (it's absolute, not a delta)
}
