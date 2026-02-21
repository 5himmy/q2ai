#!/usr/bin/env python3
"""Quick test: read shared memory from a running q2ai client and print observations."""

import ctypes
import mmap
import posix_ipc
import struct
import time
import sys

Q2AI_MAX_ENTITIES = 64
MAX_ITEMS = 256

class AiEntityObs(ctypes.Structure):
    _fields_ = [
        ("origin", ctypes.c_float * 3),
        ("angles", ctypes.c_float * 3),
        ("modelindex", ctypes.c_int32),
        ("effects", ctypes.c_int32),
        ("solid", ctypes.c_int32),
        ("frame", ctypes.c_int32),
        ("skinnum", ctypes.c_int32),
        ("active", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
    ]

class AiObservation(ctypes.Structure):
    _fields_ = [
        ("origin", ctypes.c_float * 3),
        ("velocity", ctypes.c_float * 3),
        ("viewangles", ctypes.c_float * 3),
        ("health", ctypes.c_int32),
        ("armor", ctypes.c_int32),
        ("ammo", ctypes.c_int32),
        ("current_weapon", ctypes.c_int32),
        ("frags", ctypes.c_int32),
        ("pm_type", ctypes.c_int32),
        ("pm_flags", ctypes.c_int32),
        ("inventory", ctypes.c_int32 * MAX_ITEMS),
        ("num_entities", ctypes.c_int32),
        ("entities", AiEntityObs * Q2AI_MAX_ENTITIES),
        ("mapname", ctypes.c_char * 64),
        ("serverframe", ctypes.c_int32),
        ("servertime", ctypes.c_int32),
        ("client_num", ctypes.c_int32),
    ]

class AiAction(ctypes.Structure):
    _fields_ = [
        ("yaw_delta", ctypes.c_float),
        ("pitch_delta", ctypes.c_float),
        ("forwardmove", ctypes.c_float),
        ("sidemove", ctypes.c_float),
        ("jump", ctypes.c_uint8),
        ("crouch", ctypes.c_uint8),
        ("attack", ctypes.c_uint8),
        ("weapon", ctypes.c_uint8),
    ]

class AiSharedState(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint32),
        ("episode_done", ctypes.c_uint8),
        ("reset_request", ctypes.c_uint8),
        ("connected", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8),
        ("reward", ctypes.c_float),
        ("cumulative_reward", ctypes.c_float),
        ("obs", AiObservation),
        ("action", AiAction),
    ]

def main():
    instance = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    shm_name = f"/q2ai_state_{instance}"
    sem_obs_name = f"/q2ai_obs_{instance}"
    sem_act_name = f"/q2ai_act_{instance}"

    print(f"Connecting to shared memory: {shm_name}")
    print(f"Struct size: {ctypes.sizeof(AiSharedState)} bytes")

    try:
        shm = posix_ipc.SharedMemory(shm_name)
    except posix_ipc.ExistentialError:
        print(f"ERROR: Shared memory {shm_name} not found. Is q2ai running?")
        sys.exit(1)

    buf = mmap.mmap(shm.fd, ctypes.sizeof(AiSharedState))
    shm.close_fd()

    try:
        sem_obs = posix_ipc.Semaphore(sem_obs_name)
        sem_act = posix_ipc.Semaphore(sem_act_name)
    except posix_ipc.ExistentialError:
        print("ERROR: Semaphores not found. Is q2ai running?")
        sys.exit(1)

    print("Connected! Reading frames...\n")

    for i in range(20):
        # Wait for observation from C client
        try:
            sem_obs.acquire(timeout=2)
        except posix_ipc.BusyError:
            print("Timeout waiting for observation")
            continue

        # Read state from shared memory
        state = AiSharedState.from_buffer_copy(buf)

        print(f"--- Frame {state.sequence} ---")
        print(f"  Connected: {state.connected}  Episode done: {state.episode_done}")
        print(f"  Reward: {state.reward:.3f}  Cumulative: {state.cumulative_reward:.3f}")
        print(f"  Map: {state.obs.mapname.decode('utf-8', errors='ignore').rstrip(chr(0))}")
        print(f"  Origin: ({state.obs.origin[0]:.1f}, {state.obs.origin[1]:.1f}, {state.obs.origin[2]:.1f})")
        print(f"  Velocity: ({state.obs.velocity[0]:.1f}, {state.obs.velocity[1]:.1f}, {state.obs.velocity[2]:.1f})")
        print(f"  View angles: ({state.obs.viewangles[0]:.1f}, {state.obs.viewangles[1]:.1f}, {state.obs.viewangles[2]:.1f})")
        print(f"  Health: {state.obs.health}  Armor: {state.obs.armor}  Ammo: {state.obs.ammo}")
        print(f"  Weapon: {state.obs.current_weapon}  Frags: {state.obs.frags}")
        print(f"  PM type: {state.obs.pm_type}  PM flags: {state.obs.pm_flags}")
        print(f"  Entities visible: {state.obs.num_entities}")
        print(f"  Server frame: {state.obs.serverframe}  Time: {state.obs.servertime}")
        print(f"  Client num: {state.obs.client_num}")

        for j in range(min(state.obs.num_entities, 5)):
            e = state.obs.entities[j]
            if e.active:
                print(f"    Entity {j}: origin=({e.origin[0]:.1f}, {e.origin[1]:.1f}, {e.origin[2]:.1f}) model={e.modelindex}")

        # Send a no-op action (just let the semaphore through)
        sem_act.release()
        print()

    buf.close()
    print("Done!")

if __name__ == "__main__":
    main()
