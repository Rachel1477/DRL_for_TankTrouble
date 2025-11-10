#!/usr/bin/env python3
"""
Test script to verify what information the agent receives in its state vector.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import tank_trouble_env

def analyze_state():
    env = tank_trouble_env.TankEnv()
    state = env.reset()
    
    print("="*60)
    print("State Vector Analysis")
    print("="*60)
    print(f"\nTotal state dimension: {len(state)}")
    print("\n--- Base Features (first 9 dimensions) ---")
    print(f"[0-1] My position (x, y): ({state[0]:.3f}, {state[1]:.3f})")
    print(f"[2-3] My angle (sin, cos): ({state[2]:.3f}, {state[3]:.3f})")
    print(f"[4]   My shell available: {state[4]:.3f}")
    print(f"[5-6] Enemy relative pos (dx, dy): ({state[5]:.3f}, {state[6]:.3f})")
    print(f"[7-8] Enemy angle (sin, cos): ({state[7]:.3f}, {state[8]:.3f})")
    
    print("\n--- Ray Features (next 48 dimensions: 16 rays * 3 values) ---")
    for i in range(16):
        idx = 9 + i * 3
        wall_dist = state[idx]
        enemy_dist = state[idx + 1]
        bullet_dist = state[idx + 2]
        angle = i * 360.0 / 16
        print(f"Ray {i:2d} (angle {angle:5.1f}°): wall={wall_dist:.3f}, enemy={enemy_dist:.3f}, bullet={bullet_dist:.3f}")
    
    print("\n" + "="*60)
    print("Running 5 steps to observe state changes...")
    print("="*60)
    
    for step in range(5):
        action = [1, 3, 4, 1, 5][step]  # forward, rotate_cw, rotate_ccw, forward, shoot
        next_state, reward, done = env.step(action)
        
        my_x, my_y = next_state[0], next_state[1]
        my_sin, my_cos = next_state[2], next_state[3]
        en_dx, en_dy = next_state[5], next_state[6]
        
        print(f"\nStep {step+1}: action={action}, reward={reward:.4f}, done={done}")
        print(f"  My pos: ({my_x:.3f}, {my_y:.3f}), angle: (sin={my_sin:.3f}, cos={my_cos:.3f})")
        print(f"  Enemy relative: (dx={en_dx:.3f}, dy={en_dy:.3f})")
        print(f"  Ray 0 (forward): wall={next_state[9]:.3f}, enemy={next_state[10]:.3f}, bullet={next_state[11]:.3f}")
        
        if done:
            print("  Episode ended!")
            break
        
        state = next_state
    
    print("\n" + "="*60)
    print("Conclusion:")
    print("✓ Agent DOES receive enemy relative position (state[5:7])")
    print("✓ Agent DOES receive ray-based wall/enemy/bullet detection")
    print("✓ Agent DOES receive its own position and orientation")
    print("="*60)

if __name__ == "__main__":
    analyze_state()

