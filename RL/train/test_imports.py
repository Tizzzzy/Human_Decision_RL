#!/usr/bin/env python
"""
Quick test to verify all modules import correctly.
Run from the RL/train directory: python test_imports.py
"""

print("[Testing imports]")
print("1. config...", end=" ")
import config
print("✓")

print("2. prompt_templates...", end=" ")
import prompt_templates
print("✓")

print("3. probe_model...", end=" ")
import probe_model
print("✓")

print("4. dataset...", end=" ")
import dataset
print("✓")

print("5. reward...", end=" ")
import reward
print("✓")

print("\n[✓] All imports successful!")
