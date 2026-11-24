"""The judge's starVLA adapter for a framework other than the judge's recipe (see stat_gate.py).

The adapter itself is unchanged. It reads the list of trainable parameters it records from the file named in the
judge's settings, which is the headline recipe's; here that one setting points to a list for the framework under
test (STAT_TRAINABLE), which the adapter writes from the reference tree on its first run.
"""
import os

from bench import starvla_adapter as judge

judge.CFG = dict(judge.CFG, numerics=dict(judge.CFG['numerics'], trainable=os.environ['STAT_TRAINABLE']))
Adapter = judge.StarVLAAdapter
