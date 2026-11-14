# S54-SuperMimic-Flat-6000

## Model Overview

A reference-conditioned humanoid motion tracking policy
for the KUAVO S54 robot.

## Framework

- Algorithm: SuperMimic (World Model + BPTT)
- Robot: KUAVO S54, 29 DoF
- Simulator: Isaac Lab 3.0 + Newton/MJWarp
- Training iterations: 6000
- Motion data: TCRS-generated flat-ground references

## Supported Motions

- Forward walking
- Turning
- In-place rotation

## Evaluation

Fill in measured training and evaluation results.

## Limitations

- Evaluated in simulation
- Generalization to unseen motion parameters is limited
- Stair locomotion has not been established by this checkpoint

## Checkpoint Usage

Provide the matching code revision, loader,
observation contract and example inference command.