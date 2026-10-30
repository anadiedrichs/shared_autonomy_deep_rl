# Project roadmap: Shared autonomy deep RL

This document outlines the development plan, current priorities, and future milestones for the shared autonomy quadcopter project.

## 🔴 Current focus: Stabilization & core fixes (next days)
*The immediate priority is ensuring a stable baseline for the RLtools training loop and Webots simulation.*
- [ ] **Critical:** Fix copilot  (Issue #33).
- [ ] Train copilot in scenario 1.
- [ ] Test the copilot with the noisy and lagged pilots.
- [ ] Evaluate model performance against baseline heuristics.
- [ ] Standardize logging outputs for evaluation metrics, review results (issue #8)
- [ ] Save the files related with the submited results (.csv, logs, models) in a results branch (e.g 2026-10-dd-results )

## 🟡 Short-term goals: Enhancements & minor issues (Q4 2026)
*Addressing technical debt and improving the teleoperation interface.*
- [ ] Resolve minor issues
- [ ] Code cleanup and dependency updates.
- [ ] Improve documentation for the local setup process.
- [ ] Document translation Eng-Spa & Spa-Eng (branch feature/docs-translation, issue #20)
## 🟢 Medium to long-term goals: Advanced autonomy features
*Expanding the capabilities of the deep reinforcement learning agent.*
- [ ] Implement advanced shared-control blending functions.
- [ ] Expand simulation environments in Webots with dynamic obstacles.(issue #12)
- [ ] Domain randomnization
- [ ] Bayesian Hyperparameter tunning
