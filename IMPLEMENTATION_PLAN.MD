# IMPLEMENTATION_PLAN.MD
**Project:** Newton-based surgical interaction sandbox (Isaac Lab `dev/newton`)  
**Branch:** `huvers-newton` (work branch)  
**Upstream mirror:** `upstream-newton` (tracks `isaac-sim/IsaacLab:dev/newton`)  
**Target behavior:** **B now** = grasp + lift + regrasp on rigid proxies; **C later** = cut/tear/slip with deformables.

---

## 0) Branch & workflow rules

### 0.1 Keep `upstream-newton` pristine
- `upstream-newton` is a clean mirror of upstream (`isaac-sim/IsaacLab:dev/newton`).
- All feature work goes into `huvers-newton`.

### 0.2 Expect churn
Newton integration is explicitly experimental and may break APIs and behaviors between updates. Plan for frequent rebases.

### 0.3 Newton capability assumptions (today)
Build assuming reliable support for:
- Articulations
- ContactSensor
- single-agent workflows
- visualization workflows

Do **not** assume deformable soft bodies are available/stable.

### 0.4 Quaternion convention
Newton Beta changes pose conventions to **xyzw** quaternions. Centralize transforms + pose error math in one module and use it everywhere.

---

## 1) Repository layout (additions in this branch)

Create a dedicated “huvers surg” task package so your work is isolated from upstream churn.

### 1.1 New directories
- `source/isaaclab_tasks/huvers_surg/`
  - `__init__.py`
  - `assets/`
  - `envs/`
  - `controllers/`
  - `interfaces/`
  - `utils/`
- `scripts/huvers_surg/`
  - `run_touch_proxy.py`
  - `run_grasp_demo.py`
  - `run_slip_test.py`

### 1.2 Naming conventions
- Everything under `huvers_surg` should import from Isaac Lab, not modify Isaac Lab core.
- Only patch Isaac Lab core if required for Newton bugs or missing hooks.

---

## 2) Milestone A — Vertical slice: robot touches rigid organ proxy (no grasp)

### Goal
A minimal script that:
1) spawns dVRK (or your surgical robot articulation)  
2) spawns a rigid organ proxy (liver/gallbladder)  
3) drives the gripper into contact  
4) confirms stable contact in Newton.

### Deliverables
- `scripts/huvers_surg/run_touch_proxy.py`
- `envs/touch_proxy_env.py` (Direct-style env)
- basic `SimulationCfg` using Newton physics config

### Acceptance criteria
- Robot moves deterministically.
- Organ proxy stays stable and doesn’t explode.
- Contact is observable (even if not yet sensed via ContactSensor).

---

## 3) Milestone B — Contact sensing on jaw links

### Goal
Add two ContactSensors:
- left jaw link
- right jaw link

Expose a clean “contact features” API for the grasp controller.

### Implementation
- Configure `ContactSensor` per jaw prim path (or regex).
- Add filtering (if supported) to isolate contacts from the organ proxy.

### Deliverables
- `controllers/contact_features.py`
  - `force_mag`
  - `in_contact` boolean (threshold + hysteresis)
  - `contact_persistence` (debounce window)
- `utils/warp_bridge.py`
  - `wp_to_torch()` helper (only if you need torch)

### Acceptance criteria
- You can print (or log) left/right jaw force magnitudes per step.
- `in_contact_left/right` toggles reliably when approaching/leaving the organ.

---

## 4) Milestone C — Grasp state machine (behavior B, logic-only)

### Goal
Implement the “B” user story:
- **grasp** (detect closure + bilateral contact)
- **lift**
- **release**
- **regrasp**

### State machine
Implement `controllers/grasp_fsm.py`:

States:
- `OPEN`
- `CLOSING`
- `CANDIDATE_GRASP`
- `HELD`
- `OPENING`
- `FAILED` (optional)

Transitions (MVP):
- `OPEN` → `CLOSING` on close command
- `CLOSING` → `CANDIDATE_GRASP` when:
  - left jaw in contact AND right jaw in contact
  - jaw gap < threshold
- `CANDIDATE_GRASP` → `HELD` when “constraint engaged”
- `HELD` → `OPENING` on open command OR drop detected
- `OPENING` → `OPEN` when jaw gap > open threshold

### Drop/slip detection (initial)
Before solver tuning, implement simple detection:
- relative pose drift between object and gripper exceeds threshold for N steps, OR
- jaw forces collapse while jaw is closed.

### Deliverables
- `controllers/grasp_fsm.py`
- `scripts/huvers_surg/run_grasp_demo.py` (scripted demo)

### Acceptance criteria
- FSM transitions deterministically.
- Grasp attempt succeeds at least in a controlled placement scenario.
- Release + regrasp repeats.

---

## 5) Milestone D — Grasp “constraint” implementation (core of B)

This milestone should be implemented as a spike + robust fallback.

### D0 Spike: runtime fixed/weld constraint
Write a tiny test to see if Newton backend supports creating a fixed constraint at runtime between:
- organ proxy rigid body
- gripper grasp frame / jaw body

**Outcome options:**
- If clean + stable: use it.
- If not supported / unstable: use fallback D1.

### D1 Fallback: servo attachment (always works)
Implement a control-level “attachment” in `controllers/grasp_constraint.py`:

Two variants:
1) **Kinematic attach (fastest MVP)**
   - In HELD: set object pose = grasp frame pose each step.
2) **Impedance/servo attach (preferred)**
   - Compute pose error `Δx, Δq`.
   - Apply wrench (force/torque) toward zero with damping.

### Grasp anchor definition (MVP)
Because ContactSensor gives net forces (not contact patch location), define anchor as:
- frame midpoint between jaw tips (or a designated gripper frame)
- store `T_grip^-1 * T_obj` at engage time
- enforce that relative transform during HELD

### Deliverables
- `controllers/grasp_constraint.py`
- `controllers/grasp_anchor.py` (jaw tip midpoint anchor helper)

### Acceptance criteria
- Object can be lifted while held.
- Regrasp is possible with repeated cycles.
- No obvious “teleport” instability on engage (may need smoothing).

---

## 6) Milestone E — Slip control and Newton solver tuning harness

### Goal
Make pinch grasp less slippery and more repeatable by:
- building a measurable slip test
- tuning Newton/MuJoCo-Warp solver knobs

### What to tune (plan)
- friction cone: `cone` (`pyramidal` vs `elliptic`)
- friction emphasis: `impratio`
- solver iterations / line search: `ls_iterations`, `ls_parallel`
- constraint buffer sizing: `nefc_per_env`

### Test harness
Create `scripts/huvers_surg/run_slip_test.py`:

Procedure:
1) place object between jaws
2) close to candidate grasp
3) engage constraint
4) apply vertical lift trajectory
5) measure slip:
   - relative displacement over time
   - drop events
   - max drift vs threshold

Add a parameter sweep runner:
- sweep `impratio` (e.g., 1, 3, 5, 10)
- compare `cone` modes
- adjust iterations if needed

### Acceptance criteria
- You can reproduce slip curves across runs.
- You can demonstrate at least one setting that materially reduces slip.

---

## 7) Milestone F — 3DGS visual integration (without blocking B)

### Goal
Integrate Gaussian Splat scene as a **visual-only** child, while physics uses a rigid proxy.

### USD scene structure
Represent organs like:

`/World/Organs/Gallbladder` (Xform parent)
- `/Visual/GaussianSplat` (render-only)
- `/Physics/ProxyMesh` (collision-only)

Simulation interacts with `/Physics/ProxyMesh`.
Visual prim follows parent transform (or is a child under same Xform).

### Deliverables
- `assets/organ_asset.py` (spawns parent Xform + children)
- optional stubs for a GS renderer plugin/extension integration

### Acceptance criteria
- Physics continues to work unchanged.
- Visual + physics remain aligned by shared parent transform.
- Replacing visual implementation later does not affect controllers/env logic.

---

## 8) Milestone G — “TissueInteractable” interface (future-proofing for C)

### Goal
Introduce a domain abstraction so later deformable/cutting work does not rewrite grasp logic.

### Interface
Create `interfaces/tissue_interactable.py`:

Methods:
- `get_physics_proxy_prim() -> str`
- `get_visual_prim() -> str`
- `compute_grasp_anchor(gripper_state) -> Anchor`
- `apply_grasp_constraint(anchor, gripper_state) -> None`
- `release_grasp() -> None`
- `step_visual_update() -> None`

### Implementations
- `RigidTissueProxy(TissueInteractable)` (now)
- `DeformableTissueProxy(TissueInteractable)` (later; stub)

### Acceptance criteria
- Env/controllers depend only on `TissueInteractable`, not organ specifics.
- Swapping to a new proxy implementation only changes `assets/` + `interfaces/`.

---

## 9) Milestone H — C later: cutting/tearing roadmap hooks

### Goal
Add hooks now so C later isn’t a refactor.

Add stubs (no-op for now):
- `request_cut(path, tool_pose) -> CutResult`
- `update_collision_proxy()`
- `update_visual_gs()` / `update_visual_mapping()`

Future path:
1) introduce deformable proxy (Warp-native FEM or custom method) while Newton remains the articulation backend
2) couple contact forces across interface boundary
3) implement topology changes + collision/visual updates
4) advect gaussians via deformation field; split/remove on cut

---

## 10) Commit plan (suggested)

1. `feat(huvers_surg): scaffold package + scripts`
2. `feat(sim): newton sim cfg + spawn robot + rigid proxy`
3. `feat(sensing): jaw contact sensors + contact_features`
4. `feat(grasp): grasp FSM + candidate detection`
5. `feat(grasp): constraint spike + servo fallback attach`
6. `feat(eval): slip test harness + param sweeps`
7. `feat(visual): organ USD structure (visual + proxy)`
8. `refactor: TissueInteractable interface + rigid impl`
9. `chore: regression scripts + smoke checks`

---

## 11) Critical “do not skip” utilities

### 11.1 Pose / quaternion utilities
Create `utils/pose.py` and route all transforms through it:
- conversions
- pose composition
- pose error
- quaternion convention enforcement (**xyzw**)

### 11.2 Logging and determinism
- seed management (if using RL later)
- log solver config and slip metrics to a structured file per run

---

## 12) Definition of Done (B)

B is “done” when:
- dVRK can approach organ proxy, close jaws, detect bilateral contact, engage grasp, lift, release, and regrasp
- behavior is stable across repeated runs
- slip can be quantified and improved through solver/friction tuning
- 3DGS visual can be present without affecting physics or controllers
- interface layer exists so deformables/cutting can be added later without rewriting grasp logic

---
