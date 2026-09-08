# Preserved experimental lane-centering controller

The active controller was rolled back at the user's request after the newer
implementation regressed driving behavior. The road-confirmed controller is
`ad3e278b31a51f79c569fe5ce8545b70e44744a8`. Active files come from `7fb091d5b999a2eb464c36ffdccb3a77deeb15a0`, which has that same
controller and retains the separate USB GPU recovery and firmware fixes.

All controller work from `7d82d2ee3f217824b0c192073616584445f20466` remains available here, with original relative
paths and a `.saved` suffix. Added experimental modules, tests, and fixtures were
moved here; experimental versions of replaced files were copied here before
restoration. The suffix keeps them out of Python imports and test discovery.
`manifest.json` records the SHA-256 of every preserved file's original bytes.

The complete original Git trees and history are also preserved and pushed:

- sunnypilot: `archive/lane-centering-20260908` at `7d82d2ee3f217824b0c192073616584445f20466`
- opendbc: `archive/lane-centering-20260908` at `ffcde19b8cb158d8f89619909ccd5a40dfcff174`

To continue experimental development in a separate checkout:

```sh
git worktree add ../sunnypilot-lane-experiment -b lane-centering-development archive/lane-centering-20260908
cd ../sunnypilot-lane-experiment
git submodule update --init --recursive
```

The active opendbc pin is restored to `85baf553c8b73bd1d14aeef20b1daa500e479b8a` to match the original
steering feedback behavior. After pulling the rollback on another checkout,
run `git submodule update --init --recursive` to apply that pin.

Rollback validation: active production files and the opendbc pin match the
pre-experiment tree exactly. The restored lane integration and unchanged GPU
availability/inference suites passed 43 tests and 66 subtests. The captured c3x
torque suite passed 9 tests and 12 subtests using its existing Params-storage
bootstrap; production PID, NNLC, sigmoid mapping, and logging ran unchanged.
All 35 archived file hashes were verified in both the working tree and Git index.
Focused Python syntax/lint and Git whitespace checks passed. This rollback was
verified locally; this task did not install it on the vehicle or perform a drive.
