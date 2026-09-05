# jagoff2 fork development and comma four releases

The source checkout is `jagoff2/sunnypilot`, branch `master`. Only `panda` and
`opendbc_repo` use `jagoff2` forks and editable `c4-dev` branches. The other five
submodules (`msgq_repo`, `rednose_repo`, `teleoprtc_repo`, `tinygrad_repo`, and
`openpilot/sunnypilot/neural_network_data`) retain their upstream URLs and
detached checkouts at the commits pinned by the parent. Initializing or
updating dependencies means `git submodule update --init --recursive`, without
`--remote`: upstream branch tips may not match this version of sunnypilot.
Submodule updates can detach HEAD; switch back to the appropriate development
branch before editing, and reconcile it with the parent pin if necessary.

## Complete the GitHub fork setup

From PowerShell in the repository:

```powershell
ssh -T git@github.com
pwsh -File scripts/setup-fork-submodules.ps1
git diff -- .gitmodules
git add .gitmodules scripts/setup-fork-submodules.ps1 docs/FORK_DEVELOPMENT.md
git commit -m "Configure editable submodule forks and document C4 releases"
git push origin master
```

The script uses Windows OpenSSH authenticated as `jagoff2` for pushes and public
HTTPS for fetches and device clones. It configures only `panda` and `opendbc`.
For each existing public fork, it preserves
the upstream as `upstream`, publishes the pinned commit on `c4-dev`, then updates
that entry in `.gitmodules`. It verifies existing forks belong to the expected
GitHub fork network and never force-pushes. SSH cannot create GitHub repositories;
the script prints fork creation links for missing repositories and retains their
original clone URLs. Create these public forks as `jagoff2` through GitHub and rerun.
Rerunning is intended
for initial setup recovery with clean, parent-pinned submodules.
It does not commit or push the parent automatically.

The parent, `panda`, and `opendbc` push to `jagoff2` over SSH. The other submodules
use their original upstream remotes; no additional forks are required.

## Publish a submodule edit

For example, after editing and testing opendbc:

```sh
git -C opendbc_repo switch c4-dev
git -C opendbc_repo add <changed-files>
git -C opendbc_repo commit -m "Describe the opendbc change"
git -C opendbc_repo push origin c4-dev
git add opendbc_repo
git commit -m "Update opendbc to the tested fork commit"
git push --recurse-submodules=check origin master
```

Always publish submodule commits first, then commit and push the parent gitlink.
Parent Git push checking is enabled locally, but does not prove anonymous access
or correct `.gitmodules` URLs. Verify a fresh recursive clone before release.
Do not change gitlinks to untested upstream versions merely to make them newer.

## Prepare an installable C4 branch

Use a fresh checkout on the Linux filesystem in WSL/Linux, not this Windows
working tree. The Windows checkout was created with `core.symlinks=false`, so
tracked symlinks are ordinary files containing target paths. Copying those files
into a release can break AGNOS manifests. A Linux clone restores Git symlink and
executable modes and avoids CRLF shell scripts.

The existing `tools/release/build_stripped.sh` and `release_files.py` flatten
submodules and materialize LFS assets for installation. Do not point the device
installer at the development branch or publish a raw Windows directory copy.
The stripped source branch builds on the device; it is not a prebuilt image and
does not replace the device's base OS flashing procedure.

From a fresh Linux checkout of the published parent commit:

```sh
git clone --recurse-submodules --branch master https://github.com/jagoff2/sunnypilot.git
cd sunnypilot
git lfs install --local
git lfs pull
git submodule foreach --recursive 'git lfs pull'
git submodule status --recursive
git diff --exit-code
git lfs fsck
./tools/op.sh setup

# The release script deletes TARGET_DIR before populating it. Use a new temp dir.
release_dir=$(mktemp -d /tmp/jagoff2-c4-release.XXXXXX)
env -u BRANCH TARGET_DIR="$release_dir" tools/release/build_stripped.sh
git -C "$release_dir" submodule status
git -C "$release_dir" lfs ls-files
git -C "$release_dir" ls-files --stage | awk '$1 == "160000" { print; bad=1 } END { exit bad }'
```

Both submodule and LFS listings in the result must be empty. The existing release
script also checks GitHub's per-file size limit and splits large model files.
Build and test the result using the repository's build-release CI procedure
(`.github/workflows/tests.yaml`), then validate startup, model loading and hardware
operation on the C4 before treating the branch as a working device release.
Host checks alone cannot establish device compatibility.

Publish only the reviewed release result to a dedicated installation branch,
for example `c4-install`. For the first publication:

```sh
git -C "$release_dir" push origin HEAD:refs/heads/c4-install
```

Each stripped release has independent history. For later publications, capture
the existing installation branch SHA and use an explicit
`--force-with-lease=refs/heads/c4-install:<expected-old-sha>` when replacing it.
Retain the previous release SHA for rollback. Never force-push source `master`.
Configure the device installation process for `jagoff2/sunnypilot` and the tested
installation branch. No custom installer endpoint or C4 flash has been verified
by this setup.

## Assets and release infrastructure

The parent `.lfsconfig` downloads existing assets from sunnypilot's GitLab LFS
server and specifies an upstream SSH upload URL. Forking the GitHub repository
does not grant write access to that LFS server. Existing assets can stay there;
before changing LFS-tracked models or assets, establish writable fork-owned LFS
hosting and ensure every pinned asset is downloadable from a clean checkout.
Do not redirect the LFS download URL without migrating the existing objects.

Upstream `release.yaml` runs only for `sunnypilot/sunnypilot`. The prebuilt workflow
also depends on a self-hosted `tici` runner and deployment configuration. Those
workflows do not automatically provide a release service for this fork. The
manual stripped-source process above uses the existing release tooling without
assuming access to upstream runners, publishing credentials or signing keys.

References: [upstream release script](https://github.com/jagoff2/sunnypilot/blob/master/tools/release/build_stripped.sh),
[release file selection](https://github.com/jagoff2/sunnypilot/blob/master/tools/release/release_files.py),
[device launcher](https://github.com/jagoff2/sunnypilot/blob/master/launch_chffrplus.sh).
