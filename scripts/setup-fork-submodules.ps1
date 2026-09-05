# Configures only panda and opendbc forks. Other submodules retain upstream pins.
# Uses Windows OpenSSH to publish branches to existing public forks. Never force-pushes.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
function Git {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
  $result = & git.exe -C $repoRoot @Arguments
  if ($LASTEXITCODE -ne 0) { throw "Git failed: $($Arguments -join ' ')" }
  return $result
}

$modules = @(
  @{ Name = 'panda'; Path = 'panda'; Source = 'sunnyhaibin/panda' },
  @{ Name = 'opendbc'; Path = 'opendbc_repo'; Source = 'sunnypilot/opendbc' }
)

if ((Git remote get-url origin) -notmatch '^https://github\.com/jagoff2/sunnypilot(?:\.git)?$') {
  throw 'Expected origin to be jagoff2/sunnypilot.'
}

$headers = @{ 'User-Agent' = 'sunnypilot-fork-setup' }
function Api {
  param([string]$Path, [string]$Method = 'Get', [string]$Body)
  $request = @{ Uri = "https://api.github.com/$Path"; Headers = $headers; Method = $Method }
  if ($Body) { $request.Body = $Body; $request.ContentType = 'application/json' }
  Invoke-RestMethod @request
}
$sshCommand = 'C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o StrictHostKeyChecking=yes'
Git config core.sshCommand $sshCommand
Git remote set-url --push origin git@github.com:jagoff2/sunnypilot.git
Git push --dry-run origin master
$publishedModules = @()
$missing = @()

foreach ($module in $modules) {
  $path = $module.Path
  if (-not (Test-Path (Join-Path $repoRoot "$path/.git"))) {
    Git submodule update --init --recursive -- $path
  }
  if (Git -C $path status --porcelain) { throw "Commit or stash changes in $path first." }
  $pinned = ((Git ls-tree HEAD -- $path) -split '\s+')[2]
  if ((Git -C $path rev-parse HEAD) -ne $pinned) { throw "$path must be at its parent-pinned commit." }
  $name = ($module.Source -split '/')[1]
  $forkPath = "jagoff2/$name"
  Git -C $path config core.sshCommand $sshCommand
  Git -C $path remote set-url --push origin "git@github.com:$forkPath.git"
  $source = Api "repos/$($module.Source)"
  $fork = $null
  try { $fork = Api "repos/$forkPath" } catch {
    if ([int]$_.Exception.Response.StatusCode -ne 404) { throw }
    $missing += "https://github.com/$($module.Source)/fork"
    Write-Warning "$forkPath is not publicly available. Keeping its original clone URL."
    continue
  }
  if (-not $fork -or $fork.private) { throw "$forkPath must be public." }
  $sourceId = if ($source.source) { $source.source.id } else { $source.id }
  if (-not $fork.fork -or $fork.source.id -ne $sourceId) { throw "$forkPath is not in the expected fork network." }

  $remotes = @(Git -C $path remote)
  if ($remotes -notcontains 'upstream') { Git -C $path remote add upstream "https://github.com/$($module.Source).git" }
  Git -C $path remote set-url --push origin "git@github.com:$forkPath.git"
  Git -C $path config core.autocrlf false
  Git -C $path config push.default current
  Git -C $path config push.recurseSubmodules check
  Git -C $path config remote.pushDefault origin
  $branches = @(Git -C $path branch --list c4-dev)
  if ($branches.Count) {
    if ((Git -C $path rev-parse c4-dev) -ne $pinned) { throw "$path c4-dev differs from the pinned commit; reconcile it manually." }
    Git -C $path switch c4-dev
  } else { Git -C $path switch -c c4-dev }
  Git -C $path push --set-upstream origin c4-dev
  $published = Git ls-remote "https://github.com/$forkPath.git" refs/heads/c4-dev
  if (($published -split '\s+')[0] -ne $pinned) { throw "Published commit mismatch for $path." }
  Git -C $path remote set-url origin "https://github.com/$forkPath.git"
  $publishedModules += $module
}

# Only change fresh-clone URLs for successfully published, publicly readable forks.
foreach ($module in $publishedModules) {
  $name = ($module.Source -split '/')[1]
  Git config -f .gitmodules "submodule.$($module.Name).url" "https://github.com/jagoff2/$name.git"
}
Git submodule sync --recursive
Git config push.recurseSubmodules check
Git config remote.pushDefault origin
Git diff --check
Git diff -- .gitmodules
if ($missing.Count) {
  Write-Warning 'SSH cannot create GitHub repositories. Create these public forks as jagoff2, then rerun:'
  $missing | ForEach-Object { Write-Host $_ }
} else { Write-Host 'All submodule forks are ready.' }
Write-Host 'Review and commit .gitmodules with the parent setup files, then push master.'
