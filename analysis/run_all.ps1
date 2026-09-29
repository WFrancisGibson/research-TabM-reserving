##########################################
#########  run the whole chain from 01 data and ccODP to the tests
#########  powershell -ExecutionPolicy Bypass -File analysis/run_all.ps1
#########    [-Quick] [-Config <profile>] [-Detach] [-Parallel 6] [-Threads 3]
#########    [-SkipClaimSplit] [-SkipBootstrap] [-SkipTests]
#########  01 -> 02-05 time-aware -> 02-05 claim-level -> 06 bootstrap -> tests
#########  02 and 04 run one process per configuration, 06 one per LoB
##########################################

param(
  [switch]$Quick,
  [string]$Config = "default",
  [switch]$Detach,
  [int]$Parallel = 6,
  [int]$Threads = 3,
  [switch]$SkipClaimSplit,
  [switch]$SkipBootstrap,
  [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$Rscript = "C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

## settings: the config.yml profile of every R process (R_CONFIG_ACTIVE);
## -Quick is the quick profile, one process at a time
if ($Quick) { $Config = "quick"; $Parallel = 1 }
$env:R_CONFIG_ACTIVE = $Config
$output = if ($Config -eq "default") { "output" } else { "output\$Config" }
$logDir = Join-Path $root "$output\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

## -Detach: relaunch hidden with the same parameters, output to run_all_log.txt
## (UTF-16, as *>> writes it); -EncodedCommand keeps the quotes of the paths
if ($Detach) {
  $flags = @()
  foreach ($kv in $PSBoundParameters.GetEnumerator()) {
    if ($kv.Key -eq "Detach") { continue }
    if ($kv.Value -is [switch]) {
      if ($kv.Value) { $flags += "-$($kv.Key)" }
    } else {
      $flags += "-$($kv.Key)"
      $flags += "$($kv.Value)"
    }
  }
  $log = Join-Path $logDir "run_all_log.txt"
  $cmd = "& '$PSCommandPath' $($flags -join ' ') *>> '$log'"
  $encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($cmd))
  Start-Process -FilePath powershell `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encoded) `
    -WorkingDirectory $root -WindowStyle Hidden
  Write-Host "started detached; follow with:  Get-Content -Wait `"$log`""
  exit 0
}
Write-Host ("config profile {0}: outputs in {1}" -f $Config, $output)

## the analysis scripts; the step ids name the log files
$scripts = @{
  "01_data_and_ccodp" = "analysis\01_data and ccODP\data and ccODP.R"
  "02_curves"         = "analysis\02_validation curves\validation curves.R"
  "03_select"         = "analysis\03_stopping rules\stopping rules.R"
  "04_refit"          = "analysis\04_refit\refit.R"
  "05_report"         = "analysis\05_report\report.R"
  "06_bootstrap"      = "analysis\06_bootstrap\bootstrap.R"
}

## runs a command with its output shown and copied to a UTF-8 log (Tee-Object
## writes UTF-16 in 5.1); stderr lines become text, so an R warning does not
## stop the chain, and success is Rscript's exit code
function Invoke-Logged([string]$log, [scriptblock]$command) {
  $ErrorActionPreference = "Continue"
  $utf8 = New-Object System.Text.UTF8Encoding($false)
  $writer = New-Object System.IO.StreamWriter($log, $false, $utf8)
  try {
    & $command 2>&1 | ForEach-Object {
      if ($_ -is [System.Management.Automation.ErrorRecord]) {
        $line = $_.Exception.Message
      } else {
        $line = "$_"
      }
      $writer.WriteLine($line)
      $writer.Flush()
      Write-Host $line          # to the console, not to the function's output
    }
    $code = $LASTEXITCODE
  } finally {
    $writer.Dispose()
  }
  $ErrorActionPreference = "Stop"
  $code
}

## runs step $name in the foreground with the positional arguments $stepArgs;
## log <name>_<stem>_log.txt
function Run-Step([string]$name, [string[]]$stepArgs, [string]$stem) {
  $log = Join-Path $logDir "$name`_$stem`_log.txt"
  Write-Host ("[{0}] {1} {2}" -f (Get-Date -Format "HH:mm:ss"),
              $name, ($stepArgs -join " "))
  $script = Join-Path $root $scripts[$name]
  $code = Invoke-Logged $log { & $Rscript $script @stepArgs }
  if ($code -ne 0) { throw "$name failed (exit $code); see $log" }
}

## runs step $name with the arguments <lead> <item> <threads> for every item,
## at most $Parallel processes at a time (one at a time: no <threads>);
## logs <name>_<stem>_<item>_log.txt (stdout) and ..._err.txt (stderr)
function Run-Parallel([string]$name, [string[]]$items, [string[]]$lead, [string]$stem) {
  if ($Parallel -le 1) {
    foreach ($item in $items) { Run-Step $name (@($lead) + @($item)) "$stem`_$item" }
    return
  }
  $jobs = @()
  foreach ($item in $items) {
    while (@($jobs | Where-Object { -not $_.Proc.HasExited }).Count -ge $Parallel) {
      Start-Sleep -Seconds 10
    }
    $out = Join-Path $logDir "$name`_$stem`_$item`_log.txt"
    $err = Join-Path $logDir "$name`_$stem`_$item`_err.txt"
    $stepArgs = @($lead) + @($item, "$Threads")
    # 5.1 joins -ArgumentList without quoting: the script path is quoted here
    $a = @("`"$(Join-Path $root $scripts[$name])`"") + $stepArgs
    Write-Host ("[{0}] {1} {2} (background)" -f (Get-Date -Format "HH:mm:ss"),
                $name, ($stepArgs -join " "))
    $p = Start-Process -FilePath $Rscript -ArgumentList $a -WorkingDirectory $root `
      -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    $jobs += [pscustomobject]@{ Proc = $p; Item = $item; Err = $err }
  }
  foreach ($j in $jobs) { $j.Proc.WaitForExit() }
  # ExitCode is sometimes $null after the exit; Rscript prints "Execution
  # halted" on any uncaught error, so the error log is checked as well
  foreach ($j in $jobs) {
    $j.Proc.Refresh()
    $code = $j.Proc.ExitCode
    $halted = (Test-Path $j.Err) -and
      (Select-String -Path $j.Err -Pattern "Execution halted" -Quiet)
    if (($null -ne $code -and $code -ne 0) -or $halted) {
      throw "$name failed for $($j.Item) (exit $code); see $($j.Err)"
    }
    Write-Host ("[{0}] {1} {2} finished" -f (Get-Date -Format "HH:mm:ss"), $name, $j.Item)
  }
}

$configs = @("single_bccnn", "tabm_tanh_signs", "tabm_tanh_normal",
             "tabm_relu_signs", "tabm_relu_normal", "packed_tanh")
$started = Get-Date

## data, ccODP and chain ladder (Paper C Table 2), split geometry
Run-Step "01_data_and_ccodp" @() "data"

## each split: validation curves, stopping rules, refits, report
$splits = @("time-aware")
if (-not $SkipClaimSplit) { $splits += "claim-level" }
foreach ($split in $splits) {
  Run-Parallel "02_curves" $configs @($split) $split
  Run-Step "03_select" @($split) $split
  Run-Parallel "04_refit" $configs @($split) $split
  Run-Step "05_report" @($split) $split
}

## bootstrap on the split of config.yml bootstrap: split (time-aware), one
## process per LoB, then one call without arguments that assembles the table
if (-not $SkipBootstrap) {
  Run-Parallel "06_bootstrap" @("1", "2", "3", "4", "5", "6") @() "time-aware"
  Run-Step "06_bootstrap" @() "time-aware_assemble"
}

## tests on the default profile; -Quick: fewer LoBs and submodels
if (-not $SkipTests) {
  $env:R_CONFIG_ACTIVE = "default"
  if ($Quick) { $env:TABM_QUICK_TESTS = "1" }
  $log = Join-Path $logDir "tests_log.txt"
  Write-Host ("[{0}] tests" -f (Get-Date -Format "HH:mm:ss"))
  $code = Invoke-Logged $log { & $Rscript (Join-Path $root "tests\testthat.R") }
  if ($code -ne 0) { throw "tests failed (exit $code); see $log" }
}

Write-Host ("done in {0:N1} minutes" -f ((Get-Date) - $started).TotalMinutes)
