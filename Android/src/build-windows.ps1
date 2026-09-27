param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]] $GradleArgs
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$aliasRoot = Join-Path $env:PUBLIC 'gallery-build'
$javaHome = Join-Path $env:ProgramFiles 'Android\Android Studio\jbr'

if (-not (Test-Path -LiteralPath (Join-Path $javaHome 'bin\java.exe'))) {
  throw "Android Studio JDK was not found at '$javaHome'."
}

if (-not (Test-Path -LiteralPath $aliasRoot)) {
  New-Item -ItemType Junction -Path $aliasRoot -Target $projectRoot | Out-Null
}
$alias = Get-Item -LiteralPath $aliasRoot -Force
if ($alias.LinkType -ne 'Junction' -or
    -not [string]::Equals([string] $alias.Target, $projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "'$aliasRoot' already exists and does not point to '$projectRoot'."
}

$previousJavaHome = $env:JAVA_HOME
$previousJavaToolOptions = $env:JAVA_TOOL_OPTIONS
$exitCode = 1
try {
  $env:JAVA_HOME = $javaHome
  $env:JAVA_TOOL_OPTIONS = "$previousJavaToolOptions -Djdk.net.unixdomain.tmpdir=$env:PUBLIC".Trim()
  if (-not $GradleArgs -or $GradleArgs.Count -eq 0) {
    $GradleArgs = @(':app:assembleDebug', '--console=plain')
  }

  Push-Location (Join-Path $aliasRoot 'Android\src')
  try {
    & .\gradlew.bat @GradleArgs
    $exitCode = $LASTEXITCODE
  } finally {
    Pop-Location
  }
} finally {
  $env:JAVA_HOME = $previousJavaHome
  $env:JAVA_TOOL_OPTIONS = $previousJavaToolOptions
}

exit $exitCode
