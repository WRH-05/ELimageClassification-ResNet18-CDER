[CmdletBinding()]
param(
    [ValidateSet("all", "train", "export", "report", "infer")]
    [string]$Mode = "all",

    [switch]$UseGPU,

    [string]$CsvPath = "labels.csv",
    [string]$DataRoot = ".",

    [int]$Epochs = 20,
    [int]$BatchSize = 32,
    [double]$LearningRate = 1e-4,
    [double]$WeightDecay = 1e-5,
    [int]$NumWorkers = 2,
    [int]$ImageSize = 224,
    [int]$Seed = 42,

    [string]$LossType = "smoothl1",
    [double]$HuberBeta = 0.5,
    [int]$SchedulerPatience = 3,
    [double]$SchedulerFactor = 0.5,
    [double]$SchedulerMinLr = 1e-6,
    [int]$EarlyStoppingPatience = 7,
    [double]$EarlyStoppingDelta = 1e-4,

    [string]$CheckpointPath = "best_model_v2_full.pth",
    [string]$OnnxOutput = "best_model_v2_full.onnx",

    [string]$ReportCsv = "test_split_report_v2_onnx.csv",

    [string]$InferenceImagePath = "images/cell0001.png",
    [string]$PadId = "simulated_pad_01",
    [double]$CriticalThreshold = 0.8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe. Create/select the project .venv first."
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StepName,
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    Write-Host "`n=== $StepName ===" -ForegroundColor Cyan
    Write-Host "$pythonExe $($Args -join ' ')" -ForegroundColor DarkGray

    & $pythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

$device = if ($UseGPU) { "cuda" } else { "cpu" }

if ($Mode -in @("all", "train")) {
    Invoke-PythonStep -StepName "Train V2" -Args @(
        "train.py",
        "--csv_path", $CsvPath,
        "--data_root", $DataRoot,
        "--epochs", "$Epochs",
        "--batch_size", "$BatchSize",
        "--learning_rate", "$LearningRate",
        "--weight_decay", "$WeightDecay",
        "--num_workers", "$NumWorkers",
        "--image_size", "$ImageSize",
        "--seed", "$Seed",
        "--checkpoint_path", $CheckpointPath,
        "--device", $device,
        "--loss_type", $LossType,
        "--huber_beta", "$HuberBeta",
        "--scheduler_patience", "$SchedulerPatience",
        "--scheduler_factor", "$SchedulerFactor",
        "--scheduler_min_lr", "$SchedulerMinLr",
        "--early_stopping_patience", "$EarlyStoppingPatience",
        "--early_stopping_delta", "$EarlyStoppingDelta"
    )
}

if ($Mode -in @("all", "export")) {
    Invoke-PythonStep -StepName "Export ONNX" -Args @(
        "export_model.py",
        "--checkpoint", $CheckpointPath,
        "--onnx_output", $OnnxOutput,
        "--image_size", "$ImageSize"
    )
}

if ($Mode -in @("all", "report")) {
    Invoke-PythonStep -StepName "Generate Test Report" -Args @(
        "evaluate_test_split_report.py",
        "--csv_path", $CsvPath,
        "--data_root", $DataRoot,
        "--seed", "$Seed",
        "--image_size", "$ImageSize",
        "--onnx_model", $OnnxOutput,
        "--output_csv", $ReportCsv
    )
}

if ($Mode -in @("all", "infer")) {
    Invoke-PythonStep -StepName "Run Inference" -Args @(
        "inference_mqtt_mock.py",
        "--onnx_model", $OnnxOutput,
        "--image_path", $InferenceImagePath,
        "--pad_id", $PadId,
        "--critical_threshold", "$CriticalThreshold",
        "--image_size", "$ImageSize"
    )
}

Write-Host "`nCompleted mode '$Mode' successfully." -ForegroundColor Green
