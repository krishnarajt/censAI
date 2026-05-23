# Process-Videos.ps1
param (
    [Parameter(Mandatory=$true)]
    [string]$RootFolder
)

# Define supported video extensions
$videoExtensions = @("*.mp4", "*.mkv", "*.avi", "*.mov", "*.wmv", "*.flv", "*.webm")

# Create log file
$logFile = Join-Path -Path $RootFolder -ChildPath "SceneDetect_Log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
"SceneDetect Processing Log - Started at $(Get-Date)" | Out-File -FilePath $logFile -Append

# Get all video files
$videoFiles = Get-ChildItem -Path $RootFolder -Include $videoExtensions -File -Recurse
$totalFiles = $videoFiles.Count
$currentFile = 0

# Process each video file
foreach ($video in $videoFiles) {
    $currentFile++
    $videoName = [System.IO.Path]::GetFileNameWithoutExtension($video.Name)
    $outputDir = Join-Path -Path $RootFolder -ChildPath $videoName

    # Update progress bar
    $percentComplete = ($currentFile / $totalFiles) * 100
    Write-Progress -Activity "Processing Videos" -Status "Processing $videoName ($currentFile of $totalFiles)" -PercentComplete $percentComplete

    # Create output directory if it doesn't exist
    if (-not (Test-Path -Path $outputDir)) {
        New-Item -Path $outputDir -ItemType Directory | Out-Null
    }

    # Run SceneDetect command
    $command = "scenedetect -i `"$($video.FullName)`" save-images --num-images 1 --output `"$outputDir`""
    try {
        Invoke-Expression $command
        "[$(Get-Date)] Successfully processed: $videoName" | Out-File -FilePath $logFile -Append
    }
    catch {
        "[$(Get-Date)] Error processing $videoName : $_" | Out-File -FilePath $logFile -Append
    }
}

# Complete progress bar
Write-Progress -Activity "Processing Videos" -Status "Completed" -Completed
"SceneDetect Processing Log - Completed at $(Get-Date)" | Out-File -FilePath $logFile -Append
Write-Host "Processing complete. Log file saved at: $logFile"