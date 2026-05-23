# Prompt user for the root folder containing images
$folderPath = Read-Host "Enter the full path to the root folder containing images"

# Verify folder exists
if (-not (Test-Path -Path $folderPath -PathType Container)) {
    Write-Host "The specified folder does not exist. Exiting."
    exit
}

# Define the output PDF file path (in the same folder, named output.pdf)
# Get the folder name from the path
$folderName = Split-Path -Path $folderPath -Leaf
# Define the output PDF file path using the folder name
$outputPdf = Join-Path -Path $folderPath -ChildPath "$folderName.pdf"


# Get all image files in the folder (adjust extensions as needed)

$imageFiles = Get-ChildItem -Path $folderPath -Include *.jpg, *.jpeg, *.png, *.bmp, *.gif, *.tiff, *.JPG, *.JPEG, *.PNG, *.BMP, *.GIF, *.TIFF -File -Recurse | Sort-Object LastWriteTime

if ($imageFiles.Count -eq 0) {
    Write-Host "No image files found in the specified folder."
    exit
}

# Prepare image paths and show progress bar while processing
$imagePaths = @()
$total = $imageFiles.Count
for ($i = 0; $i -lt $total; $i++) {
    $imagePaths += "`"$($imageFiles[$i].FullName)`""
    $percent = [int](($i + 1) / $total * 100)
    Write-Progress -Activity "Processing images" -Status "Adding image $($i + 1) of $total" -PercentComplete $percent
}

# Join all image paths into a single string separated by spaces
$imagePathsString = $imagePaths -join " "

# Run ImageMagick's magick command to convert images to a single PDF
$command = "magick $imagePathsString `"$outputPdf`""

Write-Host "Running command:"
Write-Host $command

Invoke-Expression $command

Write-Host "PDF created successfully at $outputPdf"
