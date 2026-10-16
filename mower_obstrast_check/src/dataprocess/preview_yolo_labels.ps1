param(
    [string]$Dataset = "$PSScriptRoot\..\..\data\yolo_check",
    [string]$Output = "$PSScriptRoot\..\..\data\yolo_train_review_100",
    [int]$Count = 100,
    [int]$Seed = 20261016,
    [ValidateSet('train', 'val', 'test')][string]$Split = 'train'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$Dataset = (Resolve-Path -LiteralPath $Dataset).Path
if ((Test-Path -LiteralPath $Output) -and (Get-ChildItem -LiteralPath $Output -Force | Select-Object -First 1)) {
    throw "Output must be empty: $Output"
}
$names = @{}
foreach ($line in Get-Content -Encoding UTF8 -LiteralPath "$Dataset\data.yaml") {
    if ($line -match '^\s+(\d+):\s+(.+?)\s*$') { $names[[int]$Matches[1]] = $Matches[2].Trim('"', "'") }
}
$files = @(Get-ChildItem -LiteralPath "$Dataset\images\$Split" -Filter '*.jpg' | Sort-Object Name)
if ($Count -eq 0) { $Count = $files.Count }
if ($Count -lt 1 -or $Count -gt $files.Count) { throw 'Invalid sample count' }
$random = [System.Random]::new($Seed)
for ($i = $files.Count - 1; $i -gt 0; $i--) {
    $j = $random.Next($i + 1)
    $temp = $files[$i]; $files[$i] = $files[$j]; $files[$j] = $temp
}
foreach ($folder in @('annotated', 'images', 'labels')) {
    New-Item -ItemType Directory -Path "$Output\$folder" -Force | Out-Null
}
$palette = @('Lime', 'Orange', 'Cyan', 'Magenta', 'Yellow', 'DeepSkyBlue')
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$boxes = [System.Collections.Generic.List[object]]::new()
$selection = [System.Collections.Generic.List[object]]::new()
foreach ($file in $files[0..($Count - 1)]) {
    $label = "$Dataset\labels\$Split\$($file.BaseName).txt"
    # Independent copies, never draw into the dataset's hardlinked images.
    Copy-Item -LiteralPath $file.FullName -Destination "$Output\images\$($file.Name)"
    Copy-Item -LiteralPath $label -Destination "$Output\labels\$($file.BaseName).txt"
    $sourceImage = [System.Drawing.Image]::FromFile($file.FullName)
    $bitmap = [System.Drawing.Bitmap]::new($sourceImage.Width, $sourceImage.Height, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try { $graphics.DrawImage($sourceImage, 0, 0, $sourceImage.Width, $sourceImage.Height) }
    finally { $sourceImage.Dispose() }
    $font = [System.Drawing.Font]::new('Arial', 11, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    try {
        $index = 0
        foreach ($line in Get-Content -LiteralPath $label) {
            if (-not $line.Trim()) { continue }
            $fields = $line.Trim() -split '\s+'
            if ($fields.Count -ne 5) { throw "Invalid detection label: $label" }
            $class = [int]$fields[0]
            if (-not $names.ContainsKey($class)) { throw "Unknown class: $class" }
            $xc, $yc, $w, $h = @($fields[1..4] | ForEach-Object { [double]::Parse($_, $culture) })
            $x1 = ($xc - $w / 2) * $bitmap.Width
            $y1 = ($yc - $h / 2) * $bitmap.Height
            $x2 = ($xc + $w / 2) * $bitmap.Width
            $y2 = ($yc + $h / 2) * $bitmap.Height
            $index++
            $pen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromName($palette[$class % $palette.Count]), 2)
            try {
                $graphics.DrawRectangle($pen, [single]$x1, [single]$y1, [single]($x2-$x1), [single]($y2-$y1))
                $text = "$index`: $($names[$class]) [$class]"
                $size = $graphics.MeasureString($text, $font)
                $tx = [single][Math]::Max(0, [Math]::Min($x1, $bitmap.Width - $size.Width))
                $ty = [single][Math]::Max(0, $y1 - $size.Height)
                $graphics.FillRectangle([System.Drawing.Brushes]::Black, $tx, $ty, $size.Width, $size.Height)
                $graphics.DrawString($text, $font, [System.Drawing.Brushes]::White, $tx, $ty)
            } finally { $pen.Dispose() }
            $boxes.Add([pscustomobject]@{
                image = $file.Name; box = $index; class_id = $class; class_name = $names[$class]
                image_width = $bitmap.Width; image_height = $bitmap.Height
                x1 = $x1; y1 = $y1; x2 = $x2; y2 = $y2
            })
        }
        $bitmap.Save("$Output\annotated\$($file.BaseName).png", [System.Drawing.Imaging.ImageFormat]::Png)
        $selection.Add([pscustomobject]@{image = $file.Name; boxes = $index})
    } finally { $font.Dispose(); $graphics.Dispose(); $bitmap.Dispose() }
}
$boxes | Export-Csv -LiteralPath "$Output\pixel_boxes.csv" -NoTypeInformation -Encoding UTF8
@{seed = $Seed; count = $Count; dataset = $Dataset; split = $Split; samples = @($selection.ToArray())} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$Output\selection.json" -Encoding UTF8
Write-Output "Completed: $Count images, $($boxes.Count) boxes. Output: $Output"
