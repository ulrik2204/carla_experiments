# Define a function to execute when CTRL+C is caught
function Cleanup-And-Exit {
    Write-Host "Interrupt received. Exiting..."
    # Perform any necessary cleanup here
    npx kill-port 2000
    # Exit the script
    exit 1
}

# Trap CTRL+C and call the Cleanup-And-Exit function when caught
trap { Cleanup-And-Exit }

# Get current date and time
$this_time = Get-Date -Format "yyyy-MM-dd-HH-mm"

# Set root folder based on arguments
if ($args.Count -eq 0) {
    $root_folder = "output/$this_time"
}
else {
    $root_folder = $args[0]
}

# Set progress file based on arguments
if ($args.Count -eq 2) {
    $progress_file = $args[1]
}
else {
    $progress_file = "progress-$this_time.txt"
}

$comma_folder = "$root_folder/comma2k19"

$max_attempts = 2300
$attempt_num = 1

while ($true) {
    Write-Host "Attempt $attempt_num"
    Write-Host "Starting in $root_folder"
    # Start-Process -FilePath "..\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe", "-RenderOffScreen"
    Start-Process -FilePath "..\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe" -ArgumentList "-RenderOffScreen"
    Write-Host "Waiting for Carla to start..."
    Start-Sleep -Seconds 10
    poetry run python -m openpilot_exploration.datagen.generate_comma2k19_data --root-folder=$comma_folder --progress-file=$progress_file
    
    # Check the exit status of the command
    if ($LastExitCode -eq 0) {
        Write-Host "Command succeeded."
        break
    }
    else {
        $attempt_num++
        if ($attempt_num -gt $max_attempts) {
            Write-Host "Maximum attempts reached. Exiting."
            break
        }
        else {
            Write-Host "Command failed. Retrying..."
            Start-Sleep -Seconds 2
            npx kill-port 2000
            Start-Sleep -Seconds 2
        }
    }
}

poetry run python -m openpilot_exploration.datagen.extract_comma2k19 --root-folder=$root_folder --comma2k19-folder=$comma_folder  # TODO: Add args
Start-Sleep -Seconds 2
npx kill-port 2000