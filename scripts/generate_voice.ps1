# Renders the phrase "Come here!" with the Windows built-in speech synthesizer
# (System.Speech, voice "Microsoft Zira Desktop" if installed) to a 44.1 kHz mono WAV.
#   powershell -ExecutionPolicy Bypass -File scripts/generate_voice.ps1 -OutFile <path.wav>
param(
    [Parameter(Mandatory = $true)][string]$OutFile,
    [string]$Text = "Come here!",
    [string]$Voice = "Microsoft Zira Desktop",
    [int]$Rate = -1
)
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$names = $synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }
if ($names -contains $Voice) { $synth.SelectVoice($Voice) } elseif ($names.Count -gt 0) { $synth.SelectVoice($names[0]) }
$synth.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($OutFile, $fmt)
$synth.Speak($Text)
$used = $synth.Voice.Name
$synth.SetOutputToNull()
$synth.Dispose()
Write-Output ("voice=" + $used)
