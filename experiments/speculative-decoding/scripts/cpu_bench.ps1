# CPU-only speculative-decoding benchmark (PowerShell HTTP client — robust on long CPU requests).
# Mirrors bench.py methodology: 3B target + 0.5B draft, greedy, 256 tokens, 12-prompt suite.
$ErrorActionPreference = "Stop"
# Paths are configurable (override via env vars); defaults match the repo layout.
$root = Split-Path -Parent $PSScriptRoot
$exe = if ($env:SPECBENCH_VULKAN_EXE) { $env:SPECBENCH_VULKAN_EXE } else { "$root\runtimes\vulkan\llama-server.exe" }
$modelsDir = if ($env:SPECBENCH_MODELS_DIR) { $env:SPECBENCH_MODELS_DIR } else { "$root\models" }
$tgt = "$modelsDir\Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
$drf = "$modelsDir\Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"
$prompts = Get-Content "$root\results\prompts.json" -Raw | ConvertFrom-Json
$port = 8099
$base = @('--host','127.0.0.1','--port',"$port",'--model',$tgt,'-ngl','0','--device','none','-c','4096','--jinja','--no-webui')

$methods = @(
  @{ name='baseline';   label='No speculation';      args=@() },
  @{ name='draft_0_5b'; label='Draft model (0.5B)';  args=@('-md',$drf,'--spec-type','draft-simple','--spec-draft-n-max','5') },
  @{ name='ngram';      label='N-gram (model-free)'; args=@('--spec-type','ngram-cache','--spec-draft-n-max','5') }
)

function Send($content,$mt){
  $body = @{ messages=@(@{role='user';content=$content}); max_tokens=$mt; temperature=0; top_k=1; seed=1234; cache_prompt=$true } | ConvertTo-Json -Compress -Depth 6
  $rr = Invoke-RestMethod "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 300
  return $rr
}

function Start-Srv($extra){
  Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep 1
  $p = Start-Process -FilePath $exe -ArgumentList ($base + $extra) -PassThru -RedirectStandardOutput "$root\results\server-cpu.log" -RedirectStandardError "$root\results\server-cpu.err.log"
  for($i=0;$i -lt 60;$i++){ Start-Sleep 2; if($p.HasExited){break}; try{ $h=Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2; if($h.status -eq 'ok'){ return $p } }catch{} }
  return $p
}

$runs = @()
foreach($m in $methods){
  Write-Host "=== cpu / $($m.name) ==="
  # Draft loads two models on CPU and was unstable across requests -> restart the server per prompt.
  $perPrompt = ($m.name -eq 'draft_0_5b')
  $records = @()
  $p = $null
  if(-not $perPrompt){ $p = Start-Srv $m.args; try { Send "Say hello." 16 | Out-Null } catch {} }
  foreach($pr in $prompts){
    if($perPrompt){ $p = Start-Srv $m.args; if($p.HasExited){ Write-Host "  $($pr.id): server failed"; continue } }
    try {
      $rr = Send $pr.prompt 256
      $tps = [math]::Round($rr.timings.predicted_per_second,2)
      $records += @{ id=$pr.id; category=$pr.category; tps=$tps; predicted_n=$rr.timings.predicted_n }
      Write-Host ("  {0,-20} {1,7} tok/s" -f $pr.id,$tps)
    } catch {
      Write-Host "  $($pr.id): ERR $($_.Exception.Message)"
      $records += @{ id=$pr.id; category=$pr.category; error="$($_.Exception.Message)" }
    }
    if($perPrompt -and -not $p.HasExited){ Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; Start-Sleep 1 }
  }
  $runs += @{ backend='cpu'; method=$m.name; label=$m.label; status='ok'; records=$records }
  if($p -and -not $p.HasExited){ Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  Start-Sleep 2
}

$out = @{ meta=@{ target=$tgt; draft=$drf; max_tokens=256; ctx=4096; gpu='CPU (no GPU)'; model='Qwen2.5-Coder-3B Q4_K_M' }; runs=$runs }
$out | ConvertTo-Json -Depth 8 | Out-File -FilePath "$root\results\results_cpu.json" -Encoding utf8
Write-Host "Wrote results_cpu.json"
