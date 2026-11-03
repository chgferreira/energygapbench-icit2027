#!/usr/bin/env bash
# The supported entry point for the existing cuda_env on the Linux CUDA server.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
task_mode="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
task_threads=32
task_budget=''
task_id=''
task_preset=focused
task_preset_explicit=false
task_config=''
task_default_budget=24
task_validation_level=fast
task_validation_device=cuda:0
task_energy_workers=''
task_energy_pending=''
task_allow_runtime_patch=false
task_report_workers=''
task_report_chunk=''
task_report_bootstrap_chunk=''
task_usage() {
  printf '%s\n' \
    'Usage: bash scripts/server.sh setup [--preset focused|full | --config PATH] [--threads 32]' \
    '       bash scripts/server.sh run [--preset focused|full | --config PATH] [--threads 32] [--budget-hours HOURS] [--experiment-id NEW_ID]' \
    '       bash scripts/server.sh resume ID [--threads 32] [--budget-hours TOTAL_HOURS]' \
    '       Resume scheduling options: --energy-cpu-workers N --energy-cpu-max-pending N --allow-runtime-patch' \
    '       Report scheduling options: --report-workers N --report-chunk-size N --report-bootstrap-chunk-size N' \
    '       bash scripts/server.sh validate [fast|core|native|full] [--preset focused|full | --config PATH] [--threads 32] [--device cuda:0|cpu]' \
    'Use the already active cuda_env. Setup installs extras and downloads pinned public checkpoints.' \
    'Default preset: focused (24 cumulative hours); full keeps the complete matrix (168 hours).' \
    'Run starts the selected configuration from zero; resume uses only its recorded configuration.' \
    'Validation is optional: fast checks sources/config; core runs regressions once; native checks selected adapters; full runs all tests once.' \
    'Logs are written automatically under artifacts/analyses/logs; both run modes use tqdm.'
}
if [[ "$task_mode" == --help || "$task_mode" == -h ]]; then task_usage; exit 0; fi
if [[ "$task_mode" != setup && "$task_mode" != run && "$task_mode" != resume && "$task_mode" != validate ]]; then task_usage >&2; exit 2; fi
if [[ "$task_mode" == resume && $# -gt 0 && "$1" != --* ]]; then task_id="$1"; shift; fi
if [[ "$task_mode" == validate && $# -gt 0 && "$1" != --* ]]; then task_validation_level="$1"; shift; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) task_usage; exit 0 ;;
    --threads) task_threads="${2:?Missing thread count}"; shift 2 ;;
    --budget-hours) task_budget="${2:?Missing cumulative budget}"; shift 2 ;;
    --experiment-id) task_id="${2:?Missing experiment ID}"; shift 2 ;;
    --preset) task_preset="${2:?Missing preset}"; task_preset_explicit=true; shift 2 ;;
    --config) task_config="${2:?Missing configuration path}"; shift 2 ;;
    --energy-cpu-workers) task_energy_workers="${2:?Missing correction worker count}"; shift 2 ;;
    --energy-cpu-max-pending) task_energy_pending="${2:?Missing correction queue size}"; shift 2 ;;
    --allow-runtime-patch) task_allow_runtime_patch=true; shift ;;
    --report-workers) task_report_workers="${2:?Missing bootstrap worker count}"; shift 2 ;;
    --report-chunk-size) task_report_chunk="${2:?Missing report export batch size}"; shift 2 ;;
    --report-bootstrap-chunk-size) task_report_bootstrap_chunk="${2:?Missing bootstrap publication size}"; shift 2 ;;
    --device)
      if [[ "$task_mode" != validate ]]; then printf '%s\n' '--device is supported only by validate.' >&2; exit 2; fi
      task_validation_device="${2:?Missing validation device}"; shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ "$task_mode" != run && "$task_mode" != resume && ( -n "$task_energy_workers" || -n "$task_energy_pending" || "$task_allow_runtime_patch" == true || -n "$task_report_workers" || -n "$task_report_chunk" || -n "$task_report_bootstrap_chunk" ) ]]; then
  printf '%s\n' 'Runtime scheduling options are supported only by run/resume.' >&2; exit 2
fi
if [[ "$task_allow_runtime_patch" == true && "$task_mode" != resume ]]; then
  printf '%s\n' '--allow-runtime-patch requires resume of an existing ID.' >&2; exit 2
fi
for task_number in "$task_energy_workers" "$task_energy_pending" "$task_report_workers" "$task_report_chunk" "$task_report_bootstrap_chunk"; do
  if [[ -n "$task_number" && ! "$task_number" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s\n' 'Runtime worker/queue sizes must be positive integers.' >&2; exit 2
  fi
done
if [[ "$task_preset" != focused && "$task_preset" != full ]]; then printf '%s\n' 'Preset must be focused or full.' >&2; exit 2; fi
if [[ "$task_preset_explicit" == true && -n "$task_config" ]]; then printf '%s\n' 'Use either --preset or --config, not both.' >&2; exit 2; fi
if [[ "$task_mode" == resume && ( "$task_preset_explicit" == true || -n "$task_config" ) ]]; then
  printf '%s\n' 'Resume uses the locked configuration; --preset and --config are not allowed.' >&2; exit 2
fi
if [[ -z "$task_config" ]]; then
  task_config="configs/experiments/server_${task_preset}.yaml"
  if [[ "$task_preset" == full ]]; then task_default_budget=168; fi
else
  task_preset=custom
fi
if [[ ! "$task_threads" =~ ^[1-9][0-9]*$ ]]; then printf '%s\n' 'Threads must be a positive integer.' >&2; exit 2; fi
if [[ -n "$task_budget" ]] && { [[ ! "$task_budget" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$task_budget" =~ ^0+([.]0+)?$ ]]; }; then
  printf '%s\n' 'Budget must be a positive number of cumulative hours.' >&2; exit 2
fi
if [[ ( "$task_mode" == setup || "$task_mode" == validate ) && ( -n "$task_id" || -n "$task_budget" ) ]]; then task_usage >&2; exit 2; fi
if [[ "$task_mode" == resume && -z "$task_id" ]]; then printf '%s\n' 'Resume requires the existing experiment ID.' >&2; exit 2; fi
if [[ -n "$task_id" && ! "$task_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  printf '%s\n' 'Experiment ID must contain only letters, numbers, dots, underscores, or hyphens.' >&2; exit 2
fi
if [[ "$task_mode" == validate ]]; then
  exec python -u scripts/validate_release.py --level "$task_validation_level" \
    --threads "$task_threads" --device "$task_validation_device" --config "$task_config"
fi
task_stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ "$task_mode" == run && -z "$task_id" ]]; then task_id="conference-${task_stamp}"; fi
task_log_id="${task_id:-setup-${task_stamp}}"
task_log="artifacts/analyses/logs/${task_log_id}.log"
mkdir -p -- "$(dirname -- "$task_log")"

task_execute() {
  printf '[1/5] %s; projeto: %s\nLog: %s/%s\n' "$task_mode" "$task_root" "$task_root" "$task_log"
  printf 'Ambiente: cuda_env ativo; dispositivo: cuda:0; threads: %s.\n' "$task_threads"
  if [[ "$task_mode" != resume ]]; then printf 'Preset: %s; configuração: %s.\n' "$task_preset" "$task_config"; fi
  if [[ "$(uname -s)" != Linux ]]; then printf '%s\n' 'Execute no servidor Linux CUDA. O macOS não é usado para validar CUDA.' >&2; return 2; fi
  local task_tool
  for task_tool in python timeout flock; do
    if ! command -v "$task_tool" >/dev/null; then printf 'Comando necessário ausente: %s\n' "$task_tool" >&2; return 2; fi
  done
  # This project lock closes the race between simultaneous shell launches.
  exec 9>artifacts/analyses/server.lock
  if ! flock --nonblock 9; then printf '%s\n' 'Já há um setup ou coordenador iniciado por server.sh neste projeto. Nenhum processo foi interrompido.' >&2; return 2; fi
  export PYTHONUNBUFFERED=1
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="$task_root/src"
  export MPLBACKEND=Agg
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/energygap-mpl"
  export MPL_IGNORE_SYSTEM_FONTS=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export OMP_NUM_THREADS="$task_threads"
  export MKL_NUM_THREADS="$task_threads"
  export OPENBLAS_NUM_THREADS="$task_threads"
  export NUMEXPR_NUM_THREADS="$task_threads"
  mkdir -p -- "$MPLCONFIGDIR"
  printf '[2/5] Verificando o Python do cuda_env e outros coordenadores...\n'
  if timeout --kill-after=5s 15s python -u scripts/server_dependencies.py --environment-only; then :; else
    local task_status=$?
    printf 'Verificação do ambiente falhou (código %s; 124 indica timeout de 15 s). Nenhum experimento foi iniciado.\n' "$task_status" >&2
    return "$task_status"
  fi
  if [[ "$task_mode" == setup ]]; then
    if [[ ! -r "$task_config" ]]; then printf 'Configuração ausente: %s\n' "$task_config" >&2; return 2; fi
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    printf '[3/5] Resolvendo requisitos extras, preservando o PyTorch e o CUDA existentes...\n'
    python -u scripts/server_dependencies.py --install --audit-dir "artifacts/analyses/server_environment/setup-${task_stamp}"
    local task_models_text task_model
    task_models_text="$(python -u scripts/validate_release.py --config "$task_config" --required-models)"
    local task_models=()
    while IFS= read -r task_model; do
      if [[ -n "$task_model" ]]; then task_models+=("$task_model"); fi
    done <<< "$task_models_text"
    printf '[4/5] Obtendo e verificando as fontes e os checkpoints fixados...\n'
    while IFS= read -r task_model; do
      case "$task_model" in
        T1) python -u scripts/fetch_t1.py ;;
        MOMENT-1-base) python -u scripts/fetch_moment.py ;;
        TS-ICL) python -u scripts/fetch_tsicl.py ;;
      esac
    done <<< "$task_models_text"
    printf '[5/5] Verificando CUDA e os adaptadores do ambiente único...\n'
    timeout --kill-after=5s 120s python -u -m energygap doctor --device cuda:0 --threads "$task_threads"
    if [[ ${#task_models[@]} -gt 0 ]]; then
      if [[ " ${task_models[*]} " == *' SAITS '* ]]; then
        python -u -m energygap gpu-smoke --device cuda:0 --threads "$task_threads"
      fi
      if [[ "$task_models_text" != $'BRITS\nSAITS\nCSDI\nImputeFormer\nT1\nMOMENT-1-base\nTS-ICL' ]]; then
        python -u scripts/verify_server_models.py --device cuda:0 --threads "$task_threads" \
          --output "artifacts/analyses/server_environment/setup-${task_stamp}/native_models.json" --methods "${task_models[@]}"
      else
        python -u scripts/verify_server_models.py --device cuda:0 --threads "$task_threads" \
          --output "artifacts/analyses/server_environment/setup-${task_stamp}/native_models.json"
      fi
    fi
    printf 'Setup concluído. Inicie: bash scripts/server.sh run --config %q --threads %s --budget-hours %s\n' "$task_config" "$task_threads" "$task_default_budget"
    return
  fi
  if [[ ! -r src/energygap/cli.py || ( "$task_mode" != resume && ! -r "$task_config" ) ]]; then printf '%s\n' 'Código ou configuração ausente. Extraia o ZIP atualizado na raiz do projeto.' >&2; return 2; fi
  if [[ "$task_mode" == run ]]; then
    local task_sites task_site
    task_sites="$(python -u scripts/validate_release.py --config "$task_config" --required-sites)"
    while IFS= read -r task_site; do
      if [[ -n "$task_site" && ! -s "raw_data/${task_site}.csv" ]]; then printf 'Dado ausente ou vazio: %s/raw_data/%s.csv\n' "$task_root" "$task_site" >&2; return 2; fi
    done <<< "$task_sites"
  fi
  if [[ "$task_mode" == run && -e "runs/$task_id" ]]; then printf 'O ID já existe: runs/%s. Use resume ou omita --experiment-id para começar uma execução nova.\n' "$task_id" >&2; return 2; fi
  if [[ "$task_mode" == resume && ! -d "runs/$task_id" ]]; then printf 'Execução inexistente: runs/%s.\n' "$task_id" >&2; return 2; fi
  printf '[3/5] Verificando requisitos instalados e driver NVIDIA...\n'
  python -u scripts/server_dependencies.py --check
  if command -v nvidia-smi >/dev/null; then
    if timeout --kill-after=5s 15s nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv; then :; else
      local task_status=$?
      printf 'Consulta NVIDIA falhou (código %s; 124 indica timeout de 15 s). A matriz não foi iniciada.\n' "$task_status" >&2
      return "$task_status"
    fi
  fi
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export ENERGYGAP_STARTUP_TRACE_SECONDS=60
  printf '[4/5] Testando uma operação real CUDA (limite de 120 segundos)...\n'
  if timeout --kill-after=5s 120s python -u -m energygap doctor --device cuda:0 --threads "$task_threads"; then :; else
    local task_status=$?
    printf 'Teste CUDA falhou (código %s; 124 indica timeout de 120 s). A matriz não foi iniciada.\n' "$task_status" >&2
    return "$task_status"
  fi
  printf '[5/5] Execução: %s; progresso e estimativa de tempo em tqdm.\n' "$task_id"
  printf 'Acompanhar: python -m energygap status --experiment-id %s --watch\n' "$task_id"
  printf 'Retomar: bash scripts/server.sh resume %s --threads %s' "$task_id" "$task_threads"
  if [[ -n "$task_energy_workers" ]]; then printf ' --energy-cpu-workers %s' "$task_energy_workers"; fi
  if [[ -n "$task_energy_pending" ]]; then printf ' --energy-cpu-max-pending %s' "$task_energy_pending"; fi
  if [[ "$task_allow_runtime_patch" == true ]]; then printf ' --allow-runtime-patch'; fi
  if [[ -n "$task_report_workers" ]]; then printf ' --report-workers %s' "$task_report_workers"; fi
  if [[ -n "$task_report_chunk" ]]; then printf ' --report-chunk-size %s' "$task_report_chunk"; fi
  if [[ -n "$task_report_bootstrap_chunk" ]]; then printf ' --report-bootstrap-chunk-size %s' "$task_report_bootstrap_chunk"; fi
  printf '\n'
  local task_args
  if [[ "$task_mode" == run ]]; then
    task_budget="${task_budget:-$task_default_budget}"
    printf 'Teto cumulativo: %s horas; não é uma previsão de duração.\n' "$task_budget"
    task_args=(python -u -m energygap run --profile conference --config "$task_config" \
      --experiment-id "$task_id" --resume never --device cuda:0 --threads "$task_threads" \
      --budget-total-hours "$task_budget" --progress tqdm)
  else
    task_args=(python -u -m energygap resume --experiment-id "$task_id" --device cuda:0 --threads "$task_threads" --progress tqdm)
    if [[ -n "$task_budget" ]]; then task_args+=(--budget-total-hours "$task_budget"); fi
  fi
  if [[ -n "$task_energy_workers" ]]; then task_args+=(--energy-cpu-workers "$task_energy_workers"); fi
  if [[ -n "$task_energy_pending" ]]; then task_args+=(--energy-cpu-max-pending "$task_energy_pending"); fi
  if [[ "$task_allow_runtime_patch" == true ]]; then task_args+=(--allow-runtime-patch); fi
  if [[ -n "$task_report_workers" ]]; then task_args+=(--report-workers "$task_report_workers"); fi
  if [[ -n "$task_report_chunk" ]]; then task_args+=(--report-chunk-size "$task_report_chunk"); fi
  if [[ -n "$task_report_bootstrap_chunk" ]]; then task_args+=(--report-bootstrap-chunk-size "$task_report_bootstrap_chunk"); fi
  "${task_args[@]}"
}

# Keep the log pipe open on Ctrl+C while Python reaches its safe stop boundary.
# pipefail still preserves failed checks and the coordinator exit status.
task_execute 2>&1 | tee -ia -- "$task_log"
