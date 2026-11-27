set -e
source /workspace/fudan/donghaosen/ATI-OPD/run/common.sh
WORKSPACE=/workspace/fudan/donghaosen/ATI-OPD
DATASET_DIR=/data/fd/donghaosen/ati-opd
VC_WORKER_NUM=2
DEFAULT_EXPERIMENT_NAME=${EXPERIMENT_NAME:-dmpd--train_on_teacher--loss_opd--prune_future}

teacher_intervention_segment_length=${TEACHER_INTERVENTION_SEGMENT_LENGTH:-128}
teacher_intervention_teacher_length=${TEACHER_INTERVENTION_TEACHER_LENGTH:-128}
teacher_intervention_gap_metric=${TEACHER_INTERVENTION_GAP_METRIC:-k3}
teacher_intervention_gap_distribution_topk=${TEACHER_INTERVENTION_GAP_DISTRIBUTION_TOPK:-16}
teacher_intervention_gap_aggregation=${TEACHER_INTERVENTION_GAP_AGGREGATION:-mean}
teacher_intervention_gap_threshold=${TEACHER_INTERVENTION_GAP_THRESHOLD:-0.15}
teacher_intervention_max_count=${TEACHER_INTERVENTION_MAX_COUNT:-1}
teacher_intervention_confirmation_mode=${TEACHER_INTERVENTION_CONFIRMATION_MODE:-none}
teacher_intervention_train_teacher=${TEACHER_INTERVENTION_TRAIN_TEACHER:-True}
teacher_intervention_loss_mode=${TEACHER_INTERVENTION_LOSS_MODE:-opd}
teacher_intervention_min_trainable_tokens=${TEACHER_INTERVENTION_MIN_TRAINABLE_TOKENS:-1}
teacher_intervention_temperature=${TEACHER_INTERVENTION_TEMPERATURE:-0.6}
teacher_intervention_top_p=${TEACHER_INTERVENTION_TOP_P:-0.95}
teacher_intervention_top_k=${TEACHER_INTERVENTION_TOP_K:-20}
teacher_intervention_prune_mode=${TEACHER_INTERVENTION_PRUNE_FUTURE_MODE:-${TEACHER_INTERVENTION_PRUNE_MODE:-adaptive_future}}
teacher_intervention_prune_future_window_tokens=${TEACHER_INTERVENTION_PRUNE_FUTURE_WINDOW_TOKENS:-64}
teacher_intervention_prune_future_min_valid_tokens=${TEACHER_INTERVENTION_PRUNE_FUTURE_MIN_VALID_TOKENS:-32}
teacher_intervention_prune_future_bad_window_persistence=${TEACHER_INTERVENTION_PRUNE_FUTURE_BAD_WINDOW_PERSISTENCE:-2}
teacher_intervention_prune_future_healthy_quantile=${TEACHER_INTERVENTION_PRUNE_FUTURE_HEALTHY_QUANTILE:-0.90}
teacher_intervention_prune_future_calibration_window_blocks=${TEACHER_INTERVENTION_PRUNE_FUTURE_CALIBRATION_WINDOW_BLOCKS:-2048}
teacher_intervention_prune_future_calibration_min_blocks=${TEACHER_INTERVENTION_PRUNE_FUTURE_CALIBRATION_MIN_BLOCKS:-128}
teacher_intervention_prune_future_max_healthy_blocks_per_sample=${TEACHER_INTERVENTION_PRUNE_FUTURE_MAX_HEALTHY_BLOCKS_PER_SAMPLE:-8}
teacher_intervention_prune_future_initial_threshold=${TEACHER_INTERVENTION_PRUNE_FUTURE_INITIAL_THRESHOLD:-null}
dmpd_collect_outcome_transitions=${DMPD_COLLECT_OUTCOME_TRANSITIONS:-True}
dmpd_outcome_cases_per_transition=${DMPD_OUTCOME_CASES_PER_TRANSITION:-8}
TEACHER_INTERVENTION_SKIP_CORRECT=${TEACHER_INTERVENTION_SKIP_CORRECT:-False}
TEACHER_INTERVENTION_PRUNE=${TEACHER_INTERVENTION_PRUNE:-True}

METHOD_HYDRA_OVERRIDES=(
    "distillation.intervention.enabled=True"
    "distillation.intervention.student_segment_length=${teacher_intervention_segment_length}"
    "distillation.intervention.teacher_segment_length=${teacher_intervention_teacher_length}"
    "distillation.intervention.gap_metric=${teacher_intervention_gap_metric}"
    "distillation.intervention.gap_distribution_topk=${teacher_intervention_gap_distribution_topk}"
    "distillation.intervention.gap_aggregation=${teacher_intervention_gap_aggregation}"
    "distillation.intervention.gap_threshold=${teacher_intervention_gap_threshold}"
    "distillation.intervention.max_interventions_per_sample=${teacher_intervention_max_count}"
    "distillation.intervention.confirmation_mode=${teacher_intervention_confirmation_mode}"
    "distillation.intervention.outcome_aware_candidate_selection_enabled=False"
    "distillation.intervention.train_on_teacher_segments=${teacher_intervention_train_teacher}"
    "distillation.intervention.teacher_segment_loss_mode=${teacher_intervention_loss_mode}"
    "distillation.intervention.min_trainable_tokens_after_repair=${teacher_intervention_min_trainable_tokens}"
    "distillation.intervention.teacher_temperature=${teacher_intervention_temperature}"
    "distillation.intervention.teacher_top_p=${teacher_intervention_top_p}"
    "distillation.intervention.teacher_top_k=${teacher_intervention_top_k}"
    "distillation.intervention.prune_mode=${teacher_intervention_prune_mode}"
    "distillation.intervention.prune_future_window_tokens=${teacher_intervention_prune_future_window_tokens}"
    "distillation.intervention.prune_future_min_valid_tokens=${teacher_intervention_prune_future_min_valid_tokens}"
    "distillation.intervention.prune_future_bad_window_persistence=${teacher_intervention_prune_future_bad_window_persistence}"
    "distillation.intervention.prune_future_healthy_quantile=${teacher_intervention_prune_future_healthy_quantile}"
    "distillation.intervention.prune_future_calibration_window_blocks=${teacher_intervention_prune_future_calibration_window_blocks}"
    "distillation.intervention.prune_future_calibration_min_blocks=${teacher_intervention_prune_future_calibration_min_blocks}"
    "distillation.intervention.prune_future_max_healthy_blocks_per_sample=${teacher_intervention_prune_future_max_healthy_blocks_per_sample}"
    "distillation.intervention.prune_future_initial_threshold=${teacher_intervention_prune_future_initial_threshold}"
    "distillation.diagnostics.collect_intervention_outcome_transitions=${dmpd_collect_outcome_transitions}"
    "distillation.diagnostics.outcome_case_samples_per_transition=${dmpd_outcome_cases_per_transition}"
)


capture_configuration() {
    TRAIN_DATASET=${TRAIN_DATASET:-dapo_math_17k}
    model_root=${MODEL_ROOT:-$DATASET_DIR/models}
    dataset_root=${DATASET_ROOT:-$DATASET_DIR/datasets}

    _configure_enable_thinking || exit $?

    enable_tool_call=${ENABLE_TOOL_CALL:-False}
    case "$enable_tool_call" in
        True | true | 1) enable_tool_call=True ;;
        False | false | 0) enable_tool_call=False ;;
        *)
            echo "ENABLE_TOOL_CALL must be True or False, got: $enable_tool_call" >&2
            return 2
            ;;
    esac

    sod_style=${SOD_STYLE:-False}
    case "$sod_style" in
        True | true | 1) sod_style=True ;;
        False | false | 0) sod_style=False ;;
        *)
            echo "SOD_STYLE must be True or False, got: $sod_style" >&2
            return 2
            ;;
    esac
    if [[ "$sod_style" == "True" && "$enable_tool_call" != "True" ]]; then
        echo "SOD_STYLE=True requires ENABLE_TOOL_CALL=True." >&2
        return 2
    fi
    if [[ "$sod_style" == "True" ]]; then
        case "$TRAIN_DATASET" in
            dapo_math_17k | deepmath_103k) ;;
            *)
                echo "SOD_STYLE=True currently supports only dapo_math_17k and deepmath_103k, got: $TRAIN_DATASET" >&2
                return 2
                ;;
        esac
    fi

    if [[ "$sod_style" == "True" ]]; then
        STUDENT_MODEL=${STUDENT_MODEL:-$model_root/Qwen/Qwen3-1.7B}
        TEACHER_MODEL=${TEACHER_MODEL:-$model_root/youngzhong/SOD-GRPO_teacher-4B}
        CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$STUDENT_MODEL}
    elif [[ "$enable_tool_call" == "True" || "$TRAIN_DATASET" == "nemotron_cascade_2_rl_mopd" ]]; then
        STUDENT_MODEL=${STUDENT_MODEL:-$model_root/Qwen/Qwen3-1.7B}
        TEACHER_MODEL=${TEACHER_MODEL:-$model_root/Qwen/Qwen3-8B}
        CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$STUDENT_MODEL}
    else
        STUDENT_MODEL=${STUDENT_MODEL:-$model_root/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
        TEACHER_MODEL=${TEACHER_MODEL:-$model_root/hbx/JustRL-DeepSeek-1.5B}
        CHAT_TEMPLATE_MODEL=${CHAT_TEMPLATE_MODEL:-$TEACHER_MODEL}
    fi

    python_sandbox_url=${PYTHON_SANDBOX_URL:-}
    if [[ "$sod_style" == "True" ]]; then
        tool_system_prompt_path=null
        tool_user_prompt_suffix_path=${TOOL_USER_PROMPT_SUFFIX_PATH:-$WORKSPACE/run/prompts/qwen3_math_tool_sod.txt}
        function_tool_path=${FUNCTION_TOOL_PATH:-$WORKSPACE/verl/tools/sod_code_interpreter_tool.py}
        tool_max_assistant_turns=${TOOL_MAX_ASSISTANT_TURNS:-16}
        tool_max_user_turns=${TOOL_MAX_USER_TURNS:-16}
        tool_max_response_chars=${TOOL_MAX_RESPONSE_CHARS:-256}
        tool_response_truncate_side=${TOOL_RESPONSE_TRUNCATE_SIDE:-middle}
        export PYTHON_SANDBOX_TIMEOUT=${PYTHON_SANDBOX_TIMEOUT:-60}
    else
        tool_system_prompt_path=${TOOL_SYSTEM_PROMPT_PATH:-$WORKSPACE/run/prompts/qwen3_math_tool_optional.txt}
        tool_user_prompt_suffix_path=${TOOL_USER_PROMPT_SUFFIX_PATH:-null}
        function_tool_path=${FUNCTION_TOOL_PATH:-$WORKSPACE/verl/tools/code_interpreter_tool.py}
        tool_max_assistant_turns=${TOOL_MAX_ASSISTANT_TURNS:-null}
        tool_max_user_turns=${TOOL_MAX_USER_TURNS:-null}
        tool_max_response_chars=${TOOL_MAX_RESPONSE_CHARS:-4096}
        tool_response_truncate_side=${TOOL_RESPONSE_TRUNCATE_SIDE:-middle}
        export PYTHON_SANDBOX_TIMEOUT=${PYTHON_SANDBOX_TIMEOUT:-10}
    fi

    case "$TRAIN_DATASET" in
        dapo_math_17k)
            default_train_file=${DAPO_MATH_TRAIN:-$dataset_root/open-r1/DAPO-Math-17k-Processed-verl--qwen3/train.parquet}
            default_valid_file=${DAPO_MATH_VALID:-$dataset_root/open-r1/DAPO-Math-17k-Processed-verl--qwen3/test.parquet}
            ;;
        deepmath_103k)
            default_train_file=${DEEPMATH_103K_TRAIN:-$dataset_root/zwhe99/DeepMath-103K-verl--qwen3/train.parquet}
            default_valid_file=${DEEPMATH_103K_VALID:-$dataset_root/zwhe99/DeepMath-103K-verl--qwen3/test.parquet}
            ;;
        nemotron_cascade_2_rl_mopd)
            default_train_file=${NEMOTRON_CASCADE_2_RL_MOPD_TRAIN:-$dataset_root/nvidia/Nemotron-Cascade-2-RL-data-MOPD-verl--qwen3/train.parquet}
            default_valid_file=${NEMOTRON_CASCADE_2_RL_MOPD_VALID:-$dataset_root/nvidia/Nemotron-Cascade-2-RL-data-MOPD-verl--qwen3/test.parquet}
            ;;
        *)
            echo "Unsupported TRAIN_DATASET=$TRAIN_DATASET. Use dapo_math_17k, deepmath_103k, or nemotron_cascade_2_rl_mopd." >&2
            return 2
            ;;
    esac
    train_files=${TRAIN_FILES:-"['$default_train_file']"}
    valid_files=${VALID_FILES:-"['$default_valid_file']"}

    NNODES=${NNODES:-$VC_WORKER_NUM}
    STUDENT_NNODES=${STUDENT_NNODES:-$((VC_WORKER_NUM - 1))}
    TEACHER_NNODES=${TEACHER_NNODES:-1}
    STUDENT_NGPUS_PER_NODE=${STUDENT_NGPUS_PER_NODE:-8}
    TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-8}
    student_tp=${STUDENT_TP:-2}
    student_gpu_mem_util=${STUDENT_GPU_MEM_UTIL:-0.6}
    teacher_tp=${TEACHER_TP:-2}
    teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.6}
    sp_size=${SP_SIZE:-4}
    TEACHER_REPLICAS=${TEACHER_REPLICAS:-$((TEACHER_NNODES * TEACHER_NGPUS_PER_NODE / teacher_tp))}

    if [[ "$sod_style" == "True" ]]; then
        max_prompt_length=${MAX_PROMPT_LENGTH:-2560}
        max_response_length=${MAX_RESPONSE_LENGTH:-20480}
    elif [[ "$enable_tool_call" == "True" ]]; then
        max_prompt_length=${MAX_PROMPT_LENGTH:-$((2 * 1024))}
        max_response_length=${MAX_RESPONSE_LENGTH:-$((23 * 1024))}
    else
        max_prompt_length=${MAX_PROMPT_LENGTH:-$((1 * 1024))}
        max_response_length=${MAX_RESPONSE_LENGTH:-$((23 * 1024))}
    fi
    sequence_token_budget=$((max_prompt_length + max_response_length + 1))
    max_num_tokens=${MAX_NUM_TOKENS:-${MAX_MODEL_LEN:-$sequence_token_budget}}
    if [[ "$TRAIN_DATASET" == "nemotron_cascade_2_rl_mopd" ]]; then
        default_rollout_max_model_len=32768
        default_teacher_max_model_len=$sequence_token_budget
    else
        default_rollout_max_model_len=$max_num_tokens
        default_teacher_max_model_len=$max_num_tokens
    fi
    rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-${STUDENT_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-${MAX_NUM_TOKENS:-$default_rollout_max_model_len}}}}
    teacher_max_model_len=${TEACHER_MAX_MODEL_LEN:-${MAX_MODEL_LEN:-${MAX_NUM_TOKENS:-$default_teacher_max_model_len}}}
    rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-2048}
    teacher_max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS:-2048}
    rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-128}
    teacher_max_num_seqs=${TEACHER_MAX_NUM_SEQS:-128}
    rollout_n=${ROLLOUT_N:-4}
    rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
    rollout_top_p=${ROLLOUT_TOP_P:-1.0}
    rollout_top_k=${ROLLOUT_TOP_K:--1}
    rollout_do_sample=${ROLLOUT_DO_SAMPLE:-True}
    val_temperature=${VAL_TEMPERATURE:-0.7}
    val_top_p=${VAL_TOP_P:-0.95}
    val_top_k=${VAL_TOP_K:-$rollout_top_k}
    val_do_sample=${VAL_DO_SAMPLE:-True}
    val_n=${VAL_N:-16}
    test_temperature=${TEST_TEMPERATURE:-0.7}
    test_top_p=${TEST_TOP_P:-0.95}
    test_top_k=${TEST_TOP_K:-$rollout_top_k}
    test_do_sample=${TEST_DO_SAMPLE:-True}
    test_benchmark_n=${TEST_BENCHMARK_N:-${TEST_N:-16}}

    actor_lr=${ACTOR_LR:-1e-6}
    train_batch_size=${TRAIN_BATCH_SIZE:-64}
    ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
    ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( ((max_num_tokens + sp_size - 1) / sp_size + 127) / 128 * 128 ))}
    train_max_samples=${TRAIN_MAX_SAMPLES:--1}
    val_max_samples=${VAL_MAX_SAMPLES:-256}
    val_batch_size=${VAL_BATCH_SIZE:-256}
    test_benchmark_root=${TEST_BENCHMARK_ROOT:-$dataset_root}
    if [[ "$TRAIN_DATASET" == "nemotron_cascade_2_rl_mopd" ]]; then
        validation_reward_data_source=${VALIDATION_REWARD_DATA_SOURCE:-nemotron_cascade_mopd}
        default_test_benchmark_suite=nemotron_cascade
        default_test_benchmarks="[]"
        export NLTK_DATA=${NEMOTRON_NLTK_DATA:-${NLTK_DATA:-$dataset_root/nltk_data}}
    else
        validation_reward_data_source=${VALIDATION_REWARD_DATA_SOURCE:-math_verify}
        default_test_benchmark_suite=
        default_test_benchmarks="['math-ai/aime24','math-ai/aime25','math-ai/amc23','meituan-longcat/AMO-Bench']"
    fi
    test_benchmark_suite=${TEST_BENCHMARK_SUITE:-$default_test_benchmark_suite}
    test_benchmark_seed=${TEST_BENCHMARK_SEED:-42}
    test_benchmarks=${TEST_BENCHMARKS:-$default_test_benchmarks}
    test_benchmark_reward_data_source=${TEST_BENCHMARK_REWARD_DATA_SOURCE:-math_verify}
    test_benchmark_max_samples=${TEST_BENCHMARK_MAX_SAMPLES:--1}
    test_benchmark_batch_size=${TEST_BENCHMARK_BATCH_SIZE:-256}
    data_seed=${DATA_SEED:--42}

    distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
    top_k_strategy=${TOP_K_STRATEGY:-only_stu_rethinking_style}
    reward_weight_mode=${REWARD_WEIGHT_MODE:-student_p}
    use_task_rewards=${USE_TASK_REWARDS:-False}
    rlvr_loss_coef=${RLVR_LOSS_COEF:-1.0}
    rlvr_adv_estimator=${RLVR_ADV_ESTIMATOR:-grpo}
    rlvr_credit_mode=${RLVR_CREDIT_MODE:-no_intervention_only}
    use_policy_gradient=${USE_POLICY_GRADIENT:-True}
    distillation_topk=${DISTILLATION_TOPK:-16}
    teacher_topk=${TEACHER_TOPK:-128}
    distillation_loss_max_clamp=${DISTILLATION_LOSS_MAX_CLAMP:-10.0}
    distillation_log_prob_min_clamp=${DISTILLATION_LOG_PROB_MIN_CLAMP:--10.0}
    teacher_intervention_rejection_token_cap=${TEACHER_INTERVENTION_REJECTION_TOKEN_CAP:-10.0}
    teacher_intervention_skip_correct=${TEACHER_INTERVENTION_SKIP_CORRECT:-False}
    teacher_intervention_prune=${TEACHER_INTERVENTION_PRUNE:-False}
    opd_diagnostics_enabled=${OPD_DIAGNOSTICS_ENABLED:-True}
    opd_diagnostics_collect_topk_overlap=${OPD_DIAGNOSTICS_COLLECT_TOPK_OVERLAP:-True}
    opd_diagnostics_collect_rethinking_opd=${OPD_DIAGNOSTICS_COLLECT_RETHINKING_OPD:-True}
    opd_diagnostics_collect_repetition=${OPD_DIAGNOSTICS_COLLECT_REPETITION:-True}
    opd_diagnostics_collect_train_text_zlib_repetition=${OPD_DIAGNOSTICS_COLLECT_TRAIN_TEXT_ZLIB_REPETITION:-True}
    opd_diagnostics_topk=${OPD_DIAGNOSTICS_TOPK:-16}
    opd_diagnostics_repetition_tail_chars=${OPD_DIAGNOSTICS_REPETITION_TAIL_CHARS:-10000}
    opd_diagnostics_repetition_comp_ratio_threshold=${OPD_DIAGNOSTICS_REPETITION_COMP_RATIO_THRESHOLD:-10.0}
    opd_diagnostics_repetition_tail_token_window=${OPD_DIAGNOSTICS_REPETITION_TAIL_TOKEN_WINDOW:-4096}
    opd_diagnostics_position_bin_size=${OPD_DIAGNOSTICS_POSITION_BIN_SIZE:-512}
    opd_diagnostics_log_interval=${OPD_DIAGNOSTICS_LOG_INTERVAL:-1}
    opd_diagnostics_topk_log_interval=${OPD_DIAGNOSTICS_TOPK_LOG_INTERVAL:-10}
    opd_diagnostics_async_metrics=${OPD_DIAGNOSTICS_ASYNC_METRICS:-True}
    opd_diagnostics_heatmap_interval=${OPD_DIAGNOSTICS_HEATMAP_INTERVAL:-10}
    opd_diagnostics_heatmap_history_steps=${OPD_DIAGNOSTICS_HEATMAP_HISTORY_STEPS:-80}
    opd_diagnostics_log_heatmap=${OPD_DIAGNOSTICS_LOG_HEATMAP:-True}
    opd_diagnostics_high_gap_threshold=${OPD_DIAGNOSTICS_HIGH_GAP_THRESHOLD:-1.0}
    opd_diagnostics_collect_intervention_boundary_metrics=${OPD_DIAGNOSTICS_COLLECT_INTERVENTION_BOUNDARY_METRICS:-True}
    opd_diagnostics_intervention_boundary_text_samples_per_step=${OPD_DIAGNOSTICS_INTERVENTION_BOUNDARY_TEXT_SAMPLES_PER_STEP:-8}
    opd_diagnostics_intervention_boundary_text_context_tokens=${OPD_DIAGNOSTICS_INTERVENTION_BOUNDARY_TEXT_CONTEXT_TOKENS:-32}
    opd_diagnostics_bad_tail_min_position=${OPD_DIAGNOSTICS_BAD_TAIL_MIN_POSITION:-12000}
    opd_diagnostics_bad_tail_deep_position=${OPD_DIAGNOSTICS_BAD_TAIL_DEEP_POSITION:-16000}
    opd_diagnostics_bad_tail_extreme_position=${OPD_DIAGNOSTICS_BAD_TAIL_EXTREME_POSITION:-20000}
    opd_diagnostics_bad_tail_overlap_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_OVERLAP_THRESHOLD:-0.5}
    opd_diagnostics_bad_tail_teacher_prob_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_TEACHER_PROB_THRESHOLD:-0.2}
    opd_diagnostics_bad_tail_gap_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_GAP_THRESHOLD:-1.0}
    opd_diagnostics_bad_tail_reward_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_REWARD_THRESHOLD:--2.0}
    opd_diagnostics_bad_tail_entropy_threshold=${OPD_DIAGNOSTICS_BAD_TAIL_ENTROPY_THRESHOLD:-4.0}

    adv_estimator=${ADV_ESTIMATOR:-token_reward_direct}
    distillation_use_actor_loss=${DISTILLATION_USE_ACTOR_LOSS:-False}
    use_kl_in_reward=${USE_KL_IN_REWARD:-False}
    filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS:-True}
    data_truncation=${DATA_TRUNCATION:-error}
    data_shuffle=${DATA_SHUFFLE:-False}
    enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-True}
    rollout_name=${ROLLOUT_NAME:-vllm}
    reward_manager_name=${REWARD_MANAGER_NAME:-naive}
    trainer_logger=${TRAINER_LOGGER:-'["console","wandb"]'}
    trainer_device=${TRAINER_DEVICE:-npu}
    _configure_in_training_evaluation || exit 1
    distillation_enabled=${DISTILLATION_ENABLED:-True}

    project_name=${PROJECT_NAME:-DMPD}
    experiment_timestamp=${EXPERIMENT_TIMESTAMP:-$(date +"%Y%m%d-%H%M%S")}
    experiment_base_name=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME:-opd}}
    experiment_name=${EXPERIMENT_FULL_NAME:-"${experiment_base_name}-${experiment_timestamp}"}
    default_local_dir=${DEFAULT_LOCAL_DIR:-$WORKSPACE/outputs/${experiment_name}}
    total_epochs=${TOTAL_EPOCHS:-1}
    save_freq=${SAVE_FREQ:-20}
    metric_logging_mode=${METRIC_LOGGING_MODE:-balanced}
    metric_logging_resume_policy=${METRIC_LOGGING_RESUME_POLICY:-strict}
    metric_logging_live_key_warning_threshold=${METRIC_LOGGING_LIVE_KEY_WARNING_THRESHOLD:-250}
    metric_logging_archive_enabled=${METRIC_LOGGING_ARCHIVE_ENABLED:-True}
    metric_logging_archive_local_dir=${METRIC_LOGGING_ARCHIVE_LOCAL_DIR:-null}
    metric_logging_archive_upload_to_wandb=${METRIC_LOGGING_ARCHIVE_UPLOAD_TO_WANDB:-auto}
    metric_logging_archive_finish_timeout_seconds=${METRIC_LOGGING_ARCHIVE_FINISH_TIMEOUT_SECONDS:-60}
    metric_logging_archive_shard_interval_steps=${METRIC_LOGGING_ARCHIVE_SHARD_INTERVAL_STEPS:-}
    if [[ -z "$metric_logging_archive_shard_interval_steps" ]]; then
        if [[ "$save_freq" =~ ^[1-9][0-9]*$ ]]; then
            metric_logging_archive_shard_interval_steps=$save_freq
        else
            metric_logging_archive_shard_interval_steps=20
        fi
    fi
    if [[ ! "$metric_logging_archive_shard_interval_steps" =~ ^[1-9][0-9]*$ ]]; then
        echo "METRIC_LOGGING_ARCHIVE_SHARD_INTERVAL_STEPS must be a positive integer, got: $metric_logging_archive_shard_interval_steps" >&2
        exit 1
    fi
    test_benchmarks_max_response_length=${TEST_BENCHMARKS_MAX_RESPONSE_LENGTH:-$max_response_length}
    log_test_generations=${LOG_TEST_GENERATIONS:-0}
    resume_required=${RESUME_REQUIRED:-False}

    local -a hydra_args=(
        "algorithm.adv_estimator=${adv_estimator}"
        "algorithm.use_kl_in_reward=${use_kl_in_reward}"
        "data.train_files=${train_files}"
        "data.val_files=${valid_files}"
        "data.train_batch_size=${train_batch_size}"
        "data.val_batch_size=${val_batch_size}"
        "data.train_max_samples=${train_max_samples}"
        "data.val_max_samples=${val_max_samples}"
        "data.validation_reward_data_source=${validation_reward_data_source}"
        "data.test_benchmarks.root=${test_benchmark_root}"
        "data.test_benchmarks.names=${test_benchmarks}"
        "data.test_benchmarks.seed=${test_benchmark_seed}"
        "data.test_benchmarks.batch_size=${test_benchmark_batch_size}"
        "data.test_benchmarks.max_samples_per_benchmark=${test_benchmark_max_samples}"
        "data.test_benchmarks.reward_data_source=${test_benchmark_reward_data_source}"
        "data.max_prompt_length=${max_prompt_length}"
        "data.max_response_length=${max_response_length}"
        "data.user_prompt_suffix_path=${tool_user_prompt_suffix_path}"
        "data.filter_overlong_prompts=${filter_overlong_prompts}"
        "data.truncation=${data_truncation}"
        "data.shuffle=${data_shuffle}"
        "data.seed=${data_seed}"
        "actor_rollout_ref.model.path=${STUDENT_MODEL}"
        "actor_rollout_ref.model.chat_template_from_model=${CHAT_TEMPLATE_MODEL}"
        "actor_rollout_ref.model.use_remove_padding=True"
        "actor_rollout_ref.model.enable_gradient_checkpointing=${enable_gradient_checkpointing}"
        "actor_rollout_ref.actor.use_torch_compile=False"
        "actor_rollout_ref.actor.optim.lr=${actor_lr}"
        "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
        "actor_rollout_ref.actor.use_dynamic_bsz=True"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}"
        "actor_rollout_ref.actor.fsdp_config.param_offload=True"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"
        "actor_rollout_ref.ref.fsdp_config.forward_prefetch=True"
        "actor_rollout_ref.rollout.name=${rollout_name}"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${student_tp}"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${student_gpu_mem_util}"
        "actor_rollout_ref.rollout.n=${rollout_n}"
        "actor_rollout_ref.rollout.temperature=${rollout_temperature}"
        "actor_rollout_ref.rollout.top_p=${rollout_top_p}"
        "actor_rollout_ref.rollout.top_k=${rollout_top_k}"
        "actor_rollout_ref.rollout.do_sample=${rollout_do_sample}"
        "actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}"
        "actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}"
        "actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}"
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
        "actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}"
        "actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}"
        "actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k}"
        "actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}"
        "actor_rollout_ref.rollout.val_kwargs.n=${val_n}"
        "actor_rollout_ref.rollout.test_kwargs.temperature=${test_temperature}"
        "actor_rollout_ref.rollout.test_kwargs.top_p=${test_top_p}"
        "actor_rollout_ref.rollout.test_kwargs.top_k=${test_top_k}"
        "actor_rollout_ref.rollout.test_kwargs.do_sample=${test_do_sample}"
        "actor_rollout_ref.rollout.test_kwargs.n=${test_benchmark_n}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.legacy.temperature=${test_temperature}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.legacy.top_p=${test_top_p}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.legacy.top_k=${test_top_k}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.legacy.do_sample=${test_do_sample}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.legacy.n=${test_benchmark_n}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.train_sampling.temperature=${rollout_temperature}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.train_sampling.top_p=${rollout_top_p}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.train_sampling.top_k=${rollout_top_k}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.train_sampling.do_sample=${rollout_do_sample}"
        "actor_rollout_ref.rollout.benchmark_sampling_profiles.train_sampling.n=${test_benchmark_n}"
        "reward.reward_manager.name=${reward_manager_name}"
        "trainer.balance_batch=True"
        "trainer.logger=${trainer_logger}"
        "trainer.metric_logging.mode=${metric_logging_mode}"
        "trainer.metric_logging.resume_policy=${metric_logging_resume_policy}"
        "trainer.metric_logging.live_key_warning_threshold=${metric_logging_live_key_warning_threshold}"
        "trainer.metric_logging.archive.enabled=${metric_logging_archive_enabled}"
        "trainer.metric_logging.archive.local_dir=${metric_logging_archive_local_dir}"
        "trainer.metric_logging.archive.shard_interval_steps=${metric_logging_archive_shard_interval_steps}"
        "trainer.metric_logging.archive.upload_to_wandb=${metric_logging_archive_upload_to_wandb}"
        "trainer.metric_logging.archive.finish_timeout_seconds=${metric_logging_archive_finish_timeout_seconds}"
        "trainer.project_name=${project_name}"
        "trainer.experiment_name=${experiment_name}"
        "trainer.n_gpus_per_node=${STUDENT_NGPUS_PER_NODE}"
        "trainer.nnodes=${STUDENT_NNODES}"
        "trainer.device=${trainer_device}"
        "trainer.default_local_dir=${default_local_dir}"
        "trainer.val_before_train=${val_before_train}"
        "trainer.test_before_train=${test_before_train}"
        "trainer.teacher_val_before_train=${teacher_val_before_train}"
        "trainer.teacher_val_max_samples=${val_max_samples}"
        "trainer.log_teacher_val_generations=${val_batch_size}"
        "trainer.log_test_generations=${log_test_generations}"
        "trainer.save_freq=${save_freq}"
        "trainer.test_freq=${test_freq}"
        "trainer.test_benchmarks_freq=${test_benchmarks_freq}"
        "trainer.test_benchmarks_max_response_length=${test_benchmarks_max_response_length}"
        "trainer.total_epochs=${total_epochs}"
        "distillation.enabled=${distillation_enabled}"
        "distillation.use_actor_loss=${distillation_use_actor_loss}"
        "distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}"
        "distillation.nnodes=${TEACHER_NNODES}"
        "distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}"
        "distillation.teacher_models.teacher_model.num_replicas=${TEACHER_REPLICAS}"
        "distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}"
        "distillation.teacher_models.teacher_model.inference.name=vllm"
        "distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}"
        "distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_model_len}"
        "distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${teacher_max_num_batched_tokens}"
        "distillation.teacher_models.teacher_model.inference.max_num_seqs=${teacher_max_num_seqs}"
        "distillation.distillation_loss.loss_mode=${distillation_loss_mode}"
        "distillation.distillation_loss.topk=${distillation_topk}"
        "distillation.distillation_loss.teacher_topk=${teacher_topk}"
        "distillation.distillation_loss.top_k_strategy=${top_k_strategy}"
        "distillation.distillation_loss.reward_weight_mode=${reward_weight_mode}"
        "distillation.distillation_loss.use_task_rewards=${use_task_rewards}"
        "distillation.distillation_loss.rlvr_loss_coef=${rlvr_loss_coef}"
        "distillation.distillation_loss.rlvr_adv_estimator=${rlvr_adv_estimator}"
        "distillation.distillation_loss.rlvr_credit_mode=${rlvr_credit_mode}"
        "distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}"
        "distillation.distillation_loss.loss_max_clamp=${distillation_loss_max_clamp}"
        "distillation.distillation_loss.log_prob_min_clamp=${distillation_log_prob_min_clamp}"
        "distillation.intervention.teacher_rejection_token_cap=${teacher_intervention_rejection_token_cap}"
        "distillation.intervention.skip_correct=${teacher_intervention_skip_correct}"
        "distillation.intervention.prune=${teacher_intervention_prune}"
        "distillation.diagnostics.enabled=${opd_diagnostics_enabled}"
        "distillation.diagnostics.collect_topk_overlap=${opd_diagnostics_collect_topk_overlap}"
        "distillation.diagnostics.collect_rethinking_opd_metrics=${opd_diagnostics_collect_rethinking_opd}"
        "distillation.diagnostics.collect_repetition_metrics=${opd_diagnostics_collect_repetition}"
        "distillation.diagnostics.collect_train_text_zlib_repetition=${opd_diagnostics_collect_train_text_zlib_repetition}"
        "distillation.diagnostics.topk=${opd_diagnostics_topk}"
        "distillation.diagnostics.repetition_tail_chars=${opd_diagnostics_repetition_tail_chars}"
        "distillation.diagnostics.repetition_comp_ratio_threshold=${opd_diagnostics_repetition_comp_ratio_threshold}"
        "distillation.diagnostics.repetition_tail_token_window=${opd_diagnostics_repetition_tail_token_window}"
        "distillation.diagnostics.position_bin_size=${opd_diagnostics_position_bin_size}"
        "distillation.diagnostics.log_interval=${opd_diagnostics_log_interval}"
        "distillation.diagnostics.topk_log_interval=${opd_diagnostics_topk_log_interval}"
        "distillation.diagnostics.async_metrics=${opd_diagnostics_async_metrics}"
        "distillation.diagnostics.heatmap_interval=${opd_diagnostics_heatmap_interval}"
        "distillation.diagnostics.heatmap_history_steps=${opd_diagnostics_heatmap_history_steps}"
        "distillation.diagnostics.log_wandb_heatmap=${opd_diagnostics_log_heatmap}"
        "distillation.diagnostics.high_gap_threshold=${opd_diagnostics_high_gap_threshold}"
        "distillation.diagnostics.collect_intervention_boundary_metrics=${opd_diagnostics_collect_intervention_boundary_metrics}"
        "distillation.diagnostics.intervention_boundary_text_samples_per_step=${opd_diagnostics_intervention_boundary_text_samples_per_step}"
        "distillation.diagnostics.intervention_boundary_text_context_tokens=${opd_diagnostics_intervention_boundary_text_context_tokens}"
        "distillation.diagnostics.bad_tail_min_position=${opd_diagnostics_bad_tail_min_position}"
        "distillation.diagnostics.bad_tail_deep_position=${opd_diagnostics_bad_tail_deep_position}"
        "distillation.diagnostics.bad_tail_extreme_position=${opd_diagnostics_bad_tail_extreme_position}"
        "distillation.diagnostics.bad_tail_overlap_threshold=${opd_diagnostics_bad_tail_overlap_threshold}"
        "distillation.diagnostics.bad_tail_teacher_prob_threshold=${opd_diagnostics_bad_tail_teacher_prob_threshold}"
        "distillation.diagnostics.bad_tail_gap_threshold=${opd_diagnostics_bad_tail_gap_threshold}"
        "distillation.diagnostics.bad_tail_reward_threshold=${opd_diagnostics_bad_tail_reward_threshold}"
        "distillation.diagnostics.bad_tail_entropy_threshold=${opd_diagnostics_bad_tail_entropy_threshold}"
    )

    if [[ -n "$test_benchmark_suite" ]]; then
        hydra_args+=("+test_benchmarks@data.test_benchmarks=${test_benchmark_suite}")
    fi

    if [[ "$enable_tool_call" == "True" ]]; then
        hydra_args+=(
            "data.system_prompt_path=${tool_system_prompt_path}"
            "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent"
            "actor_rollout_ref.rollout.multi_turn.enable=True"
            "actor_rollout_ref.rollout.multi_turn.function_tool_path=${function_tool_path}"
            "actor_rollout_ref.rollout.multi_turn.format=hermes"
            "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${tool_max_assistant_turns}"
            "actor_rollout_ref.rollout.multi_turn.max_user_turns=${tool_max_user_turns}"
            "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1"
            "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${tool_max_response_chars}"
            "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=${tool_response_truncate_side}"
            "actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable"
        )
    fi

    hydra_args+=("${METHOD_HYDRA_OVERRIDES[@]}")
    hydra_args+=("++data.apply_chat_template_kwargs.enable_thinking=${enable_thinking}")

printf "%s\0" "${hydra_args[@]}"
}
capture_configuration