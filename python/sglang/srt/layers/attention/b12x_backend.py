"""b12x SM120 paged attention backend for sglang."""

from __future__ import annotations

import os as _os

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from b12x.integration import B12XExecutionLane
    from b12x.integration.attention import PagedAttentionWorkspace
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_B12X_PAGE_SIZE = 64
_DECODE_GRAPH_METADATA_BUFFER_ATTRS = (
    "request_indices",
    "qo_tile_indices",
    "kv_tile_indices",
    "merge_indptr",
    "o_indptr",
    "kv_chunk_size_ptr",
    "kv_window_start_tokens",
    "total_num_rows_ptr",
    "block_valid_mask",
)


def _b12x_config_candidates(cfg):
    seen = set()

    def add(candidate):
        if candidate is None or id(candidate) in seen:
            return
        seen.add(id(candidate))
        yield candidate

    yield from add(cfg)
    get_text_config = getattr(cfg, "get_text_config", None)
    if callable(get_text_config):
        try:
            yield from add(get_text_config())
        except TypeError:
            pass
    yield from add(getattr(cfg, "text_config", None))


def _b12x_get_config_attr(cfg, names):
    for candidate in _b12x_config_candidates(cfg):
        for name in names:
            value = getattr(candidate, name, None)
            if value is not None:
                return value
    return None


def _b12x_graph_replay_func(name: str):
    try:
        from b12x.attention.paged import graph_replay
    except ImportError:
        return None
    return getattr(graph_replay, name, None)


@dataclass
class B12xForwardMetadata:
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    page_table: torch.Tensor
    swa_page_table: torch.Tensor | None
    mode: str
    use_cuda_graph: bool
    active_total_q: int
    graph_key: tuple[object, ...] | None = None
    req_pool_indices: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    decode_preprocess_in_graph: bool = False
    prepared_workspace_keys: frozenset[tuple[object, ...]] = frozenset()


@dataclass(frozen=True)
class _DecodeGraphPreparePlan:
    metadata_sources: tuple[PagedAttentionWorkspace, ...]


class B12xAttnBackend(AttentionBackend):
    """Paged attention backend using b12x SM120 kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        from b12x.integration.attention import PagedAttentionWorkspace

        self.workspace_cls = PagedAttentionWorkspace
        self._is_draft_backend = bool(getattr(model_runner, "is_draft_worker", False))
        self.page_size = model_runner.page_size
        if self.page_size != _B12X_PAGE_SIZE:
            raise ValueError(
                f"b12x attention backend requires page_size={_B12X_PAGE_SIZE}, got {self.page_size}"
            )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim_qk = model_runner.model_config.head_dim
        self.head_dim_vo = getattr(
            model_runner.model_config, "v_head_dim", self.head_dim_qk
        )
        self.swa_head_dim_qk = getattr(
            model_runner.model_config, "swa_head_dim", self.head_dim_qk
        )
        self.swa_head_dim_vo = getattr(
            model_runner.model_config, "swa_v_head_dim", self.head_dim_vo
        )
        self.q_dtype = model_runner.dtype
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.device = torch.device(model_runner.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.swa_kv_pool = (
            self.token_to_kv_pool
            if hasattr(self.token_to_kv_pool, "translate_loc_from_full_to_swa")
            and hasattr(self.token_to_kv_pool, "layers_mapping")
            else None
        )
        self.use_sliding_window_kv_pool = self.swa_kv_pool is not None
        self.server_args = model_runner.server_args
        self.max_running_requests = int(
            getattr(model_runner, "max_running_requests", 1)
        )
        self.max_context_len = model_runner.model_config.context_len
        self.max_pages_per_req = (
            self.max_context_len + self.page_size - 1
        ) // self.page_size
        self.verify_graph_max_cache_seqlen = int(
            _os.environ.get("B12X_VERIFY_GRAPH_MAX_CACHE_SEQLEN", "0")
        )
        self.kv_contract_layer_id = self._select_kv_contract_layer_id()
        model = getattr(model_runner, "model", None)
        self.has_attention_sinks = bool(
            getattr(model_runner.model_config, "has_attention_sinks", False)
        ) or self._model_has_attention_sinks(model)
        self.attention_layers = self._collect_attention_layers(
            model
        )
        self.eager_attention_layers = self._collect_eager_attention_layers()
        if self.attention_layers:
            self.num_q_heads = max(
                int(layer.tp_q_head_num) for layer in self.attention_layers
            )
            self.num_kv_heads = max(
                int(layer.tp_k_head_num) for layer in self.attention_layers
            )
        self.num_cache_pages = self._max_num_cache_pages()

        self.forward_metadata: Optional[B12xForwardMetadata] = None
        self.eager_workspaces: dict[tuple[object, ...], PagedAttentionWorkspace] = {}
        self.cuda_graph_workspaces: dict[
            tuple[object, ...], PagedAttentionWorkspace
        ] = {}
        self.cuda_graph_decode_metadata_sources: dict[
            tuple[object, ...], PagedAttentionWorkspace
        ] = {}
        self.cuda_graph_prepare_plans: dict[
            tuple[object, ...], _DecodeGraphPreparePlan
        ] = {}
        self.fp8_descale_cache: dict[
            tuple[int, int, float, float], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        self.graph_page_offsets = torch.arange(
            0,
            self.max_pages_per_req * self.page_size,
            self.page_size,
            dtype=torch.int64,
            device=self.device,
        )

        self.cuda_graph_cu_seqlens_q: Optional[torch.Tensor] = None
        self.cuda_graph_cache_seqlens: Optional[torch.Tensor] = None
        self.cuda_graph_page_table: Optional[torch.Tensor] = None
        self.cuda_graph_swa_page_table: Optional[torch.Tensor] = None
        self.cuda_graph_row_indices: Optional[torch.Tensor] = None
        self.cuda_graph_metadata_ready_event: Optional[torch.cuda.Event] = None
        self.b12x_lane: B12XExecutionLane = self._init_b12x_execution_lane(model_runner)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        mode = self._mode_from_forward_mode(forward_batch.forward_mode)
        cache_seqlens = self._build_cache_seqlens(forward_batch, bs, mode)
        cu_seqlens_q = self._build_eager_cu_seqlens_q(forward_batch, bs, mode)
        active_total_q = self._active_total_q_from_forward_batch(
            forward_batch, bs, mode
        )
        page_table, swa_page_table = self._build_page_tables(
            forward_batch.req_pool_indices[:bs], cache_seqlens
        )
        prepared_workspace_keys: frozenset[tuple[object, ...]] = frozenset()
        if mode in ("extend", "verify"):
            prepared_workspace_keys = self._prepare_eager_extend_workspaces(
                mode=mode,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                swa_page_table=swa_page_table,
                cu_seqlens_q=cu_seqlens_q,
                active_total_q=active_total_q,
            )
        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            swa_page_table=swa_page_table,
            mode=mode,
            use_cuda_graph=False,
            active_total_q=active_total_q,
            req_pool_indices=(
                forward_batch.req_pool_indices[:bs] if mode == "decode" else None
            ),
            prepared_workspace_keys=prepared_workspace_keys,
        )

    def requires_seq_lens_cpu_for_replay(
        self, forward_mode: Optional[ForwardMode] = None
    ) -> bool:
        del forward_mode
        return False

    def can_run_cuda_graph_replay(
        self, forward_batch: ForwardBatch, capture_forward_mode: ForwardMode
    ) -> bool:
        if not capture_forward_mode.is_target_verify():
            return True

        max_cache_seqlen = int(self.verify_graph_max_cache_seqlen)
        if max_cache_seqlen <= 0:
            return True

        bs = int(getattr(forward_batch, "batch_size", 0) or 0)
        if bs <= 0:
            return True

        tokens_per_req = int(
            getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", 0)
            or 0
        )
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            if torch.is_tensor(seq_lens_cpu):
                runtime_max = int(seq_lens_cpu[:bs].max().item())
            else:
                runtime_max = max(int(x) for x in seq_lens_cpu[:bs])
        else:
            seq_lens = getattr(forward_batch, "seq_lens", None)
            if seq_lens is None:
                return True
            runtime_max = int(seq_lens[:bs].max().item())

        return runtime_max + max(tokens_per_req, 0) <= max_cache_seqlen

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        del max_num_tokens
        self.cuda_graph_cu_seqlens_q = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cu_seqlens_q.copy_(
            torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.device)
        )
        self.cuda_graph_cache_seqlens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_page_table = torch.zeros(
            max_bs, self.max_pages_per_req, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_swa_page_table = (
            torch.zeros(
                max_bs, self.max_pages_per_req, dtype=torch.int32, device=self.device
            )
            if self.use_sliding_window_kv_pool
            else None
        )
        self.cuda_graph_row_indices = torch.arange(
            max_bs, dtype=torch.long, device=self.device
        )
        self.cuda_graph_decode_metadata_sources.clear()
        self.cuda_graph_prepare_plans.clear()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        assert encoder_lens is None, (
            "b12x backend does not support encoder-decoder models"
        )
        mode = self._mode_from_forward_mode(forward_mode)
        graph_key = (mode, bs)
        if mode == "decode":
            build_decode_page_table = False
            decode_preprocess_in_graph = True
            cache_seqlens, page_table, swa_page_table, cu_seqlens_q = (
                self._decode_cuda_graph_metadata_views(bs)
            )
        else:
            build_decode_page_table = True
            decode_preprocess_in_graph = False
            cache_seqlens, page_table, swa_page_table, cu_seqlens_q = (
                self._fill_cuda_graph_metadata(
                    bs=bs,
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    forward_mode=forward_mode,
                    spec_info=spec_info,
                    total_q_hint=num_tokens,
                )
            )
        self._prepare_cuda_graph_workspaces(
            graph_key,
            mode=mode,
            bs=bs,
            total_q_capacity=num_tokens,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            swa_page_table=swa_page_table,
            cu_seqlens_q=cu_seqlens_q,
            req_pool_indices=req_pool_indices[:bs] if mode == "decode" else None,
            decode_page_table_prebuilt=build_decode_page_table,
            decode_preprocess_in_graph=decode_preprocess_in_graph,
            reset_decode_preprocess_capture=decode_preprocess_in_graph,
        )
        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            swa_page_table=swa_page_table,
            mode=mode,
            use_cuda_graph=True,
            active_total_q=bs if mode == "decode" else num_tokens,
            graph_key=graph_key,
            req_pool_indices=req_pool_indices[:bs] if mode == "decode" else None,
            seq_lens=seq_lens[:bs] if mode == "decode" else None,
            decode_preprocess_in_graph=decode_preprocess_in_graph,
        )

    def init_forward_metadata_replay_cuda_graph_no_cpu(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        assert encoder_lens is None, (
            "b12x backend does not support encoder-decoder models"
        )
        mode = self._mode_from_forward_mode(forward_mode)
        graph_key = (mode, bs)
        if mode == "decode":
            decode_preprocess_in_graph = (
                self._decode_graph_prepare_plan_metadata_captured(graph_key)
            )
            build_decode_page_table = not decode_preprocess_in_graph
            if decode_preprocess_in_graph:
                cache_seqlens, page_table, swa_page_table, cu_seqlens_q = (
                    self._decode_cuda_graph_metadata_views(bs)
                )
            else:
                cache_seqlens, page_table, swa_page_table, cu_seqlens_q = (
                    self._fill_decode_cuda_graph_metadata(
                        bs=bs,
                        req_pool_indices=req_pool_indices,
                        seq_lens=seq_lens,
                        build_page_table=build_decode_page_table,
                    )
                )
        else:
            build_decode_page_table = True
            decode_preprocess_in_graph = False
            cache_seqlens, page_table, swa_page_table, cu_seqlens_q = (
                self._fill_cuda_graph_metadata(
                    bs=bs,
                    req_pool_indices=req_pool_indices,
                    seq_lens=seq_lens,
                    forward_mode=forward_mode,
                    spec_info=spec_info,
                    total_q_hint=(
                        self._captured_graph_total_q_capacity(graph_key) or seq_lens_sum
                    ),
                )
            )
        if mode == "decode":
            total_q_capacity = self._captured_graph_total_q_capacity(graph_key) or max(
                1, bs
            )
        else:
            total_q_capacity = self._captured_graph_total_q_capacity(graph_key)
            if total_q_capacity is None:
                total_q_capacity = self._cuda_graph_total_q_hint_from_host(
                    bs=bs,
                    seq_lens_sum=seq_lens_sum,
                    spec_info=spec_info,
                )

        self._prepare_cuda_graph_workspaces(
            graph_key,
            mode=mode,
            bs=bs,
            total_q_capacity=total_q_capacity,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            swa_page_table=swa_page_table,
            cu_seqlens_q=cu_seqlens_q,
            req_pool_indices=req_pool_indices[:bs] if mode == "decode" else None,
            decode_page_table_prebuilt=build_decode_page_table,
            decode_preprocess_in_graph=decode_preprocess_in_graph,
            reset_decode_preprocess_capture=False,
        )
        self._record_cuda_graph_metadata_ready_for_overlap(mode)
        if _os.environ.get("DFLASH_B12X_DUMP") == "1" and mode == "verify":
            self._dump_graph_verify_workspaces(
                cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q
            )
        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            swa_page_table=swa_page_table,
            mode=mode,
            use_cuda_graph=True,
            active_total_q=bs if mode == "decode" else total_q_capacity,
            graph_key=graph_key,
            req_pool_indices=req_pool_indices[:bs] if mode == "decode" else None,
            seq_lens=seq_lens[:bs] if mode == "decode" else None,
            decode_preprocess_in_graph=decode_preprocess_in_graph,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        del seq_lens_cpu
        self.init_forward_metadata_replay_cuda_graph_no_cpu(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

    def init_forward_metadata_replay_cuda_graph_with_cache_loc(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        del seq_lens_cpu
        self.init_forward_metadata_replay_cuda_graph_no_cpu(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )
        if out_cache_loc is None or not forward_mode.is_decode_or_idle():
            return
        md = self._require_forward_metadata("decode")
        if md.decode_preprocess_in_graph:
            return
        if out_cache_loc.shape[0] < bs:
            return
        self._sanitize_decode_graph_padding_metadata_from_cache_loc(
            md,
            out_cache_loc=out_cache_loc[:bs],
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cuda_graph_metadata_ready_event(self) -> Optional[torch.cuda.Event]:
        return self.cuda_graph_metadata_ready_event

    def on_after_cuda_graph_warmup(self) -> None:
        pass

    def _record_cuda_graph_metadata_ready_for_overlap(self, mode: str) -> None:
        if mode != "decode":
            return
        if getattr(self.server_args, "disable_overlap_schedule", False):
            return
        if self.device.type != "cuda":
            return
        if self.cuda_graph_metadata_ready_event is None:
            self.cuda_graph_metadata_ready_event = torch.cuda.Event(blocking=False)
        stream = torch.cuda.current_stream(self.device)
        self.cuda_graph_metadata_ready_event.record(stream)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._validate_supported_kwargs(kwargs)
        self._validate_layer_contract(layer)
        if save_kv_cache and k is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self._require_forward_metadata("decode")
        self._stage_decode_graph_preprocess_for_layer(md, layer, forward_batch)
        workspace = self._get_workspace(
            md, total_q=q.shape[0], layer=layer, has_sinks=sinks is not None
        )
        window_left = self._layer_window_left(layer)
        workspace.prepare(
            self._page_table_for_layer(md, layer),
            md.cache_seqlens,
            md.cu_seqlens_q,
            window_left=window_left,
        )

        k_cache, v_cache = self._get_paged_kv_buffers(
            forward_batch.token_to_kv_pool, layer.layer_id
        )
        q3 = q.view(q.shape[0], layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        output = torch.empty(
            q.shape[0],
            layer.tp_q_head_num,
            layer.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        k_descale, v_descale = self._get_descale_tensors(
            layer, md.cache_seqlens.shape[0]
        )
        prepare_decode_graph_metadata = self._decode_graph_run_prepare_metadata(
            md,
            layer,
            workspace,
            window_left=window_left,
        )
        out, _ = workspace.run(
            q3,
            k_cache,
            v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=sinks,
            prepare_decode_graph_metadata=prepare_decode_graph_metadata,
        )
        self._mark_decode_graph_metadata_captured(
            md,
            layer,
            workspace,
            window_left=window_left,
            prepared=prepare_decode_graph_metadata,
        )
        return out.view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._validate_supported_kwargs(kwargs)
        self._validate_layer_contract(layer)
        if save_kv_cache and k is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self._require_forward_metadata_any("extend", "verify")
        workspace = self._get_workspace(
            md, total_q=q.shape[0], layer=layer, has_sinks=sinks is not None
        )
        if md.use_cuda_graph:
            if not workspace.prepared:
                raise RuntimeError(
                    "b12x CUDA graph extend workspace was not prepared before graph "
                    "capture/replay"
                )
        else:
            workspace_key = self._eager_workspace_key(md.mode, layer)
            if workspace_key not in md.prepared_workspace_keys:
                raise RuntimeError(
                    "b12x eager extend workspace was not prepared during metadata "
                    f"initialization for key {workspace_key}"
                )
            if not workspace.prepared or int(workspace.plan.total_q) != int(q.shape[0]):
                raise RuntimeError(
                    "b12x eager extend workspace metadata is stale; expected "
                    f"total_q={q.shape[0]}"
                )

        if (
            _os.environ.get("DFLASH_B12X_DUMP") == "1"
            and md.mode == "verify"
            and layer.layer_id in (0, 1)
            and self._layer_is_causal(layer)
        ):
            self._dump_verify_workspace_metadata(md, workspace, layer)

        k_cache, v_cache = self._get_paged_kv_buffers(
            forward_batch.token_to_kv_pool, layer.layer_id
        )
        q3 = q.view(q.shape[0], layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        output = torch.empty(
            q.shape[0],
            layer.tp_q_head_num,
            layer.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        k_descale, v_descale = self._get_descale_tensors(
            layer, md.cache_seqlens.shape[0]
        )
        out, _ = workspace.run(
            q3,
            k_cache,
            v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
            attention_sink_bias=sinks,
        )
        return out.view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def _dump_graph_verify_workspaces(self, *, cache_seqlens, cu_seqlens_q) -> None:
        try:
            if torch.cuda.is_current_stream_capturing():
                return
            import torch.distributed as _dist

            if _dist.is_initialized() and _dist.get_rank() != 0:
                return
            for key, ws in getattr(self, "cuda_graph_workspaces", {}).items():
                if not key or key[0] != "verify":
                    continue
                self._dump_workspace_record(
                    ws,
                    tag={"key": tuple(str(k) for k in key)},
                    use_cuda_graph=True,
                    layer_id=-1,
                    mode="verify",
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                )
        except Exception as e:
            logger.warning("b12x graph verify metadata dump failed: %s", e)

    def _dump_verify_workspace_metadata(self, md, workspace, layer) -> None:
        try:
            if torch.cuda.is_current_stream_capturing():
                return
            import torch.distributed as _dist

            if _dist.is_initialized() and _dist.get_rank() != 0:
                return
            self._dump_workspace_record(
                workspace,
                tag={},
                use_cuda_graph=bool(md.use_cuda_graph),
                layer_id=int(layer.layer_id),
                mode=md.mode,
                cache_seqlens=md.cache_seqlens,
                cu_seqlens_q=md.cu_seqlens_q,
            )
        except Exception as e:
            logger.warning("b12x verify metadata dump failed: %s", e)

    def _dump_workspace_record(
        self,
        workspace,
        *,
        tag,
        use_cuda_graph,
        layer_id,
        mode,
        cache_seqlens,
        cu_seqlens_q,
    ) -> None:
        dump_dir = _os.environ.get("DFLASH_B12X_DUMP_DIR", "/cache/dflash_dumps")
        _os.makedirs(dump_dir, exist_ok=True)
        n = getattr(self, "_b12x_dump_counter", 0)
        if n > 2400:
            return
        self._b12x_dump_counter = n + 1

        def _t(x, limit=96):
            if x is None:
                return None
            return x.reshape(-1)[:limit].detach().cpu().clone()

        plan = getattr(workspace, "_plan", None)
        plan_info = {}
        if plan is not None:
            for f in (
                "total_q",
                "causal",
                "window_left",
                "split_kv",
                "cta_tile_q",
                "gqa_group_size",
                "kv_chunk_size",
                "page_table_shape",
                "num_qo_tiles",
                "new_batch_size",
            ):
                v = getattr(plan, f, None)
                if isinstance(v, torch.Tensor):
                    v = _t(v, 16)
                plan_info[f] = v
        rec = {
            "use_cuda_graph": bool(use_cuda_graph),
            "layer_id": layer_id,
            "mode": mode,
            "tag": tag,
            "is_draft": bool(getattr(self, "_is_draft_backend", False)),
            "cache_seqlens": _t(cache_seqlens, 16),
            "cu_seqlens_q": _t(cu_seqlens_q, 16),
            "ws_cache_seqlens": _t(getattr(workspace, "cache_seqlens", None), 16),
            "ws_cu_seqlens_q": _t(getattr(workspace, "cu_seqlens_q", None), 16),
            "page_table_row0": _t(getattr(workspace, "page_table", None), 16),
            "request_indices": _t(getattr(workspace, "request_indices", None)),
            "qo_tile_indices": _t(getattr(workspace, "qo_tile_indices", None)),
            "kv_tile_indices": _t(getattr(workspace, "kv_tile_indices", None)),
            "block_valid_mask": _t(getattr(workspace, "block_valid_mask", None)),
            "merge_indptr": _t(getattr(workspace, "merge_indptr", None), 32),
            "o_indptr": _t(getattr(workspace, "o_indptr", None), 32),
            "kv_chunk_size_ptr": _t(getattr(workspace, "kv_chunk_size_ptr", None), 4),
            "kv_window_start_tokens": _t(
                getattr(workspace, "kv_window_start_tokens", None), 16
            ),
            "total_num_rows_ptr": _t(getattr(workspace, "total_num_rows_ptr", None), 4),
            "plan": plan_info,
        }
        torch.save(rec, _os.path.join(dump_dir, f"b12x{n}.pt"))

    def _forward_encoder_only_extend(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        md: B12xForwardMetadata,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Handle DFlash draft non-causal block attention.

        B12X paged attention is causal-only for extend. DFlash draft layers are
        encoder-only and require each draft token to attend over the whole
        materialized draft context, including later tokens in the current block.
        """

        q3 = q.view(q.shape[0], layer.tp_q_head_num, layer.qk_head_dim).contiguous()
        k_flat = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_flat = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        output = torch.empty(
            q.shape[0],
            layer.tp_q_head_num,
            layer.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        cu_q = md.cu_seqlens_q
        req_pool_indices = forward_batch.req_pool_indices
        cache_seqlens = md.cache_seqlens
        req_to_token = self.req_to_token
        gqa_group = max(1, int(layer.tp_q_head_num) // max(1, int(layer.tp_k_head_num)))

        for i in range(int(req_pool_indices.shape[0])):
            q_start = int(cu_q[i].item())
            q_end = int(cu_q[i + 1].item())
            if q_end <= q_start:
                continue

            cache_len = int(cache_seqlens[i].item())
            if cache_len <= 0:
                output[q_start:q_end].zero_()
                continue

            req_idx = int(req_pool_indices[i].item())
            q_len = q_end - q_start
            window_left = self._layer_window_left(layer)
            if window_left >= 0:
                # Match DFlash's non-causal SWA semantics: keep a bounded left
                # context while still allowing every token in the current draft
                # block to see the whole block, including later mask tokens.
                kv_start = max(0, cache_len - q_len - int(window_left))
            else:
                kv_start = 0
            token_locs = req_to_token[req_idx, kv_start:cache_len].to(torch.long)
            k_req = k_flat.index_select(0, token_locs)
            v_req = v_flat.index_select(0, token_locs)
            if gqa_group > 1:
                k_req = k_req.repeat_interleave(gqa_group, dim=1)
                v_req = v_req.repeat_interleave(gqa_group, dim=1)

            q_req = q3[q_start:q_end].transpose(0, 1).unsqueeze(0)
            k_req = k_req.transpose(0, 1).unsqueeze(0)
            v_req = v_req.transpose(0, 1).unsqueeze(0)
            if sinks is None:
                out_req = torch.nn.functional.scaled_dot_product_attention(
                    q_req,
                    k_req,
                    v_req,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=float(layer.scaling),
                )
            else:
                scores = torch.matmul(
                    q_req.to(torch.float32), k_req.transpose(-2, -1).to(torch.float32)
                )
                scores.mul_(float(layer.scaling))
                sink_scores = sinks.to(device=scores.device, dtype=torch.float32).view(
                    1, -1, 1, 1
                )
                scores = torch.cat(
                    [
                        scores,
                        sink_scores.expand(
                            scores.shape[0], -1, scores.shape[2], 1
                        ),
                    ],
                    dim=-1,
                )
                probs = torch.softmax(scores, dim=-1)[..., :-1].to(v_req.dtype)
                out_req = torch.matmul(probs, v_req)
            output[q_start:q_end].copy_(out_req.squeeze(0).transpose(0, 1))

        return output.view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def _require_forward_metadata(self, expected_mode: str) -> B12xForwardMetadata:
        if self.forward_metadata is None:
            raise RuntimeError("b12x backend metadata has not been initialized")
        if self.forward_metadata.mode != expected_mode:
            raise RuntimeError(
                f"b12x backend expected {expected_mode} metadata, got {self.forward_metadata.mode}"
            )
        return self.forward_metadata

    def _require_forward_metadata_any(
        self, *expected_modes: str
    ) -> B12xForwardMetadata:
        if self.forward_metadata is None:
            raise RuntimeError("b12x backend metadata has not been initialized")
        if self.forward_metadata.mode not in expected_modes:
            expected_desc = "/".join(expected_modes)
            raise RuntimeError(
                f"b12x backend expected {expected_desc} metadata, got {self.forward_metadata.mode}"
            )
        return self.forward_metadata

    @torch._dynamo.disable
    def _stage_decode_graph_preprocess_for_layer(
        self,
        md: B12xForwardMetadata,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ) -> None:
        if not md.decode_preprocess_in_graph:
            self._sanitize_decode_graph_padding_metadata(md, forward_batch)
            return
        if not self._is_first_attention_layer(layer):
            return
        if md.req_pool_indices is None or md.seq_lens is None:
            raise RuntimeError("b12x decode graph preprocessing requires replay inputs")

        self._stage_decode_graph_metadata(
            md,
            req_pool_indices=md.req_pool_indices,
            seq_lens=md.seq_lens,
        )
        self._sanitize_decode_graph_padding_metadata(md, forward_batch)

    def _stage_decode_graph_metadata(
        self,
        md: B12xForwardMetadata,
        *,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        bs = md.cache_seqlens.shape[0]
        if (
            md.cache_seqlens.device.type == "cuda"
            and md.page_table.device == md.cache_seqlens.device
            and req_pool_indices.device == md.cache_seqlens.device
            and seq_lens.device == md.cache_seqlens.device
            and self.req_to_token.device == md.cache_seqlens.device
        ):
            swa_index_mapping = (
                None
                if md.swa_page_table is None
                else getattr(self.swa_kv_pool, "full_to_swa_index_mapping", None)
            )
            if (md.swa_page_table is None or swa_index_mapping is not None) and _b12x_graph_replay_func("stage_decode_cuda_graph_metadata") is not None:
                from b12x.attention.paged.graph_replay import (
                    stage_decode_cuda_graph_metadata,
                )

                stage_decode_cuda_graph_metadata(
                    req_to_token=self.req_to_token,
                    req_pool_indices=req_pool_indices[:bs],
                    seq_lens=seq_lens[:bs],
                    cache_seqlens=md.cache_seqlens,
                    cu_seqlens_q=md.cu_seqlens_q,
                    page_table=md.page_table,
                    swa_page_table=md.swa_page_table,
                    swa_index_mapping=swa_index_mapping,
                    page_size=self.page_size,
                )
                return

        md.cache_seqlens.copy_(seq_lens[:bs].to(torch.int32))
        md.cu_seqlens_q.copy_(
            torch.arange(0, bs + 1, dtype=torch.int32, device=md.cu_seqlens_q.device)
        )
        self._build_page_tables_into(
            req_pool_indices[:bs],
            md.cache_seqlens,
            md.page_table,
            md.swa_page_table,
            bs,
            max_pages=md.page_table.shape[1],
        )

    @torch._dynamo.disable
    def _decode_graph_run_prepare_metadata(
        self,
        md: B12xForwardMetadata,
        layer: RadixAttention,
        workspace: PagedAttentionWorkspace,
        *,
        window_left: int,
    ) -> bool | None:
        if not md.use_cuda_graph:
            return None
        if not md.decode_preprocess_in_graph:
            return False
        if self.device.type != "cuda" or not torch.cuda.is_current_stream_capturing():
            return False
        source = self._decode_graph_metadata_source(md, layer, window_left=window_left)
        if source is None or getattr(source, "_decode_graph_metadata_captured_in_graph", False):
            return False
        return self._workspace_shares_decode_graph_metadata(workspace, source)

    @torch._dynamo.disable
    def _mark_decode_graph_metadata_captured(
        self,
        md: B12xForwardMetadata,
        layer: RadixAttention,
        workspace: PagedAttentionWorkspace,
        *,
        window_left: int,
        prepared: bool | None,
    ) -> None:
        if not prepared:
            return
        if self.device.type != "cuda" or not torch.cuda.is_current_stream_capturing():
            return
        source = self._decode_graph_metadata_source(md, layer, window_left=window_left)
        if source is None:
            return
        if self._workspace_shares_decode_graph_metadata(workspace, source):
            source._decode_graph_metadata_captured_in_graph = True

    def _is_first_attention_layer(self, layer: RadixAttention) -> bool:
        return bool(self.attention_layers) and int(layer.layer_id) == int(
            self.attention_layers[0].layer_id
        )

    def _sanitize_decode_graph_padding_metadata(
        self, md: B12xForwardMetadata, forward_batch: ForwardBatch
    ) -> None:
        if not md.use_cuda_graph:
            return

        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None:
            return

        bs = md.cache_seqlens.shape[0]
        if out_cache_loc.shape[0] < bs:
            return
        out_cache_loc_swa = getattr(forward_batch, "out_cache_loc_swa", None)
        self._sanitize_decode_graph_padding_metadata_from_cache_loc(
            md,
            out_cache_loc=out_cache_loc[:bs],
            out_cache_loc_swa=out_cache_loc_swa[:bs]
            if out_cache_loc_swa is not None and out_cache_loc_swa.shape[0] >= bs
            else None,
        )

    def _sanitize_decode_graph_padding_metadata_from_cache_loc(
        self,
        md: B12xForwardMetadata,
        *,
        out_cache_loc: torch.Tensor,
        out_cache_loc_swa: Optional[torch.Tensor] = None,
    ) -> None:
        bs = md.cache_seqlens.shape[0]
        if out_cache_loc.shape[0] < bs:
            return
        out_cache_loc = out_cache_loc[:bs]
        if out_cache_loc_swa is not None:
            if out_cache_loc_swa.shape[0] < bs:
                out_cache_loc_swa = None
            else:
                out_cache_loc_swa = out_cache_loc_swa[:bs]

        if self._patch_decode_graph_padding_metadata_with_b12x(
            md,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
        ):
            return

        valid_rows = md.cache_seqlens.ne(
            self.get_cuda_graph_seq_len_fill_value()
        ) | out_cache_loc.ne(0)
        invalid_rows = valid_rows.logical_not()
        md.page_table.masked_fill_(invalid_rows[:, None], 0)
        if md.swa_page_table is not None:
            md.swa_page_table.masked_fill_(invalid_rows[:, None], 0)
        md.cache_seqlens.copy_(
            torch.where(valid_rows, md.cache_seqlens, torch.ones_like(md.cache_seqlens))
        )
        self._refresh_decode_graph_current_pages_from_cache_loc(
            md,
            valid_rows=valid_rows,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
        )

    def _patch_decode_graph_padding_metadata_with_b12x(
        self,
        md: B12xForwardMetadata,
        *,
        out_cache_loc: torch.Tensor,
        out_cache_loc_swa: Optional[torch.Tensor] = None,
    ) -> bool:
        if md.cache_seqlens.device.type != "cuda":
            return False
        if (
            md.page_table.device != md.cache_seqlens.device
            or out_cache_loc.device != md.cache_seqlens.device
        ):
            return False
        if md.swa_page_table is not None and md.swa_page_table.device != md.cache_seqlens.device:
            return False
        if out_cache_loc_swa is not None and out_cache_loc_swa.device != md.cache_seqlens.device:
            return False

        swa_index_mapping = None
        if md.swa_page_table is not None and out_cache_loc_swa is None:
            swa_index_mapping = getattr(self.swa_kv_pool, "full_to_swa_index_mapping", None)
            if swa_index_mapping is None or swa_index_mapping.device != md.cache_seqlens.device:
                return False

        if _b12x_graph_replay_func("patch_decode_cuda_graph_current_pages") is None:
            return False

        from b12x.attention.paged.graph_replay import (
            patch_decode_cuda_graph_current_pages,
        )

        patch_decode_cuda_graph_current_pages(
            cache_seqlens=md.cache_seqlens,
            page_table=md.page_table,
            out_cache_loc=out_cache_loc,
            page_size=self.page_size,
            fill_value=self.get_cuda_graph_seq_len_fill_value(),
            swa_page_table=md.swa_page_table,
            out_cache_loc_swa=out_cache_loc_swa,
            swa_index_mapping=swa_index_mapping,
        )
        return True

    def _refresh_decode_graph_current_pages_from_cache_loc(
        self,
        md: B12xForwardMetadata,
        valid_rows: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        out_cache_loc_swa: Optional[torch.Tensor] = None,
    ) -> None:
        if out_cache_loc is None and forward_batch is not None:
            out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None:
            return

        bs = md.cache_seqlens.shape[0]
        if out_cache_loc.shape[0] < bs or md.page_table.shape[1] <= 0:
            return

        row_indices = self.cuda_graph_row_indices
        if (
            row_indices is None
            or row_indices.device != md.page_table.device
            or row_indices.numel() < bs
        ):
            row_indices = torch.arange(
                bs, dtype=torch.long, device=md.page_table.device
            )
        else:
            row_indices = row_indices[:bs]

        logical_pages = torch.div(
            (md.cache_seqlens.to(torch.int64) - 1).clamp_min(0),
            self.page_size,
            rounding_mode="floor",
        ).clamp_max(md.page_table.shape[1] - 1)

        current_pages = torch.div(
            out_cache_loc[:bs].to(torch.int64),
            self.page_size,
            rounding_mode="floor",
        ).to(torch.int32)
        existing_pages = md.page_table[row_indices, logical_pages]
        md.page_table[row_indices, logical_pages] = torch.where(
            valid_rows, current_pages, existing_pages
        )

        if md.swa_page_table is None:
            return

        if out_cache_loc_swa is None and forward_batch is not None:
            out_cache_loc_swa = getattr(forward_batch, "out_cache_loc_swa", None)
        if out_cache_loc_swa is None:
            if self.swa_kv_pool is None:
                return
            out_cache_loc_swa = self.swa_kv_pool.translate_loc_from_full_to_swa(
                out_cache_loc[:bs]
            )
        else:
            out_cache_loc_swa = out_cache_loc_swa[:bs]
        if out_cache_loc_swa.shape[0] < bs:
            return

        current_swa_pages = torch.div(
            out_cache_loc_swa.to(torch.int64),
            self.page_size,
            rounding_mode="floor",
        ).to(torch.int32)
        existing_swa_pages = md.swa_page_table[row_indices, logical_pages]
        md.swa_page_table[row_indices, logical_pages] = torch.where(
            valid_rows, current_swa_pages, existing_swa_pages
        )

    def _get_workspace(
        self,
        md: B12xForwardMetadata,
        *,
        total_q: int,
        layer: RadixAttention,
        has_sinks: bool,
    ) -> PagedAttentionWorkspace:
        if md.use_cuda_graph:
            assert md.graph_key is not None
            workspace_key = self._cuda_graph_workspace_key(
                md.graph_key, layer, has_sinks
            )
            workspace = self.cuda_graph_workspaces.get(workspace_key)
            if workspace is None:
                raise RuntimeError(
                    f"missing captured b12x cuda-graph workspace for {workspace_key}"
                )
            return workspace
        workspace_key = self._eager_workspace_key(md.mode, layer)
        if md.mode in ("extend", "verify"):
            return self._get_or_create_eager_extend_workspace(mode=md.mode, layer=layer)
        workspace = self.eager_workspaces.get(workspace_key)
        if workspace is None:
            workspace = self._make_arena_workspace(
                mode=md.mode,
                head_dim_qk=layer.qk_head_dim,
                head_dim_vo=layer.v_head_dim,
                num_q_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                max_total_q=self._paged_decode_q_rows_capacity(),
                max_batch=self._paged_decode_batch_capacity(),
                max_page_table_width=self.max_pages_per_req,
                max_work_items=self._paged_decode_work_items_capacity(),
                max_partial_rows=self._paged_decode_partial_rows_capacity(),
                num_cache_pages=self._num_cache_pages_for_layer(layer),
                use_cuda_graph=False,
            )
            self.eager_workspaces[workspace_key] = workspace
        return workspace

    def _prepare_eager_extend_workspaces(
        self,
        *,
        mode: str,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        swa_page_table: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        active_total_q: int,
    ) -> frozenset[tuple[object, ...]]:
        if not self.eager_attention_layers:
            raise RuntimeError(
                "b12x backend could not find RadixAttention layers for eager extend"
            )

        md = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            swa_page_table=swa_page_table,
            mode=mode,
            use_cuda_graph=False,
            active_total_q=active_total_q,
        )
        prepared_keys: set[tuple[object, ...]] = set()
        for layer in self.eager_attention_layers:
            self._validate_layer_contract(layer)
            workspace_key = self._eager_workspace_key(mode, layer)
            workspace = self._get_or_create_eager_extend_workspace(
                mode=mode, layer=layer
            )
            workspace.prepare(
                self._page_table_for_layer(md, layer),
                cache_seqlens,
                cu_seqlens_q,
                causal=self._layer_is_causal(layer),
                window_left=self._layer_window_left(layer),
            )
            prepared_keys.add(workspace_key)
        return frozenset(prepared_keys)

    def _get_or_create_eager_extend_workspace(
        self,
        *,
        mode: str = "extend",
        layer: RadixAttention,
    ) -> PagedAttentionWorkspace:
        workspace_key = self._eager_workspace_key(mode, layer)
        workspace = self.eager_workspaces.get(workspace_key)
        if workspace is not None:
            return workspace

        if mode == "verify":
            total_q_capacity = self._eager_verify_total_q_capacity()
            batch_capacity = self._paged_decode_batch_capacity()
            max_work_items, max_partial_rows = self._paged_verify_graph_capacities(
                batch_capacity=batch_capacity,
                total_q_capacity=total_q_capacity,
                head_dim_qk=layer.qk_head_dim,
                head_dim_vo=layer.v_head_dim,
                num_q_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                window_left=self._layer_window_left(layer),
                num_cache_pages=self._num_cache_pages_for_layer(layer),
            )
        else:
            total_q_capacity = self._eager_extend_total_q_capacity()
            batch_capacity = self._eager_extend_batch_capacity(total_q_capacity)
            max_work_items = self.workspace_cls.eager_extend_work_items_capacity(
                max_total_q=total_q_capacity,
                num_q_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
            )
            max_partial_rows = 0
        # Prepared eager metadata must stay resident across all layers in the
        # stage, so these workspaces cannot share the joint arena scratch.
        workspace = self.workspace_cls.for_fixed_capacity(
            mode=mode,
            device=self.device,
            dtype=self.q_dtype,
            kv_dtype=self.kv_cache_dtype,
            num_q_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_dim_qk=layer.qk_head_dim,
            head_dim_vo=layer.v_head_dim,
            page_size=self.page_size,
            max_total_q=total_q_capacity,
            max_batch=batch_capacity,
            max_page_table_width=self.max_pages_per_req,
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
            num_cache_pages=self._num_cache_pages_for_layer(layer),
            use_cuda_graph=False,
        )
        self.eager_workspaces[workspace_key] = workspace
        return workspace

    def _eager_extend_total_q_capacity(self) -> int:
        chunked_prefill_size = int(
            getattr(self.server_args, "chunked_prefill_size", -1) or -1
        )
        if chunked_prefill_size > 0:
            return chunked_prefill_size
        return int(self.server_args.max_prefill_tokens)

    def _eager_verify_total_q_capacity(self) -> int:
        draft_tokens = int(
            getattr(self.server_args, "speculative_num_draft_tokens", None) or 1
        )
        return max(1, self._paged_decode_batch_capacity() * draft_tokens)

    def _moe_extend_total_q_capacity(self) -> int:
        max_prefill_tokens = max(
            1, int(getattr(self.server_args, "max_prefill_tokens", 1) or 1)
        )
        chunked_prefill_size = max(
            0, int(getattr(self.server_args, "chunked_prefill_size", 0) or 0)
        )
        decode_page_reserve = self._paged_decode_batch_capacity() * int(
            self.page_size
        )
        capacity = max_prefill_tokens + chunked_prefill_size + decode_page_reserve
        return ((capacity + int(self.page_size) - 1) // int(self.page_size)) * int(
            self.page_size
        )

    def _eager_extend_batch_capacity(self, total_q_capacity: int) -> int:
        configured_batch = self.server_args.prefill_max_requests
        if configured_batch is None:
            configured_batch = self.max_running_requests
        return max(1, min(int(configured_batch), int(total_q_capacity)))

    def _paged_decode_q_rows_capacity(self) -> int:
        return max(1, int(self.max_running_requests))

    def _paged_decode_batch_capacity(self) -> int:
        return max(1, int(self.max_running_requests))

    def _paged_decode_work_items_capacity(self) -> int:
        return self._paged_decode_batch_capacity() * self.max_pages_per_req

    def _paged_decode_partial_rows_capacity(self) -> int:
        return self._paged_decode_work_items_capacity()

    def _max_eager_extend_work_items_capacity(self, total_q_capacity: int) -> int:
        layers = self.attention_layers or [None]
        max_work_items = 1
        for layer in layers:
            num_q_heads = (
                self.num_q_heads if layer is None else int(layer.tp_q_head_num)
            )
            num_kv_heads = (
                self.num_kv_heads if layer is None else int(layer.tp_k_head_num)
            )
            max_work_items = max(
                max_work_items,
                self.workspace_cls.eager_extend_work_items_capacity(
                    max_total_q=total_q_capacity,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                ),
            )
        return max_work_items

    def _graph_extend_work_items_capacity(
        self,
        *,
        total_q_capacity: int,
        num_q_heads: int,
        num_kv_heads: int,
    ) -> int:
        eager_work_items = self.workspace_cls.eager_extend_work_items_capacity(
            max_total_q=total_q_capacity,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
        )
        return max(
            eager_work_items,
            self._graph_block_valid_capacity(num_kv_heads=num_kv_heads),
        )

    def _max_graph_extend_work_items_capacity(self, total_q_capacity: int) -> int:
        layers = self.attention_layers or [None]
        max_work_items = 1
        for layer in layers:
            num_q_heads = (
                self.num_q_heads if layer is None else int(layer.tp_q_head_num)
            )
            num_kv_heads = (
                self.num_kv_heads if layer is None else int(layer.tp_k_head_num)
            )
            max_work_items = max(
                max_work_items,
                self._graph_extend_work_items_capacity(
                    total_q_capacity=total_q_capacity,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                ),
            )
        return max_work_items

    def _graph_block_valid_capacity(self, *, num_kv_heads: int) -> int:
        # B12X graph-mode extend pads block_valid_mask to the resident graph grid.
        # Keep this in sync with the planner's default graph CTA policy.
        graph_ctas_per_sm = 2
        num_sms = int(
            torch.cuda.get_device_properties(self.device).multi_processor_count
        )
        return max(1, (num_sms * graph_ctas_per_sm) // max(1, int(num_kv_heads)))

    def _max_paged_verify_capacities(
        self, *, batch_capacity: int, total_q_capacity: int
    ) -> tuple[int, int]:
        layers = self.attention_layers or [None]
        max_work_items = 1
        max_partial_rows = 1
        for layer in layers:
            if layer is None:
                head_dim_qk = max(int(self.head_dim_qk), int(self.swa_head_dim_qk))
                head_dim_vo = max(int(self.head_dim_vo), int(self.swa_head_dim_vo))
                num_q_heads = self.num_q_heads
                num_kv_heads = self.num_kv_heads
                window_left = -1
                num_cache_pages = self.num_cache_pages
            else:
                head_dim_qk = int(layer.qk_head_dim)
                head_dim_vo = int(layer.v_head_dim)
                num_q_heads = int(layer.tp_q_head_num)
                num_kv_heads = int(layer.tp_k_head_num)
                window_left = self._layer_window_left(layer)
                num_cache_pages = self._num_cache_pages_for_layer(layer)
            work_items, partial_rows = self._paged_verify_graph_capacities(
                batch_capacity=batch_capacity,
                total_q_capacity=total_q_capacity,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                window_left=window_left,
                num_cache_pages=num_cache_pages,
            )
            max_work_items = max(max_work_items, work_items)
            max_partial_rows = max(max_partial_rows, partial_rows)
        return max_work_items, max_partial_rows

    def _get_or_create_graph_workspace(
        self,
        graph_key: tuple[object, ...],
        *,
        mode: str,
        total_q_capacity: int,
        bs: int,
        head_dim_qk: int,
        head_dim_vo: int,
        num_q_heads: int,
        num_kv_heads: int,
        causal: bool,
        window_left: int,
        num_cache_pages: int,
        runtime_page_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> PagedAttentionWorkspace:
        workspace = self.cuda_graph_workspaces.get(graph_key)
        if workspace is None:
            if mode == "decode":
                max_batch = max(1, int(bs))
                max_page_table_width = self.max_pages_per_req
                max_work_items = max(
                    max_batch * self.max_pages_per_req,
                    self._graph_block_valid_capacity(num_kv_heads=num_kv_heads),
                )
                max_partial_rows = max_work_items
            elif mode == "verify":
                max_batch = max(1, int(bs))
                max_page_table_width = self.max_pages_per_req
                max_work_items, max_partial_rows = self._paged_verify_graph_capacities(
                    batch_capacity=max_batch,
                    total_q_capacity=total_q_capacity,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    window_left=window_left,
                    num_cache_pages=num_cache_pages,
                )
            else:
                max_batch = max(1, int(bs))
                max_page_table_width = self.max_pages_per_req
                max_work_items = self._graph_extend_work_items_capacity(
                    total_q_capacity=total_q_capacity,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                )
                max_partial_rows = 0
            if mode in ("extend", "verify"):
                # Graph replay updates metadata before launch. Keep one resident
                # metadata buffer per graph attention group rather than sharing
                # the joint arena across full/SWA groups.
                workspace = self.workspace_cls.for_fixed_capacity(
                    mode=mode,
                    device=self.device,
                    dtype=self.q_dtype,
                    kv_dtype=self.kv_cache_dtype,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    page_size=self.page_size,
                    max_total_q=total_q_capacity,
                    max_batch=max_batch,
                    max_page_table_width=max_page_table_width,
                    max_work_items=max_work_items,
                    max_partial_rows=max_partial_rows,
                    num_cache_pages=num_cache_pages,
                    use_cuda_graph=True,
                )
            else:
                workspace = self._make_arena_workspace(
                    mode=mode,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    max_total_q=total_q_capacity,
                    max_batch=max_batch,
                    max_page_table_width=max_page_table_width,
                    max_work_items=max_work_items,
                    max_partial_rows=max_partial_rows,
                    num_cache_pages=num_cache_pages,
                    use_cuda_graph=True,
                )
            self._prime_graph_workspace_capacity(
                workspace,
                mode=mode,
                bs=bs,
                total_q_capacity=total_q_capacity,
                causal=causal,
                window_left=window_left,
                num_cache_pages=num_cache_pages,
                runtime_page_table=runtime_page_table,
                cu_seqlens_q=cu_seqlens_q,
            )
            self.cuda_graph_workspaces[graph_key] = workspace
            return workspace
        if workspace.total_q_capacity != total_q_capacity:
            raise ValueError(
                "b12x backend currently expects one cuda-graph q-capacity per (mode, batch) bucket"
            )
        return workspace

    def _paged_verify_graph_capacities(
        self,
        *,
        batch_capacity: int,
        total_q_capacity: int,
        head_dim_qk: int,
        head_dim_vo: int,
        num_q_heads: int,
        num_kv_heads: int,
        window_left: int,
        num_cache_pages: int,
    ) -> tuple[int, int]:
        q_tile_capacity = self.workspace_cls.eager_extend_work_items_capacity(
            max_total_q=total_q_capacity,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
        )
        max_effective_kv_pages = max(
            1, min(self.max_pages_per_req, int(num_cache_pages))
        )
        if window_left >= 0:
            max_window_pages = max(
                1,
                (int(window_left) + self.page_size + self.page_size - 1)
                // self.page_size,
            )
            max_effective_kv_pages = min(max_effective_kv_pages, max_window_pages)

        max_chunks_per_req = max_effective_kv_pages
        graph_ctas_per_sm = 2
        gqa_group_size = max(1, int(num_q_heads) // max(1, int(num_kv_heads)))
        try:
            from b12x.attention.paged.tuning import get_decode_graph_policy
            from b12x.attention.paged.graph_replay import (
                summarize_decode_chunk_pages_lut,
            )
            from b12x.attention.paged.planner import build_decode_chunk_pages_lut

            kv_dtype_key = None
            if self.kv_cache_dtype == torch.bfloat16:
                kv_dtype_key = "bf16"
            elif self.kv_cache_dtype == torch.float16:
                kv_dtype_key = "fp16"
            elif self.kv_cache_dtype == torch.float8_e4m3fn:
                kv_dtype_key = "fp8_e4m3fn"
            regime = (
                "decode"
                if (
                    self.page_size == 64
                    and head_dim_qk == 256
                    and head_dim_vo == 256
                    and gqa_group_size == 8
                )
                else f"decode_qk{head_dim_qk}_vo{head_dim_vo}_gqa{gqa_group_size}"
            )
            _, max_chunks_per_req = summarize_decode_chunk_pages_lut(
                build_decode_chunk_pages_lut(
                    q_dtype=self.q_dtype,
                    kv_dtype=self.kv_cache_dtype,
                    batch=max(1, int(total_q_capacity)),
                    page_size=self.page_size,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    gqa_group_size=gqa_group_size,
                    max_effective_kv_pages=max_effective_kv_pages,
                )
            )
            if kv_dtype_key is not None:
                graph_ctas_per_sm = int(
                    get_decode_graph_policy(
                        kv_dtype=kv_dtype_key,
                        regime=regime,
                        batch=max(1, int(total_q_capacity)),
                    ).graph_ctas_per_sm
                )
        except (ImportError, KeyError, TypeError, ValueError):
            pass

        graph_split_capacity = (
            int(torch.cuda.get_device_properties(self.device).multi_processor_count)
            * int(graph_ctas_per_sm)
        ) // max(1, int(num_kv_heads))
        batch_capacity = max(1, int(batch_capacity))
        q_rows_per_req = max(
            1,
            (int(total_q_capacity) + batch_capacity - 1) // batch_capacity,
        )
        q_tiles_per_req = max(1, (q_rows_per_req * gqa_group_size + 15) // 16)
        # The verify graph planner may ignore the decode LUT and pick the
        # smallest split size that fills the resident graph work-item budget.
        graph_budget_chunks_per_req = max(
            1,
            int(graph_split_capacity) // max(1, batch_capacity * q_tiles_per_req),
        )
        max_work_items = max(1, int(q_tile_capacity) * int(max_chunks_per_req))
        max_work_items = max(max_work_items, graph_split_capacity)
        max_partial_rows = max(
            1,
            int(total_q_capacity)
            * max(int(max_chunks_per_req), int(graph_budget_chunks_per_req)),
        )
        return max_work_items, max_partial_rows

    def _init_b12x_execution_lane(self, model_runner: ModelRunner):
        from b12x.integration import (
            B12XJointArenaSpec,
            ensure_b12x_execution_lane_arena,
            get_b12x_execution_lane,
        )

        paged_caps = self._build_paged_attention_arena_caps()
        moe_caps = self._build_b12x_moe_arena_caps(model_runner)
        spec = B12XJointArenaSpec(
                device=self.device,
                paged_attention_caps=paged_caps,
                moe_caps=moe_caps,
            )
        try:
            lane = ensure_b12x_execution_lane_arena(spec)
        except RuntimeError:
            if not getattr(model_runner, "is_draft_worker", False):
                raise
            lane = get_b12x_execution_lane(
                self.device, create_standalone_moe_pool=False
            )
            if lane is None or lane.arena is None:
                raise
        if lane.arena is None or lane.arena.paged_attention_arena is None:
            raise RuntimeError(
                "b12x execution lane was allocated without paged attention arena"
            )
        return lane

    def _build_paged_attention_arena_caps(self):
        from b12x.integration.attention import PagedAttentionArenaCaps

        extend_total_q = self._eager_extend_total_q_capacity()
        extend_batch = self._eager_extend_batch_capacity(extend_total_q)
        decode_q = self._paged_decode_q_rows_capacity()
        max_total_q = max(extend_total_q, decode_q)
        max_batch = max(extend_batch, self._paged_decode_batch_capacity())
        extend_work_items = self._max_graph_extend_work_items_capacity(extend_total_q)
        decode_work_items = self._paged_decode_work_items_capacity()
        verify_total_q = self._eager_verify_total_q_capacity()
        verify_batch = self._paged_decode_batch_capacity()
        verify_work_items, verify_partial_rows = self._max_paged_verify_capacities(
            batch_capacity=verify_batch,
            total_q_capacity=verify_total_q,
        )
        return PagedAttentionArenaCaps(
            device=self.device,
            dtype=self.q_dtype,
            kv_dtype=self.kv_cache_dtype,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=max(int(self.head_dim_qk), int(self.swa_head_dim_qk)),
            max_head_dim_vo=max(int(self.head_dim_vo), int(self.swa_head_dim_vo)),
            page_size=self.page_size,
            max_total_q=max(max_total_q, verify_total_q),
            max_batch=max_batch,
            max_page_table_width=self.max_pages_per_req,
            max_work_items=max(extend_work_items, decode_work_items, verify_work_items),
            max_partial_rows=max(
                self._paged_decode_partial_rows_capacity(), verify_partial_rows
            ),
        )

    def _build_b12x_moe_arena_caps(self, model_runner: ModelRunner):
        from b12x.integration import B12XMoEArenaCaps
        from sglang.srt.distributed import get_tensor_model_parallel_world_size
        from sglang.srt.layers.moe import get_moe_runner_backend

        if not get_moe_runner_backend().is_b12x():
            return None

        cfg = model_runner.model_config.hf_config
        weight_E = _b12x_get_config_attr(
            cfg, ("n_routed_experts", "num_experts", "num_local_experts")
        )
        hidden_size = _b12x_get_config_attr(cfg, ("hidden_size",))
        intermediate_size = _b12x_get_config_attr(
            cfg, ("moe_intermediate_size", "intermediate_size")
        )
        num_topk = _b12x_get_config_attr(
            cfg,
            (
                "num_experts_per_tok",
                "top_k",
                "num_experts_per_token",
                "router_topk",
            ),
        )
        missing = [
            name
            for name, value in (
                ("n_routed_experts/num_experts/num_local_experts", weight_E),
                ("hidden_size", hidden_size),
                ("moe_intermediate_size/intermediate_size", intermediate_size),
                (
                    "num_experts_per_tok/top_k/num_experts_per_token/router_topk",
                    num_topk,
                ),
            )
            if value is None
        ]
        if missing:
            if getattr(model_runner, "is_draft_worker", False):
                return None
            raise ValueError(
                "b12x joint arena cannot size MoE workspace; missing config fields: "
                + ", ".join(missing)
            )

        tp_size = max(1, int(get_tensor_model_parallel_world_size()))
        intermediate_size = int(intermediate_size)
        if intermediate_size % tp_size != 0:
            raise ValueError(
                "b12x joint arena expected MoE intermediate_size to be divisible "
                f"by tensor parallel size, got intermediate_size={intermediate_size}, tp_size={tp_size}"
            )

        extend_total_q = self._eager_extend_total_q_capacity()
        decode_q = self._paged_decode_q_rows_capacity()
        verify_total_q = self._eager_verify_total_q_capacity()
        moe_extend_total_q = self._moe_extend_total_q_capacity()
        # Attention scratch can cap eager extend at chunked_prefill_size, but
        # the MoE core may still materialize for SGLang's max_prefill_tokens
        # bucket during target forward/verify paths. SGLang's paged decode
        # admission can also carry an adjacent chunk-sized extend bucket plus
        # one page per running decode request. Size the shared MoE arena for
        # that combined extend bucket so a later larger plan cannot outgrow it.
        max_prefill_tokens = max(
            1, int(getattr(self.server_args, "max_prefill_tokens", 1) or 1)
        )
        core_token_counts = (
            extend_total_q,
            decode_q,
            verify_total_q,
            max_prefill_tokens,
            moe_extend_total_q,
        )
        # Size the shared arena with the w4a16 layout: the FP4 MoE runtime
        # resolves a TPW4A16Workspace whose per-token core footprint (~114KB)
        # is >2x the nvfp4 layout estimate (~52KB), so sizing with "nvfp4"
        # makes large prefill chunks (>~7.5k tokens) overflow the arena at
        # _map_core_workspace_views time.
        return B12XMoEArenaCaps(
            device=self.device,
            dtype=self.q_dtype,
            quant_mode="w4a16",
            weight_E=int(weight_E),
            k=int(hidden_size),
            n=intermediate_size // tp_size,
            num_topk=int(num_topk),
            max_tokens=max(core_token_counts),
            core_token_counts=core_token_counts,
            route_num_experts=int(weight_E),
            route_logits_dtype=self.q_dtype,
        )

    def _make_arena_workspace(
        self,
        *,
        mode: str,
        head_dim_qk: int,
        head_dim_vo: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_total_q: int,
        max_batch: int,
        max_page_table_width: int,
        max_work_items: int,
        max_partial_rows: int,
        num_cache_pages: int,
        use_cuda_graph: bool,
    ) -> PagedAttentionWorkspace:
        from b12x.integration.attention import PagedAttentionWorkspaceContract

        contract = PagedAttentionWorkspaceContract(
            mode=mode,
            max_total_q=max_total_q,
            max_batch=max_batch,
            max_page_table_width=max_page_table_width,
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            num_cache_pages=num_cache_pages,
        )
        if self.b12x_lane.arena is None:
            raise RuntimeError("b12x execution lane does not have a joint arena")
        return self.b12x_lane.arena.make_paged_attention_workspace(
            contract,
            use_cuda_graph=use_cuda_graph,
        )

    def _prepare_cuda_graph_workspaces(
        self,
        graph_key: tuple[object, ...],
        *,
        mode: str,
        bs: int,
        total_q_capacity: int,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        swa_page_table: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        req_pool_indices: torch.Tensor | None = None,
        decode_page_table_prebuilt: bool = True,
        decode_preprocess_in_graph: bool = False,
        reset_decode_preprocess_capture: bool = False,
    ) -> None:
        if not self.attention_layers:
            raise RuntimeError(
                "b12x backend could not find RadixAttention layers for CUDA graph capture"
            )
        if not hasattr(self, "cuda_graph_prepare_plans"):
            self.cuda_graph_prepare_plans = {}

        if mode == "decode":
            cached_plan = self.cuda_graph_prepare_plans.get(graph_key)
            if cached_plan is not None:
                if reset_decode_preprocess_capture:
                    self._reset_decode_graph_metadata_capture(graph_key)
                if not decode_preprocess_in_graph:
                    self._update_decode_graph_prepare_plan(
                        cached_plan,
                        req_pool_indices=req_pool_indices,
                        decode_page_table_prebuilt=decode_page_table_prebuilt,
                    )
                return

        prepared_workspace_keys: set[tuple[object, ...]] = set()
        decode_metadata_sources: dict[
            tuple[object, ...], PagedAttentionWorkspace
        ] = {}
        has_sinks_options = (False, True) if self.has_attention_sinks else (False,)
        for layer in self.attention_layers:
            self._validate_layer_contract(layer)
            window_left = self._layer_window_left(layer)
            causal = self._layer_is_causal(layer)
            layer_page_table = (
                swa_page_table
                if self._layer_uses_sliding_window_kv_pool(layer)
                else page_table
            )
            if layer_page_table is None:
                raise RuntimeError(
                    f"b12x backend missing SWA page table for layer {layer.layer_id}"
                )
            for has_sinks in has_sinks_options:
                workspace_key = self._cuda_graph_workspace_key(
                    graph_key, layer, has_sinks
                )
                workspace = self._get_or_create_graph_workspace(
                    workspace_key,
                    mode=mode,
                    total_q_capacity=total_q_capacity,
                    bs=bs,
                    head_dim_qk=layer.qk_head_dim,
                    head_dim_vo=layer.v_head_dim,
                    num_q_heads=layer.tp_q_head_num,
                    num_kv_heads=layer.tp_k_head_num,
                    causal=causal,
                    window_left=window_left,
                    num_cache_pages=self._num_cache_pages_for_layer(layer),
                    runtime_page_table=layer_page_table,
                    cu_seqlens_q=cu_seqlens_q,
                )
                if workspace_key in prepared_workspace_keys:
                    continue
                prepared_workspace_keys.add(workspace_key)
                if mode == "decode":
                    metadata_key = self._decode_graph_metadata_key(
                        graph_key,
                        layer,
                        window_left=window_left,
                    )
                    metadata_source = self._ensure_decode_graph_metadata_source(
                        workspace,
                        metadata_key,
                    )
                    decode_metadata_sources.setdefault(metadata_key, metadata_source)
                    self._bind_decode_graph_runtime_buffers(
                        workspace,
                        bs,
                        cache_seqlens=cache_seqlens,
                        page_table=layer_page_table,
                        cu_seqlens_q=cu_seqlens_q,
                    )
                    if metadata_source is not workspace:
                        self._bind_decode_graph_runtime_buffers(
                            metadata_source,
                            bs,
                            cache_seqlens=cache_seqlens,
                            page_table=layer_page_table,
                            cu_seqlens_q=cu_seqlens_q,
                        )
                else:
                    workspace.update_prefill_graph_replay_metadata(
                        layer_page_table,
                        cache_seqlens,
                        cu_seqlens_q,
                        causal=causal,
                        window_left=window_left,
                    )
        if mode == "decode":
            prepare_plan = _DecodeGraphPreparePlan(
                metadata_sources=tuple(decode_metadata_sources.values())
            )
            self.cuda_graph_prepare_plans[graph_key] = prepare_plan
            if reset_decode_preprocess_capture:
                self._reset_decode_graph_metadata_capture(graph_key)
            if not decode_preprocess_in_graph:
                self._update_decode_graph_prepare_plan(
                    prepare_plan,
                    req_pool_indices=req_pool_indices,
                    decode_page_table_prebuilt=decode_page_table_prebuilt,
                )

    def _update_decode_graph_prepare_plan(
        self,
        prepare_plan: _DecodeGraphPreparePlan,
        *,
        req_pool_indices: torch.Tensor | None,
        decode_page_table_prebuilt: bool,
    ) -> None:
        decode_page_table_ready = decode_page_table_prebuilt
        for metadata_source in prepare_plan.metadata_sources:
            if getattr(metadata_source, "_decode_graph_chunk_pages_lut", None) is None:
                continue
            if getattr(metadata_source, "_decode_graph_metadata_captured_in_graph", False):
                continue
            if decode_page_table_ready:
                metadata_source.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
                continue
            if req_pool_indices is None:
                raise RuntimeError(
                    "decode graph replay metadata update requires req_pool_indices"
                )
            metadata_source.update_decode_graph_replay_metadata(
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
            )
            decode_page_table_ready = True

    def _decode_graph_metadata_key(
        self,
        graph_key: tuple[object, ...],
        layer: RadixAttention,
        *,
        window_left: int,
    ) -> tuple[object, ...]:
        return (
            *graph_key,
            "decode-metadata",
            bool(self._layer_uses_sliding_window_kv_pool(layer)),
            int(layer.tp_q_head_num),
            int(layer.tp_k_head_num),
            int(layer.qk_head_dim),
            int(layer.v_head_dim),
            int(window_left),
            self._num_cache_pages_for_layer(layer),
        )

    def _decode_graph_metadata_source(
        self,
        md: B12xForwardMetadata,
        layer: RadixAttention,
        *,
        window_left: int,
    ) -> PagedAttentionWorkspace | None:
        if md.graph_key is None:
            return None
        metadata_key = self._decode_graph_metadata_key(
            md.graph_key,
            layer,
            window_left=window_left,
        )
        return self.cuda_graph_decode_metadata_sources.get(metadata_key)

    def _workspace_shares_decode_graph_metadata(
        self,
        workspace: PagedAttentionWorkspace,
        source: PagedAttentionWorkspace,
    ) -> bool:
        for attr in _DECODE_GRAPH_METADATA_BUFFER_ATTRS:
            workspace_tensor = getattr(workspace, attr, None)
            source_tensor = getattr(source, attr, None)
            if workspace_tensor is None or source_tensor is None:
                return False
            if int(workspace_tensor.data_ptr()) != int(source_tensor.data_ptr()):
                return False
        return True

    def _decode_graph_prepare_plan_metadata_captured(
        self,
        graph_key: tuple[object, ...],
    ) -> bool:
        prepare_plan = getattr(self, "cuda_graph_prepare_plans", {}).get(graph_key)
        if prepare_plan is None or not prepare_plan.metadata_sources:
            return False
        return all(
            getattr(source, "_decode_graph_metadata_captured_in_graph", False)
            for source in prepare_plan.metadata_sources
        )

    def _reset_decode_graph_metadata_capture(
        self,
        graph_key: tuple[object, ...],
    ) -> None:
        for key, workspace in self.cuda_graph_workspaces.items():
            if key[: len(graph_key)] == graph_key:
                workspace._decode_graph_metadata_captured_in_graph = False
        prepare_plan = self.cuda_graph_prepare_plans.get(graph_key)
        if prepare_plan is None:
            return
        for source in prepare_plan.metadata_sources:
            source._decode_graph_metadata_captured_in_graph = False

    def _ensure_decode_graph_metadata_source(
        self,
        workspace: PagedAttentionWorkspace,
        metadata_key: tuple[object, ...],
    ) -> PagedAttentionWorkspace:
        source = self.cuda_graph_decode_metadata_sources.get(metadata_key)
        if source is not None:
            self._share_decode_graph_metadata_buffers(workspace, source)
            return source

        if self._decode_graph_metadata_source_conflicts(workspace):
            self._replace_decode_graph_metadata_buffers(workspace)
        self.cuda_graph_decode_metadata_sources[metadata_key] = workspace
        return workspace

    def _decode_graph_metadata_source_conflicts(
        self,
        workspace: PagedAttentionWorkspace,
    ) -> bool:
        workspace_ptrs = self._decode_graph_metadata_buffer_ptrs(workspace)
        if not workspace_ptrs:
            return False
        for source in self.cuda_graph_decode_metadata_sources.values():
            source_ptrs = self._decode_graph_metadata_buffer_ptrs(source)
            if workspace_ptrs.intersection(source_ptrs):
                return True
        return False

    def _decode_graph_metadata_buffer_ptrs(
        self,
        workspace: PagedAttentionWorkspace,
    ) -> set[int]:
        ptrs = set()
        for attr in _DECODE_GRAPH_METADATA_BUFFER_ATTRS:
            tensor = getattr(workspace, attr)
            if tensor is not None:
                ptrs.add(int(tensor.data_ptr()))
        return ptrs

    def _replace_decode_graph_metadata_buffers(
        self,
        workspace: PagedAttentionWorkspace,
    ) -> None:
        for attr in _DECODE_GRAPH_METADATA_BUFFER_ATTRS:
            tensor = getattr(workspace, attr)
            if tensor is None:
                raise RuntimeError(
                    f"decode graph workspace is missing {attr} metadata buffer"
                )
            setattr(workspace, attr, torch.empty_like(tensor))

    def _share_decode_graph_metadata_buffers(
        self,
        workspace: PagedAttentionWorkspace,
        source: PagedAttentionWorkspace,
    ) -> None:
        for attr in _DECODE_GRAPH_METADATA_BUFFER_ATTRS:
            source_tensor = getattr(source, attr)
            if source_tensor is None:
                raise RuntimeError(
                    f"decode graph metadata source is missing {attr} buffer"
                )
            setattr(workspace, attr, source_tensor)
        workspace._decode_graph_chunk_pages_lut = source._decode_graph_chunk_pages_lut
        workspace._decode_graph_max_chunks_per_req = (
            source._decode_graph_max_chunks_per_req
        )
        workspace._use_regular_decode_graph_replay = (
            source._use_regular_decode_graph_replay
        )

    def _captured_graph_total_q_capacity(
        self, graph_key: tuple[object, ...]
    ) -> int | None:
        for key, workspace in self.cuda_graph_workspaces.items():
            if key[: len(graph_key)] == graph_key:
                return int(workspace.total_q_capacity)
        return None

    def _collect_attention_layers(self, model) -> list:
        if model is None:
            return []
        layers = []
        seen_layer_ids: set[int] = set()
        for module in model.modules():
            if not all(
                hasattr(module, attr)
                for attr in (
                    "layer_id",
                    "tp_q_head_num",
                    "tp_k_head_num",
                    "tp_v_head_num",
                    "qk_head_dim",
                    "v_head_dim",
                    "sliding_window_size",
                )
            ):
                continue
            layer_id = int(module.layer_id)
            if layer_id in seen_layer_ids:
                continue
            seen_layer_ids.add(layer_id)
            layers.append(module)
        layers.sort(key=lambda layer: int(layer.layer_id))
        return layers

    def _model_has_attention_sinks(self, model) -> bool:
        if model is None:
            return False
        for module in model.modules():
            if getattr(module, "attention_sink_bias", None) is not None:
                return True
        return False

    def _collect_eager_attention_layers(self) -> list:
        representatives = {}
        for layer in self.attention_layers:
            representatives.setdefault(self._eager_layer_key(layer), layer)
        return list(representatives.values())

    def _validate_supported_kwargs(self, kwargs: dict) -> None:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise NotImplementedError(
                f"b12x attention backend does not support kwargs: {unsupported}"
            )

    def _layer_window_left(self, layer: RadixAttention) -> int:
        window_left = getattr(layer, "sliding_window_size", -1)
        if window_left is None or int(window_left) < 0:
            return -1
        return int(window_left)

    def _layer_is_causal(self, layer: RadixAttention) -> bool:
        return getattr(getattr(layer, "attn_type", None), "value", None) != "encoder_only"

    def _expected_layer_head_dims(self, layer: RadixAttention) -> tuple[int, int]:
        if self._layer_window_left(layer) >= 0:
            return int(self.swa_head_dim_qk), int(self.swa_head_dim_vo)
        return int(self.head_dim_qk), int(self.head_dim_vo)

    def _layer_uses_sliding_window_kv_pool(self, layer: RadixAttention) -> bool:
        if not self.use_sliding_window_kv_pool:
            return False
        try:
            return bool(self.swa_kv_pool.layers_mapping[int(layer.layer_id)][1])
        except KeyError as exc:
            raise KeyError(
                f"b12x backend could not find layer {layer.layer_id} in SWA KV pool mapping"
            ) from exc

    def _page_table_for_layer(
        self,
        md: B12xForwardMetadata,
        layer: RadixAttention,
    ) -> torch.Tensor:
        if not self._layer_uses_sliding_window_kv_pool(layer):
            return md.page_table
        if md.swa_page_table is None:
            raise RuntimeError(
                f"b12x backend missing SWA page table for layer {layer.layer_id}"
            )
        return md.swa_page_table

    def _num_cache_pages_for_layer(self, layer: RadixAttention | int) -> int:
        layer_id = int(layer.layer_id) if hasattr(layer, "layer_id") else int(layer)
        return int(
            self.token_to_kv_pool.get_key_buffer(layer_id).shape[0] // self.page_size
        )

    def _max_num_cache_pages(self) -> int:
        if self.attention_layers:
            return max(
                self._num_cache_pages_for_layer(layer)
                for layer in self.attention_layers
            )
        return self._num_cache_pages_for_layer(self.kv_contract_layer_id)

    def _eager_workspace_key(
        self, mode: str, layer: RadixAttention
    ) -> tuple[object, ...]:
        return (mode, *self._eager_layer_key(layer))

    def _eager_layer_key(self, layer: RadixAttention) -> tuple[object, ...]:
        return (
            getattr(getattr(layer, "attn_type", None), "value", None),
            bool(self._layer_uses_sliding_window_kv_pool(layer)),
            int(layer.tp_q_head_num),
            int(layer.tp_k_head_num),
            int(layer.qk_head_dim),
            int(layer.v_head_dim),
            self._layer_window_left(layer),
            self._num_cache_pages_for_layer(layer),
        )

    def _cuda_graph_workspace_key(
        self,
        graph_key: tuple[object, ...],
        layer: RadixAttention,
        has_sinks: bool,
    ) -> tuple[object, ...]:
        mode = graph_key[0] if graph_key else None
        if mode in ("extend", "verify"):
            return (*graph_key, *self._eager_layer_key(layer), bool(has_sinks))
        return (
            *graph_key,
            getattr(getattr(layer, "attn_type", None), "value", None),
            int(layer.layer_id),
            int(layer.qk_head_dim),
            int(layer.v_head_dim),
            self._layer_window_left(layer),
            bool(has_sinks),
        )

    def _mode_from_forward_mode(self, forward_mode: ForwardMode) -> str:
        if forward_mode.is_decode_or_idle():
            return "decode"
        if forward_mode.is_target_verify():
            return "verify"
        return "extend"

    def _select_kv_contract_layer_id(self) -> int:
        mapping = getattr(
            self.token_to_kv_pool, "full_attention_layer_id_mapping", None
        )
        if mapping is not None:
            if not mapping:
                raise ValueError(
                    "b12x attention backend requires at least one full-attention layer"
                )
            return int(next(iter(mapping)))
        return 0

    def _build_eager_cu_seqlens_q(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        mode: str,
    ) -> torch.Tensor:
        if mode == "decode":
            return torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
        if mode == "verify" and forward_batch.extend_seq_lens is None:
            tokens_per_req = self._target_verify_tokens_per_req(forward_batch)
            return torch.arange(
                0,
                bs * tokens_per_req + 1,
                tokens_per_req,
                dtype=torch.int32,
                device=self.device,
            )
        extend_lens = forward_batch.extend_seq_lens[:bs].to(torch.int32)
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(extend_lens, dim=0, out=cu_seqlens_q[1:])
        return cu_seqlens_q

    def _target_verify_tokens_per_req(self, forward_batch: ForwardBatch) -> int:
        spec_info = forward_batch.spec_info
        if spec_info is None or not hasattr(spec_info, "draft_token_num"):
            raise ValueError(
                "b12x target verify metadata requires spec_info.draft_token_num"
            )
        tokens_per_req = int(spec_info.draft_token_num)
        if tokens_per_req <= 0:
            raise ValueError(
                f"b12x target verify requires draft_token_num > 0, got {tokens_per_req}"
            )
        return tokens_per_req

    def _active_total_q_from_forward_batch(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        mode: str,
    ) -> int:
        if mode == "decode":
            return int(bs)
        if mode == "verify" and forward_batch.extend_seq_lens is None:
            return int(bs) * self._target_verify_tokens_per_req(forward_batch)

        extend_num_tokens = getattr(forward_batch, "extend_num_tokens", None)
        if extend_num_tokens is not None:
            return int(extend_num_tokens)

        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            return int(sum(extend_seq_lens_cpu[:bs]))

        return int(forward_batch.seq_lens_sum)

    def _cuda_graph_total_q_hint_from_host(
        self,
        *,
        bs: int,
        seq_lens_sum: int,
        spec_info: Optional[SpecInput],
    ) -> int:
        if spec_info is not None and hasattr(spec_info, "draft_token_num"):
            return int(bs) * int(spec_info.draft_token_num)
        return int(seq_lens_sum)

    def _build_cache_seqlens(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        mode: str,
        tokens_per_req: Optional[int] = None,
    ) -> torch.Tensor:
        cache_seqlens = forward_batch.seq_lens[:bs].to(torch.int32)
        if mode == "verify":
            if tokens_per_req is None:
                tokens_per_req = self._target_verify_tokens_per_req(forward_batch)
            cache_seqlens = cache_seqlens + tokens_per_req
        return cache_seqlens

    def _fill_cuda_graph_metadata(
        self,
        *,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        total_q_hint: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        assert self.cuda_graph_cu_seqlens_q is not None
        assert self.cuda_graph_cache_seqlens is not None
        assert self.cuda_graph_page_table is not None

        mode = self._mode_from_forward_mode(forward_mode)
        tokens_per_req: Optional[int] = None
        cu_seqlens_q = self.cuda_graph_cu_seqlens_q[: bs + 1]
        if mode == "decode":
            cu_seqlens_q.copy_(
                torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            )
        else:
            if bs <= 0:
                raise ValueError("b12x cuda-graph extend path requires bs > 0")
            if spec_info is not None and hasattr(spec_info, "draft_token_num"):
                tokens_per_req = int(spec_info.draft_token_num)
                total_q = bs * tokens_per_req
            else:
                total_q = int(total_q_hint)
                if total_q % bs != 0:
                    raise ValueError(
                        f"b12x cuda-graph extend path requires uniform q-per-request, got total_q={total_q}, bs={bs}"
                    )
                tokens_per_req = total_q // bs
            cu_seqlens_q.copy_(
                torch.arange(
                    0,
                    bs * tokens_per_req + 1,
                    tokens_per_req,
                    dtype=torch.int32,
                    device=self.device,
                )
            )

        cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
        cache_seqlens.copy_(seq_lens[:bs].to(torch.int32))
        if mode == "verify":
            if tokens_per_req is None:
                raise ValueError(
                    "b12x target verify cuda graph metadata requires spec_info.draft_token_num"
                )
            cache_seqlens.add_(tokens_per_req)

        page_table = self.cuda_graph_page_table[:bs]
        swa_page_table = (
            self.cuda_graph_swa_page_table[:bs]
            if self.cuda_graph_swa_page_table is not None
            else None
        )
        self._build_page_tables_into(
            req_pool_indices[:bs],
            cache_seqlens,
            page_table,
            swa_page_table,
            bs,
            max_pages=page_table.shape[1],
        )
        return cache_seqlens, page_table, swa_page_table, cu_seqlens_q

    def _fill_decode_cuda_graph_metadata(
        self,
        *,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        build_page_table: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        assert self.cuda_graph_cu_seqlens_q is not None
        assert self.cuda_graph_cache_seqlens is not None
        assert self.cuda_graph_page_table is not None

        cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
        cu_seqlens_q = self.cuda_graph_cu_seqlens_q[: bs + 1]
        page_table = self.cuda_graph_page_table[:bs]
        swa_page_table = (
            self.cuda_graph_swa_page_table[:bs]
            if self.cuda_graph_swa_page_table is not None
            else None
        )
        if build_page_table:
            swa_index_mapping = (
                None
                if swa_page_table is None
                else getattr(self.swa_kv_pool, "full_to_swa_index_mapping", None)
            )
            can_stage_with_b12x = (
                page_table.device.type == "cuda"
                and req_pool_indices.device.type == "cuda"
                and seq_lens.device.type == "cuda"
                and self.req_to_token.device.type == "cuda"
                and (swa_page_table is None or swa_index_mapping is not None)
                and _b12x_graph_replay_func("stage_decode_cuda_graph_metadata") is not None
            )
            if not can_stage_with_b12x:
                cache_seqlens.copy_(seq_lens[:bs].to(torch.int32))
                cu_seqlens_q.copy_(
                    torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
                )
                self._build_page_tables_into(
                    req_pool_indices[:bs],
                    cache_seqlens,
                    page_table,
                    swa_page_table,
                    bs,
                    max_pages=page_table.shape[1],
                )
            else:
                from b12x.attention.paged.graph_replay import (
                    stage_decode_cuda_graph_metadata,
                )

                stage_decode_cuda_graph_metadata(
                    req_to_token=self.req_to_token,
                    req_pool_indices=req_pool_indices[:bs],
                    seq_lens=seq_lens[:bs],
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    page_table=page_table,
                    swa_page_table=swa_page_table,
                    swa_index_mapping=swa_index_mapping,
                    page_size=self.page_size,
                )
        else:
            cache_seqlens.copy_(seq_lens[:bs].to(torch.int32))
            cu_seqlens_q.copy_(
                torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            )
        return cache_seqlens, page_table, swa_page_table, cu_seqlens_q

    def _decode_cuda_graph_metadata_views(
        self,
        bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        assert self.cuda_graph_cu_seqlens_q is not None
        assert self.cuda_graph_cache_seqlens is not None
        assert self.cuda_graph_page_table is not None
        return (
            self.cuda_graph_cache_seqlens[:bs],
            self.cuda_graph_page_table[:bs],
            self.cuda_graph_swa_page_table[:bs]
            if self.cuda_graph_swa_page_table is not None
            else None,
            self.cuda_graph_cu_seqlens_q[: bs + 1],
        )

    def _validate_layer_contract(self, layer: RadixAttention) -> None:
        if layer.tp_q_head_num <= 0 or layer.tp_k_head_num <= 0:
            raise ValueError("b12x backend expects positive layer head counts")
        if layer.tp_q_head_num % layer.tp_k_head_num != 0:
            raise ValueError(
                "b12x backend expects tp_q_head_num to be divisible by tp_k_head_num"
            )
        if layer.tp_v_head_num != layer.tp_k_head_num:
            raise ValueError(
                "b12x backend expects tp_v_head_num to match tp_k_head_num"
            )
        expected_qk, expected_v = self._expected_layer_head_dims(layer)
        if layer.qk_head_dim != expected_qk or layer.v_head_dim != expected_v:
            raise ValueError(
                "b12x backend layer head dims do not match the model config: "
                f"got qk={layer.qk_head_dim}, v={layer.v_head_dim}; "
                f"expected qk={expected_qk}, v={expected_v}"
            )

    def _prime_graph_workspace_capacity(
        self,
        workspace: PagedAttentionWorkspace,
        *,
        mode: str,
        bs: int,
        total_q_capacity: int,
        causal: bool,
        window_left: int,
        num_cache_pages: int,
        runtime_page_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> None:
        if mode == "decode":
            workspace.prepare_decode_graph_replay_state(
                batch=bs,
                total_q_capacity=total_q_capacity,
                max_page_table_width=self.max_pages_per_req,
                max_cache_page_count=max(
                    1, min(self.max_pages_per_req, int(num_cache_pages))
                ),
                window_left=window_left,
            )
            self._bind_decode_graph_runtime_buffers(
                workspace,
                bs,
                page_table=runtime_page_table,
            )
            return
        max_cache_seqlen = min(
            int(self.max_context_len), max(1, int(num_cache_pages)) * int(self.page_size)
        )
        if mode == "verify" and int(self.verify_graph_max_cache_seqlen) > 0:
            max_cache_seqlen = min(
                max_cache_seqlen, int(self.verify_graph_max_cache_seqlen)
            )
        workspace.prepare_prefill_graph_replay_state(
            batch=bs,
            total_q_capacity=total_q_capacity,
            max_page_table_width=self.max_pages_per_req,
            max_cache_seqlen=max_cache_seqlen,
            cu_seqlens_q=cu_seqlens_q,
            causal=causal,
            window_left=window_left,
        )

    def _get_paged_kv_buffers(
        self,
        token_to_kv_pool,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_flat = token_to_kv_pool.get_key_buffer(layer_id)
        v_flat = token_to_kv_pool.get_value_buffer(layer_id)
        total_slots = k_flat.shape[0]
        num_pages = total_slots // self.page_size
        kv_heads = k_flat.shape[1]
        k_head_dim = k_flat.shape[2]
        v_head_dim = v_flat.shape[2]
        k_paged = k_flat[: num_pages * self.page_size].view(
            num_pages, self.page_size, kv_heads, k_head_dim
        )
        v_paged = v_flat[: num_pages * self.page_size].view(
            num_pages, self.page_size, kv_heads, v_head_dim
        )
        return k_paged, v_paged

    def _token_indices_for_page_table(
        self,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        bs = req_pool_indices.shape[0]
        max_cache = int(cache_seqlens.max().item()) if bs > 0 else 0
        max_pages = max((max_cache + self.page_size - 1) // self.page_size, 1)

        stride = self.req_to_token.shape[1]
        page_offsets = torch.arange(
            0,
            max_pages * self.page_size,
            self.page_size,
            dtype=torch.int64,
            device=self.device,
        )
        row_indices = req_pool_indices.to(torch.int64).unsqueeze(1) * stride
        flat_indices = (row_indices + page_offsets.unsqueeze(0)).clamp(
            0, self.req_to_token.numel() - 1
        )
        return self.req_to_token.view(-1)[flat_indices]

    def _build_page_tables(
        self,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_indices = self._token_indices_for_page_table(
            req_pool_indices, cache_seqlens
        )
        page_table = (token_indices // self.page_size).to(torch.int32)
        if not self.use_sliding_window_kv_pool:
            return page_table.contiguous(), None
        swa_token_indices = self.swa_kv_pool.translate_loc_from_full_to_swa(
            token_indices.reshape(-1)
        ).reshape(token_indices.shape)
        swa_page_table = (swa_token_indices // self.page_size).to(torch.int32)
        return page_table.contiguous(), swa_page_table.contiguous()

    def _build_page_tables_into(
        self,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
        dest: torch.Tensor,
        swa_dest: torch.Tensor | None,
        bs: int,
        *,
        max_pages: int | None = None,
    ) -> None:
        token_indices, max_pages = self._token_indices_for_graph_page_table(
            req_pool_indices, dest, bs, cache_seqlens, max_pages=max_pages
        )
        dest[:bs, :max_pages] = (token_indices // self.page_size).to(torch.int32)
        if swa_dest is not None:
            swa_token_indices = self.swa_kv_pool.translate_loc_from_full_to_swa(
                token_indices.reshape(-1)
            ).reshape(token_indices.shape)
            swa_dest[:bs, :max_pages] = (swa_token_indices // self.page_size).to(
                torch.int32
            )

    def _token_indices_for_graph_page_table(
        self,
        req_pool_indices: torch.Tensor,
        dest: torch.Tensor,
        bs: int,
        cache_seqlens: torch.Tensor,
        *,
        max_pages: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        stride = self.req_to_token.shape[1]
        if max_pages is None:
            max_cache = int(cache_seqlens[:bs].max().item()) if bs > 0 else 0
            max_pages = max((max_cache + self.page_size - 1) // self.page_size, 1)
        max_pages = max(1, min(int(max_pages), dest.shape[1]))
        page_offsets = self.graph_page_offsets[:max_pages]
        row_indices = req_pool_indices[:bs].to(torch.int64).unsqueeze(1) * stride
        flat_indices = (row_indices + page_offsets.unsqueeze(0)).clamp(
            0, self.req_to_token.numel() - 1
        )
        return self.req_to_token.view(-1)[flat_indices], max_pages

    def _build_page_table(
        self,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        page_table, _ = self._build_page_tables(req_pool_indices, cache_seqlens)
        return page_table.contiguous()

    def _build_page_table_into(
        self,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
        dest: torch.Tensor,
        bs: int,
    ) -> None:
        self._build_page_tables_into(req_pool_indices, cache_seqlens, dest, None, bs)

    def _bind_decode_graph_runtime_buffers(
        self,
        workspace: PagedAttentionWorkspace,
        bs: int,
        *,
        cache_seqlens: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
    ) -> None:
        assert self.cuda_graph_page_table is not None
        assert self.cuda_graph_cu_seqlens_q is not None
        if page_table is None:
            page_table = self.cuda_graph_page_table[:bs]
        if cache_seqlens is None:
            assert self.cuda_graph_cache_seqlens is not None
            cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
        if cu_seqlens_q is None:
            cu_seqlens_q = self.cuda_graph_cu_seqlens_q[: bs + 1]
        workspace.bind_cuda_graph_runtime_metadata(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )

    def _get_descale_tensors(
        self,
        layer: RadixAttention,
        batch_size: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.kv_cache_dtype != torch.float8_e4m3fn:
            return None, None
        k_scale = 1.0 if layer.k_scale is None else float(layer.k_scale_float)
        v_scale = 1.0 if layer.v_scale is None else float(layer.v_scale_float)
        cache_key = (int(layer.layer_id), int(batch_size), k_scale, v_scale)
        cached = self.fp8_descale_cache.get(cache_key)
        if cached is not None:
            return cached

        # b12x accepts per-request descales, so avoid per-head materialization.
        k_descale = torch.full(
            (batch_size,),
            k_scale,
            dtype=torch.float32,
            device=self.device,
        )
        v_descale = torch.full(
            (batch_size,),
            v_scale,
            dtype=torch.float32,
            device=self.device,
        )
        self.fp8_descale_cache[cache_key] = (k_descale, v_descale)
        return k_descale, v_descale
