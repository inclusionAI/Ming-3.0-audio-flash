# Email: wanren.pj@antgroup.com
# Copyright (c) Ant Group. All rights reserved.
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import yaml
import re
import json
import torch
import torch.nn as nn
import torchaudio
from contextlib import nullcontext
import threading
import numpy as np
import time
import torch
import uuid
import math
import onnxruntime
import torchaudio.compliance.kaldi as kaldi
from typing import Dict, Any, Optional

from transformers import Qwen2Config, PreTrainedModel
from transformers import Qwen2Model, AutoTokenizer
from .configuration_bailing_talker import BailingTalkerConfig
from transformers.utils import ModelOutput
from .talker_tn.talker_tn import TalkerTN
import logging
from .talker_module.cfm import CFM, get_epss_timesteps
from .talker_module.dit import DiT
from .talker_module.aggregator import Aggregator
from transformers import StaticCache
from concurrent.futures import ThreadPoolExecutor

from .front.number_en import normalize_numbers
from .front.text_segment_cut import cut_text_by_semantic_length, is_chinese
from .front.toolkit import tokenize_mixed_text_iterator


class SpkembExtractor:

    def __init__(self,
                 campplus_model: str,
                 target_sr: int = 16000,
                 ):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 2
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option,
                                                             providers=["CPUExecutionProvider"])
        self.target_sr = target_sr

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(
                                                  dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding])
        return embedding

    def __call__(self, waveform, **kwargs) -> Optional[Dict[str, Any]]:
        spk_emb = self._extract_spk_embedding(waveform)

        return spk_emb



class CFMGraphExecutor:
    def __init__(self, config, cfm, aggregator, stop_head):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.initialized = False

        # 占位符
        self.last_hidden_state_placeholder = None
        self.his_lat_placeholder = None
        self.randn_like_placeholder = None
        self.t_placeholder = None
        self.sde_args_placeholder = None
        self.sde_rnd_placeholder = None
        self.gen_lat_placeholder = None
        self.inputs_embeds_placeholder = None
        self.stop_out_placeholder = None
        self.graph = None

    def execute(self, input_tensor, his_lat, cfg_strength=2., sigma=0.25, temperature=0.):
        bat_size, his_patch_size, z_dim = his_lat.shape
        randn_tensor = torch.randn((bat_size, self.config.patch_size, z_dim),
                                   device=input_tensor.device, dtype=input_tensor.dtype)
        t = get_epss_timesteps(
            self.config.steps, device=input_tensor.device, dtype=input_tensor.dtype
        )
        sde_rnd = torch.randn((self.config.steps, *randn_tensor.shape),
                              device=input_tensor.device, dtype=input_tensor.dtype)

        # 初始化
        if not self.initialized:
            self._initialize_graph(input_tensor, his_lat, randn_tensor, sde_rnd)

        self.last_hidden_state_placeholder.copy_(input_tensor)
        self.his_lat_placeholder.copy_(his_lat)
        self.randn_like_placeholder.copy_(randn_tensor)
        self.t_placeholder.copy_(t)
        self.sde_args_placeholder[0] = cfg_strength
        self.sde_args_placeholder[1] = sigma
        self.sde_args_placeholder[2] = temperature
        self.sde_rnd_placeholder.copy_(sde_rnd)
        # torch.cuda.current_stream().synchronize()

        # 回放
        self.graph.replay()

        gen_lat = torch.empty_like(self.gen_lat_placeholder)
        gen_lat.copy_(self.gen_lat_placeholder)

        inputs_embeds = torch.empty_like(self.inputs_embeds_placeholder)
        inputs_embeds.copy_(self.inputs_embeds_placeholder)

        stop_out = torch.empty_like(self.stop_out_placeholder)
        stop_out.copy_(self.stop_out_placeholder)

        # torch.cuda.current_stream().synchronize()

        return gen_lat, inputs_embeds, stop_out

    def _initialize_graph(self, input_tensor, his_lat, randn_tensor, sde_rnd):
        self.last_hidden_state_placeholder = torch.empty_like(input_tensor)
        self.his_lat_placeholder = torch.empty_like(his_lat)
        self.randn_like_placeholder = torch.empty_like(randn_tensor)
        self.t_placeholder = get_epss_timesteps(
            self.config.steps, device=input_tensor.device, dtype=input_tensor.dtype
        )
        self.sde_args_placeholder = torch.empty(3, device=input_tensor.device, dtype=input_tensor.dtype)
        self.sde_rnd_placeholder = torch.empty_like(sde_rnd)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.gen_lat_placeholder = self.cfm.sample(
                self.last_hidden_state_placeholder,
                self.his_lat_placeholder,
                self.randn_like_placeholder,
                self.t_placeholder,
                self.sde_args_placeholder,
                self.sde_rnd_placeholder,
            )

            self.inputs_embeds_placeholder = self.aggregator(self.gen_lat_placeholder)
            self.stop_out_placeholder = self.stop_head(
                self.last_hidden_state_placeholder[:, -1, :]
            ).softmax(dim=-1)

        self.initialized = True


from queue import Queue
from threading import Lock


class CFMGraphExecutorPool:
    def __init__(self, config, cfm, aggregator, stop_head, pool_size=5):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.pool_size = pool_size
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()  # 确保线程安全

        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.pool_size):
            executor = CFMGraphExecutor(
                self.config, self.cfm, self.aggregator, self.stop_head
            )
            self.pool.put(executor)

    def acquire(self):
        return self.pool.get()

    def release(self, executor):
        if isinstance(executor, CFMGraphExecutor):
            self.pool.put(executor)

    def execute(self, input_tensor, his_lat, cfg_strength=2., sigma=0.25, temperature=0.):
        executor = self.acquire()
        try:
            gen_lat, inputs_embeds, stop_out = executor.execute(
                input_tensor, his_lat,
                cfg_strength=cfg_strength, sigma=sigma, temperature=temperature)
        finally:
            self.release(executor)
            return gen_lat, inputs_embeds, stop_out

    def __len__(self):
        return self.pool.qsize()

    def __str__(self):
        return f"CFMGraphExecutorPool(pool_size={self.pool_size}, available={self.__len__()})"


@dataclass
class BailingTalkerOutputWithPast(ModelOutput):
    pass


import queue


class BailingTalker2(PreTrainedModel):
    config_class = BailingTalkerConfig
    base_model_prefix = "model"

    def __init__(self, config: BailingTalkerConfig):
        super().__init__(config)

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{self.config.name_or_path}/llm"
        )
        self.model_config = Qwen2Config.from_pretrained(
            f"{self.config.name_or_path}/llm"
        )
        self.model = Qwen2Model(self.model_config)
        self.model.config._attn_implementation = "sdpa"
        self.latent_dim = 64
        self.cfm = CFM(
            DiT(
                llm_input_dim=self.model.config.hidden_size,
                **config.flowmodel,
            ),
            steps=config.steps
        )

        self.aggregator = Aggregator(
            llm_input_dim=self.model.config.hidden_size,
            **config.aggregator,
        )

        self.stop_head = nn.Linear(self.model.config.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.model.config.hidden_size, bias=True)
        self.spkemb_extractor = SpkembExtractor(f"{self.config.name_or_path}/campplus.onnx")

        self.patch_size = config.patch_size
        self.his_patch_size = config.history_patch_size

        self.normalizer = TalkerTN(f"{self.config.name_or_path}/talker_tn")

        self.lock = threading.Lock()
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.vae_cache = {}
        self.sil_holder_cache = {}

        self.initialized = None
        self.initial_lock = threading.Lock()
        self.registered_prompt = dict()
        self.max_conc = 8
        self.executor = ThreadPoolExecutor(max_workers=self.max_conc)
        self.sampler_pool = CFMGraphExecutorPool(
            self.config, self.cfm, self.aggregator, self.stop_head, self.max_conc
        )
        self.model_graph_pool = queue.Queue()
        self.past_key_values = None
        for _ in range(self.max_conc):
            self.model_graph_pool.put((None, None, None, None, None))

        assets_dir = os.path.abspath(self.config.name_or_path)
        with open(f'{assets_dir}/data/voice_name.json', 'r', encoding='utf-8') as voice_file:
            self.voice_json_dict = json.load(voice_file)
        for key, value in self.voice_json_dict.items():
            prompt_wav_path = os.path.join(assets_dir, self.voice_json_dict[key]["prompt_wav_path"])
            self.voice_json_dict[key]["prompt_wav_path"] = prompt_wav_path

    def set_multithread_conc(self, max_thread_conc):
        self.max_conc = max_thread_conc
        self.executor = ThreadPoolExecutor(max_workers=self.max_conc)
        self.sampler_pool = CFMGraphExecutorPool(
            self.config, self.cfm, self.aggregator, self.stop_head, max_thread_conc
        )
        self.model_graph_pool = queue.Queue()
        for _ in range(self.max_conc):
            self.model_graph_pool.put((None, None, None, None, None))

        self.initial_graph()

    def initial_graph(self):

        with self.initial_lock:
            if not self.initialized:
                for _ in range(self.max_conc):
                    this_uuid = str(uuid.uuid1())

                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = []
                        self.llm_end_dict[this_uuid] = False
                        self.vae_cache[this_uuid] = {"past_key_values": None, "stream_state": (None, None, None)}
                        self.sil_holder_cache[this_uuid] = None

                    prompt = "Please generate speech based on the following description.\n"
                    text = "初始化编译图"
                    prompt_text = ""
                    prompt_wav_lat = prompt_wav_emb = None
                    future = self.executor.submit(
                        self.llm_job,
                        prompt,
                        text,
                        None,
                        None,
                        prompt_text,
                        prompt_wav_lat,
                        prompt_wav_emb,
                        this_uuid,
                    )
                    future.result()

                    with self.lock:
                        self.tts_speech_token_dict.pop(this_uuid)
                        self.llm_end_dict.pop(this_uuid)
                        self.vae_cache.pop(this_uuid)
                        self.sil_holder_cache.pop(this_uuid)

                self.initialized = True

    def set_use_vllm(self, use_vllm: bool, vllm_in_process: bool = False): ...

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.FloatTensor,
        prompt_wav_lat=None,
        min_new_token=10,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        step = 0

        his_lat = torch.zeros(1, self.his_patch_size, self.latent_dim).to(device=self.device, dtype=self.dtype)
        if prompt_wav_lat is not None:
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat

        max_cache_len = 2048
        start_t = time.perf_counter()
        max_cache_len = 2048
        past_key_values, inputs_embeds_placeholder, cache_position_placeholder, outputs_placeholder, model_graph = self.model_graph_pool.get()
        if past_key_values is None:
            past_key_values = StaticCache(config=self.model.config, max_batch_size=1, max_cache_len=max_cache_len, device=self.model.device, dtype=self.model.dtype)
        else:
            past_key_values.reset()

        prefill_len = inputs_embeds.shape[1]

        while step < 1000 and step < max_cache_len - prefill_len:
            
            if step == 0:
                outputs = self.model(
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                )
            else:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )

                # outputs = self.model(
                #     past_key_values=self.past_key_values,
                #     inputs_embeds=inputs_embeds,
                #     use_cache=True,
                #     cache_position=cache_position,
                # )

                if model_graph is None:
                    model_graph = torch.cuda.CUDAGraph()
                    inputs_embeds_placeholder = torch.empty_like(inputs_embeds)
                    cache_position_placeholder = torch.empty_like(cache_position)
                    with torch.cuda.graph(model_graph):
                        outputs_placeholder = self.model(
                            past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds_placeholder,
                            use_cache=True,
                            cache_position=cache_position_placeholder,
                        )

                inputs_embeds_placeholder.copy_(inputs_embeds)
                cache_position_placeholder.copy_(cache_position)

                # 回放
                model_graph.replay()

                outputs = outputs_placeholder

            llm_end_time = time.perf_counter()

            # # 原始实现
            # t = 1/32. * torch.tensor([0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], device=his_lat.device, dtype=his_lat.dtype)
            # gen_lat  = self.cfm.sample(outputs.last_hidden_state[:, -1:, :], his_lat, torch.randn_like(his_lat), t)
            # inputs_embeds = self.aggregator(gen_lat)
            # stop_out = self.stop_head(outputs.last_hidden_state[:, -1, :]).softmax(dim=-1).cpu()

            gen_lat, inputs_embeds, stop_out = self.sampler_pool.execute(
                outputs.last_hidden_state[:, -1:, :], his_lat,
                cfg, sigma, temperature,
            )

            end_t = time.perf_counter()
            # print(f"step time cost: {llm_end_time - start_t:.3f}s {end_t - llm_end_time:.3f}s")
            start_t = end_t

            if self.his_patch_size == self.patch_size:
                his_lat = gen_lat
            elif self.his_patch_size > self.patch_size:
                his_lat = torch.cat([his_lat[:, self.patch_size-self.his_patch_size:], gen_lat], dim=1)
            else:
                raise NotImplementedError

            if step > min_new_token and stop_out.cpu()[0, 1] > 0.5:
                yield gen_lat, True
                break

            yield gen_lat, False
            step += 1
        self.model_graph_pool.put(
            (
                past_key_values,
                inputs_embeds_placeholder,
                cache_position_placeholder,
                outputs_placeholder,
                model_graph,
            )
        )

    def omni_audio_generation_func(
        self,
        prompt,
        text,
        spk_emb,
        instruction,
        prompt_text=None,
        prompt_wav_lat=None,
        prompt_wav_emb=None,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        # TODO 拼序列的逻辑写在这
        # 是否加声纹
        spk_emb_prompt = []
        if spk_emb is not None:
            for i, se in enumerate(spk_emb):
                spk_emb_prompt.extend(
                    self.tokenizer.encode(f"  speaker_{i+1}:") +
                    self.tokenizer.encode("<|vision_start|>") +
                    self.tokenizer.encode("<|vision_pad|>") +
                    self.tokenizer.encode("<|vision_end|>\n")
                )
        # print(f'spk_emb_prompt: {self.tokenizer.decode(spk_emb_prompt)}')
        # 是否加指令控制
        instruction_prompt = []
        if instruction is not None:
            instruction_prompt = (
                self.tokenizer.encode(instruction) +
                self.tokenizer.encode('<|im_end|>')
            )
        # print(f'instruction_prompt: {self.tokenizer.decode(instruction_prompt)}')

        # 是否zero-shot
        prompt_text_token = []
        prompt_latent_token = []
        if prompt_wav_emb is not None and prompt_text is not None:
            prompt_text_token = self.tokenizer.encode(prompt_text)
            prompt_latent_token = self.tokenizer.encode('<audioPatch>') * prompt_wav_emb.size(1)
        # print(f'prompt_text_token: {self.tokenizer.decode(prompt_text_token)}')
        # print(f'prompt_latent_token: {self.tokenizer.decode(prompt_latent_token)}')

        # bgm无' Text input:\n'
        prompt2 = self.tokenizer.encode(' Text input:\n')
        if 'Genre: ' in text and 'Mood: ' in text and 'Instrument: ' in text and 'Theme: ' in text and 'Duration: ' in text:
            prompt2 = []
        # print(f'prompt2: {self.tokenizer.decode(prompt2)}')

        input_part = (
            self.tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n") +
            self.tokenizer.encode("<|im_start|>user\n") +
            self.tokenizer.encode(prompt) +
            spk_emb_prompt +
            prompt2 +
            prompt_text_token +
            self.tokenizer.encode(text) +
            self.tokenizer.encode("<|im_end|>\n") +
            self.tokenizer.encode("<|im_start|>assistant\n") +
            instruction_prompt +
            self.tokenizer.encode("<audio>") +
            prompt_latent_token
        )
        from loguru import logger
        logger.info(self.tokenizer.decode(input_part).__repr__())

        input_ids = torch.tensor(input_part, dtype=torch.long).unsqueeze(0).to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(self.device)

        # 插入声纹特征/prompt latent
        if spk_emb is not None:
            spk_token_id = self.tokenizer.encode("<|vision_start|>")
            assert len(spk_token_id) == 1
            spk_indices = torch.where(input_ids[0] == spk_token_id[0])[0]
            assert len(spk_indices) > 0
            for i, se in enumerate(spk_emb):
                inputs_embeds[0, spk_indices[i] + 1] = se

        if prompt_wav_emb is not None and prompt_text is not None:
            audio_token_id = self.tokenizer.encode("<audio>")
            assert len(audio_token_id) == 1
            audio_indices = torch.where(input_ids[0] == audio_token_id[0])[0]
            assert len(audio_indices) > 0
            # 只考虑batchsize=1
            inputs_embeds[0, audio_indices[0] + 1:audio_indices[0] + 1 + prompt_wav_emb.size(1), :] = prompt_wav_emb[0]

        for audio_token in self.generate(
            inputs_embeds=inputs_embeds,
            prompt_wav_lat=prompt_wav_lat,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
        ):
            yield audio_token

    def token2wav(
        self,
        audio_detokenizer,
        token,
        cache=None,
        stream=False,
        last_chunk=False,
    ):
        speech, stream_state, past_key_values = audio_detokenizer.decode(torch.cat(token, dim=1),
                                                                      use_cache=stream, **cache,
                                                                      last_chunk=last_chunk)
        new_cache = {"past_key_values": past_key_values, "stream_state": stream_state}
        return speech[0].detach().float(), new_cache

    @staticmethod
    def silence_holder(speech, sample_rate, sil_cache=None, last_chunk=True, sil_th=1e-3, last_sil=0.3):
        if speech.numel() == 0:
            assert not last_chunk
            return speech, sil_cache

        frame_step, frame_size = int(sample_rate * 0.1), int(sample_rate * 0.1)
        if sil_cache is None:
            sil_cache = {'holder': [], 'buffer': []}
        if sil_cache['buffer']:
            speech = torch.cat([*sil_cache['buffer'], speech], dim=-1)
            sil_cache['buffer'] = []
        if speech.shape[-1] < frame_size:
            sil_cache['buffer'].append(speech)
            if last_chunk:
                speech = torch.cat(sil_cache['holder'] + sil_cache['buffer'], dim=-1)
                return speech[..., :int(last_sil * sample_rate)], sil_cache
            return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

        num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
        cur_len = (num_frame - 1) * frame_step + frame_size
        if speech.shape[-1] > cur_len:
            sil_cache['buffer'].append(speech[..., cur_len:])
            speech = speech[..., :cur_len]
        spe_frames = speech.unfold(-1, frame_size, frame_step)
        scores = spe_frames.abs().mean(dim=-1)
        scores = scores.mean(dim=list(range(scores.dim()-1)))
        idx = scores.shape[0] - 1
        while idx >= 0:
            if scores[idx] > sil_th:
                break
            idx -= 1
        if idx < 0:
            sil_cache['holder'].append(speech)
            if last_chunk:
                speech = torch.cat(sil_cache['holder']+sil_cache['buffer'], dim=-1)
                return speech[..., :int(last_sil * sample_rate)], sil_cache
            return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache
        non_sil_len = idx * frame_step + frame_size
        if last_chunk:
            non_sil_len += int(last_sil * sample_rate)
        speech = torch.cat([*sil_cache['holder'], speech[..., :non_sil_len]], dim=-1)
        sil_cache['holder'] = []
        if non_sil_len < speech.shape[-1]:
            sil_cache['holder'].append(speech[..., non_sil_len:])
        return speech, sil_cache

    def llm_job(
        self,
        prompt,
        text,
        spk_emb,
        instruction,
        prompt_text,
        prompt_wav_lat,
        prompt_wav_emb,
        this_uuid,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            for audio_token in self.omni_audio_generation_func(
                prompt=prompt,
                text=text,
                spk_emb=spk_emb,
                instruction=instruction,
                prompt_text=prompt_text,
                prompt_wav_lat=prompt_wav_lat,
                prompt_wav_emb=prompt_wav_emb,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
            ):
                self.tts_speech_token_dict[this_uuid].append(audio_token)

        self.llm_end_dict[this_uuid] = True

    def tts_job(
        self,
        prompt,
        text,
        spk_emb,
        instruction,
        audio_detokenizer,
        prompt_text,
        prompt_wav_lat,
        prompt_wav_emb,
        stream,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
    ):
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            this_uuid = str(uuid.uuid1())

            with self.lock:
                self.tts_speech_token_dict[this_uuid] = []
                self.llm_end_dict[this_uuid] = False
                self.vae_cache[this_uuid] = {"past_key_values": None, "stream_state": (None, None, None)}
                self.sil_holder_cache[this_uuid] = None

            future = self.executor.submit(
                self.llm_job,
                prompt,
                text,
                spk_emb,
                instruction,
                prompt_text,
                prompt_wav_lat,
                prompt_wav_emb,
                this_uuid,
                cfg,
                sigma,
                temperature,
            )

            if stream is True:
                token_offset = 0
                while True:
                    time.sleep(0.1)
                    nxt = len(self.tts_speech_token_dict[this_uuid])
                    # print(nxt, token_offset)
                    if nxt > token_offset:
                        this_tts_speech_token = self.tts_speech_token_dict[this_uuid][
                            token_offset:nxt
                        ]

                        last_chunk = this_tts_speech_token[-1][-1]
                        this_tts_speech_token = [ii[0] for ii in this_tts_speech_token]
                        this_tts_speech, self.vae_cache[this_uuid] = self.token2wav(
                            audio_detokenizer=audio_detokenizer,
                            token=this_tts_speech_token,
                            cache=self.vae_cache[this_uuid],
                            stream=True,
                            last_chunk=last_chunk,
                        )
                        token_offset = nxt
                        this_tts_speech, self.sil_holder_cache[this_uuid] = self.silence_holder(
                            this_tts_speech, audio_detokenizer.config.sample_rate,
                            self.sil_holder_cache[this_uuid], last_chunk,
                        )
                        yield {"tts_speech": this_tts_speech.cpu()}

                    if self.llm_end_dict[this_uuid] is True and token_offset == len(
                        self.tts_speech_token_dict[this_uuid]
                    ):
                        break
                future.result()
            else:
                # deal with all tokens
                future.result()
                this_tts_speech_token = self.tts_speech_token_dict[this_uuid]
                this_tts_speech_token = [ii[0] for ii in this_tts_speech_token]
                this_tts_speech, self.vae_cache[this_uuid] = self.token2wav(
                    audio_detokenizer=audio_detokenizer,
                    token=this_tts_speech_token,
                    cache=self.vae_cache[this_uuid],
                    stream=False,
                    last_chunk=True
                )
                this_tts_speech, self.sil_holder_cache[this_uuid] = self.silence_holder(
                    this_tts_speech, audio_detokenizer.config.sample_rate,
                    self.sil_holder_cache[this_uuid], True
                )
                yield {"tts_speech": this_tts_speech.cpu()}

            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

            with self.lock:
                self.tts_speech_token_dict.pop(this_uuid)
                self.llm_end_dict.pop(this_uuid)
                self.vae_cache.pop(this_uuid)
                self.sil_holder_cache.pop(this_uuid)

    def register_prompt_wav(self, prompt_wav_path, audio_detokenizer):
        if isinstance(prompt_wav_path, str):
            prompt_wav_path = [prompt_wav_path]
        assert isinstance(prompt_wav_path, list)

        speech = []
        spk_emb = []
        for x in prompt_wav_path:
            speech_tmp, sample_rate = torchaudio.load(x, backend="soundfile")
            speech_tmp1 = speech_tmp.clone()
            if sample_rate != audio_detokenizer.config.sample_rate:
                speech_tmp = torchaudio.transforms.Resample(sample_rate, audio_detokenizer.config.sample_rate)(speech_tmp)
            speech.append(speech_tmp)

            if sample_rate != 16000:
                speech_tmp1 = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech_tmp1)
            se = self.spkemb_extractor(speech_tmp1)
            se = self.spk_head(se.to(device=self.device, dtype=self.dtype))
            spk_emb.append(se)

        speech = torch.cat(speech, dim=-1)

        patch_pt = audio_detokenizer.encoder.hop_size * max(1, audio_detokenizer.encoder.patch_size) * self.patch_size
        if speech.shape[-1] % patch_pt != 0:
            pad_len = (speech.shape[1] + patch_pt - 1) // patch_pt * patch_pt
            pad_speech = torch.zeros((speech.shape[0], pad_len), dtype=speech.dtype, device=speech.device)
            pad_speech[:, -speech.shape[1]:] = speech
            speech = pad_speech
        prompt_wav_lat, _ = audio_detokenizer.encode_latent(
            speech.to(dtype=torch.bfloat16, device=self.device),
            torch.tensor([speech.size(1)], dtype=torch.long, device=self.device)
        )  # btd
        assert prompt_wav_lat.shape[1] % self.patch_size == 0
        prompt_wav_lat = prompt_wav_lat.reshape(
            -1, self.patch_size, prompt_wav_lat.shape[-1]
        )
        prompt_wav_emb = self.aggregator(prompt_wav_lat)
        prompt_wav_lat = prompt_wav_lat.reshape(1, -1, prompt_wav_lat.shape[-1])
        prompt_wav_emb = prompt_wav_emb.reshape(1, -1, prompt_wav_emb.shape[-1])

        if len(prompt_wav_path) == 0:
            prompt_wav_path = prompt_wav_path[0]
        else:
            prompt_wav_path = '|'.join(prompt_wav_path)
        self.registered_prompt[prompt_wav_path] = {
            "prompt_wav_lat": prompt_wav_lat,
            "prompt_wav_emb": prompt_wav_emb,
            "spk_emb": spk_emb
        }
        logging.info(f"register_prompt_wav with {prompt_wav_path}")

    def get_prompt_emb(self, prompt_wav_path, audio_detokenizer, use_spk_emb=False, use_zero_spk_emb=False):
        if prompt_wav_path is None:
            if not use_zero_spk_emb:
                return None, None, None
            else:
                return None, None, torch.zeros(1, 896, device=self.device, dtype=self.dtype)
        if isinstance(prompt_wav_path, list):
            key = '|'.join(prompt_wav_path)
        else:
            key = prompt_wav_path
        if key not in self.registered_prompt:
            self.register_prompt_wav(prompt_wav_path, audio_detokenizer)
        registered_prompt_msg = self.registered_prompt[key]
        spk_emb = registered_prompt_msg["spk_emb"] if use_spk_emb else None
        return (
            registered_prompt_msg["prompt_wav_lat"],
            registered_prompt_msg["prompt_wav_emb"],
            spk_emb
        )

    def omni_audio_generation(
        self,
        tts_text,
        voice_name='DB30',
        prompt_text=None,
        prompt_wav_path=None,
        max_length=50,
        audio_detokenizer=None,
        stream=False,
        **kwargs,
    ):
        # 兼容新的zero-shot的tts接口
        text = tts_text
        prompt = 'Please generate speech based on the following description.\n'
        instruction = None

        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            talker_last_time = time.perf_counter()
            self.initial_graph()
            # prompt音频等处理
            if voice_name is not None and voice_name in self.voice_json_dict:
                assert prompt_wav_path is None and prompt_text is None
                prompt_text = self.voice_json_dict[voice_name]["prompt_text"]
                prompt_wav_path = self.voice_json_dict[voice_name]["prompt_wav_path"]

            prompt_wav_lat, prompt_wav_emb, spk_emb = self.get_prompt_emb(
                prompt_wav_path, audio_detokenizer, use_spk_emb=True, use_zero_spk_emb=False
            )
            # if prompt_text is None:
            #     prompt_wav_lat, prompt_wav_emb = None, None
            # print(prompt_wav_lat.size(), prompt_wav_emb.size(), spk_emb, prompt)

            assert (
                max_length > 0
            ), f"max_length must be greater than 0, but here is {max_length}"
            streaming_text = []
            count = 0
            cache_position = {}
            wds_lg_zh = 6.07
            wds_lg_en = 16

            # str2list, for english
            tts_text_list = tokenize_mixed_text_iterator(text)
            before_itrator_time = time.time()

            for i, ele in enumerate(tts_text_list):
                if i == 0:
                    logging.info("get first token time is " + str(time.time() - before_itrator_time))
                    self.first_token_time = time.time()
                
                if len(ele) == 0:
                    continue

                # 判断是否为句子结束符
                should_process = False
                if ele[-1] in "！？。，!?" and (
                    len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8
                ):
                    should_process = True
                    streaming_text.append(ele)
                
                elif ele[-1] == "." and \
                    (len(streaming_text) >= 12 or count > 0 and len(streaming_text)>=8) and \
                    bool(re.search(r'[0-9]', streaming_text[-1][-1])) is False:
                    should_process = True
                    streaming_text.append(ele)
                
                elif ele[-1] == "\n":
                    if len(streaming_text) > 0:
                        if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                            if bool(re.search(r"[\u4e00-\u9fff]", streaming_text[-1][-1])):
                                ele = "，"
                                streaming_text.append(ele)
                        else:
                            if len(ele) > 1 and bool(re.search(r"[a-zA-Z]", ele[-2])):
                                ele = ele[:-1] + "."
                            else:
                                ele = ele[:-1]
                            streaming_text.append(ele)
                    
                    if len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8:
                        should_process = True
                else:
                    streaming_text.append(ele)
                    continue

                if should_process:
                    streaming_text = "".join(streaming_text)
                    sub_output_dict = cut_text_by_semantic_length(streaming_text, max_length)
                    text_list = sub_output_dict["fragments"]
                    
                    if not text_list:
                        logging.info(f'{streaming_text}\thas no valid segments')
                        continue
                    
                    # print(text_list)
                    for text_ori in text_list:
                        all_wavs = []
                        
                        length = len(text_ori)
                        if len(cache_position) == 0:
                            cache_position.update({count: (0, length - 1)})
                        else:
                            end_idx = list(cache_position.values())[-1][1] + 1
                            cache_position.update({count: (end_idx, end_idx + length - 1)})

                        if not is_chinese(text_ori):
                            text = normalize_numbers(text_ori)
                            wds_lg = wds_lg_en
                        else:
                            text = text_ori
                            wds_lg = wds_lg_zh
                        
                        text = self.normalizer.normalize(text)
                        if text and text[0] == "，":
                            text = text[1:]
                        
                        if count == 0:  # 首句流式
                            first_chunk_start_time = time.perf_counter()
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    prompt=prompt,
                                    text=text,
                                    spk_emb=spk_emb,
                                    instruction=instruction,
                                    audio_detokenizer=audio_detokenizer,
                                    prompt_text=prompt_text,
                                    prompt_wav_lat=prompt_wav_lat,
                                    prompt_wav_emb=prompt_wav_emb,
                                    stream=stream & True,
                                )
                            ):
                                if idx == 0:
                                    logging.info(f"first_chunk time cost: {time.perf_counter() - first_chunk_start_time:.3f} seconds")
                                if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                    break
                                else:
                                    this_dura = float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)
                                    if idx == 0:
                                        this_start_idx = 0
                                        this_end_idx = min(math.ceil(this_dura * wds_lg), length) - 1
                                    else:
                                        this_start_idx = min(list(cache_position.values())[-1][1] + 1, length - 1)
                                        this_end_idx = min((math.ceil(this_dura * wds_lg) + this_start_idx), length) - 1
                                    cache_position.update({f'{count}_{idx}': (this_start_idx, this_end_idx)})
                                    if this_start_idx == this_end_idx:
                                        this_text_ori = ''
                                    else:
                                        this_text_ori = text_ori[this_start_idx: this_end_idx + 1]
                                    all_wavs.append(this_tts_speech_dict["tts_speech"])
                                    yield this_tts_speech_dict["tts_speech"], this_text_ori, cache_position[f'{count}_{idx}'], this_dura*1000

                        else:  # 非流式
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    prompt=prompt,
                                    text=text,
                                    spk_emb=spk_emb,
                                    instruction=instruction,
                                    audio_detokenizer=audio_detokenizer,
                                    prompt_text=prompt_text,
                                    prompt_wav_lat=prompt_wav_lat,
                                    prompt_wav_emb=prompt_wav_emb,
                                    stream=False,
                                )
                            ):
                                if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                    break
                                else:
                                    all_wavs.append(this_tts_speech_dict["tts_speech"])
                                    yield this_tts_speech_dict["tts_speech"], text_ori, cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)*1000
                        count += 1
                    streaming_text = []
                    # count += 1

            # 处理最后一句
            if len(streaming_text) > 0 and re.search(
                r"[a-zA-Z\u4e00-\u9fff1-9]", "".join(streaming_text)
            ):
                streaming_text = "".join(streaming_text)
                text_list = cut_text_by_semantic_length(streaming_text, max_length)
                text_list = text_list["fragments"]
                
                if text_list:
                    # print(text_list, "for last sentence")
                    logging.info("for last sentence")
                    for text_ori in text_list:
                        all_wavs = []
                        
                        length = len(text_ori)
                        if len(cache_position) == 0:
                            cache_position.update({count: (0, length - 1)})
                        else:
                            end_idx = list(cache_position.values())[-1][1] + 1
                            cache_position.update({count: (end_idx, end_idx + length - 1)})

                        if not is_chinese(text_ori):
                            text = normalize_numbers(text_ori)
                            wds_lg = wds_lg_en
                        else:
                            text = text_ori
                            wds_lg = wds_lg_zh
                        
                        text = self.normalizer.normalize(text)
                        if text and text[0] == "，":
                            text = text[1:]
                        
                        if count == 0:  # 首句流式
                            first_chunk_start_time = time.perf_counter()
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    prompt=prompt,
                                    text=text,
                                    spk_emb=spk_emb,
                                    instruction=instruction,
                                    audio_detokenizer=audio_detokenizer,
                                    prompt_text=prompt_text,
                                    prompt_wav_lat=prompt_wav_lat,
                                    prompt_wav_emb=prompt_wav_emb,
                                    stream=stream & True,
                                )
                            ):
                                if idx == 0:
                                    logging.info(f"first_chunk time cost: {time.perf_counter() - first_chunk_start_time:.3f} seconds")
                                if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                    break
                                else:
                                    this_dura = float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)
                                    if idx == 0:
                                        this_start_idx = 0
                                        this_end_idx = min(math.ceil(this_dura * wds_lg), length) - 1
                                    else:
                                        this_start_idx = min(list(cache_position.values())[-1][1] + 1, length - 1)
                                        this_end_idx = min((math.ceil(this_dura * wds_lg) + this_start_idx), length) - 1
                                    cache_position.update({f'{count}_{idx}': (this_start_idx, this_end_idx)})
                                    if this_start_idx == this_end_idx:
                                        this_text_ori = ''
                                    else:
                                        this_text_ori = text_ori[this_start_idx: this_end_idx + 1]
                                    all_wavs.append(this_tts_speech_dict["tts_speech"])
                                    yield this_tts_speech_dict["tts_speech"], this_text_ori, cache_position[f'{count}_{idx}'], this_dura*1000

                        else:  # 非流式
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    prompt=prompt,
                                    text=text,
                                    spk_emb=spk_emb,
                                    instruction=instruction,
                                    audio_detokenizer=audio_detokenizer,
                                    prompt_text=prompt_text,
                                    prompt_wav_lat=prompt_wav_lat,
                                    prompt_wav_emb=prompt_wav_emb,
                                    stream=False,
                                )
                            ):
                                if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                    break
                                else:
                                    all_wavs.append(this_tts_speech_dict["tts_speech"])
                                    yield this_tts_speech_dict["tts_speech"], text_ori, cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)*1000

    def instruct_audio_generation(
            self,
            prompt,
            text,
            use_spk_emb=False,
            use_zero_spk_emb=False,
            instruction=None,
            prompt_wav_path=None,
            prompt_text=None,
            max_decode_steps=200,
            cfg=2.0,
            sigma=0.25,
            temperature=0,
            max_length=50,
            audio_detokenizer=None,
            stream=False,
            taskname="TTS",
            **kwargs,):
        with torch.cuda.stream(torch.cuda.Stream(self.device)):
            self.initial_graph()

            prompt_wav_lat, prompt_wav_emb, spk_emb = self.get_prompt_emb(
                prompt_wav_path, audio_detokenizer, use_spk_emb=use_spk_emb, use_zero_spk_emb=use_zero_spk_emb
            )

            if taskname in ["TTA", "BGM", "STYLE", "SPEECH_BGM", "SPEECH_SOUND", "PODCAST"]:
                for idx, this_tts_speech_dict in enumerate(
                        self.tts_job(
                            prompt=prompt,
                            text=text,
                            spk_emb=spk_emb,
                            instruction=instruction,
                            audio_detokenizer=audio_detokenizer,
                            prompt_text=prompt_text,
                            prompt_wav_lat=prompt_wav_lat,
                            prompt_wav_emb=prompt_wav_emb,
                            stream=stream,
                            cfg=cfg,
                            sigma=sigma,
                            temperature=temperature,
                        )
                ):
                    yield this_tts_speech_dict["tts_speech"], None, None, None
            elif taskname in ["TTS", "EMOTION", "BASIC", "DIALECT", "IP"]:
                assert (
                    max_length > 0
                ), f"max_length must be greater than 0, but here is {max_length}"
                streaming_text = []
                count = 0
                cache_position = {}
                wds_lg_zh = 6.07
                wds_lg_en = 16

                # str2list, for english
                tts_text_list = tokenize_mixed_text_iterator(text)
                before_itrator_time = time.time()

                for i, ele in enumerate(tts_text_list):
                    if i == 0:
                        logging.info("get first token time is " + str(time.time() - before_itrator_time))
                        self.first_token_time = time.time()

                    if len(ele) == 0:
                        continue

                    # 判断是否为句子结束符
                    should_process = False
                    if ele[-1] in "！？。，!?" and (
                        len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8
                    ):
                        should_process = True
                        streaming_text.append(ele)

                    elif ele[-1] == "." and \
                        (len(streaming_text) >= 12 or count > 0 and len(streaming_text)>=8) and \
                        bool(re.search(r'[0-9]', streaming_text[-1][-1])) is False:
                        should_process = True
                        streaming_text.append(ele)

                    elif ele[-1] == "\n":
                        if len(streaming_text) > 0:
                            if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                                if bool(re.search(r"[\u4e00-\u9fff]", streaming_text[-1][-1])):
                                    ele = "，"
                                    streaming_text.append(ele)
                            else:
                                if len(ele) > 1 and bool(re.search(r"[a-zA-Z]", ele[-2])):
                                    ele = ele[:-1] + "."
                                else:
                                    ele = ele[:-1]
                                streaming_text.append(ele)

                        if len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8:
                            should_process = True
                    else:
                        streaming_text.append(ele)
                        continue

                    if should_process:
                        streaming_text = "".join(streaming_text)
                        sub_output_dict = cut_text_by_semantic_length(streaming_text, max_length)
                        text_list = sub_output_dict["fragments"]

                        if not text_list:
                            logging.info(f'{streaming_text}\thas no valid segments')
                            continue

                        # print(text_list)
                        for text_ori in text_list:
                            all_wavs = []

                            length = len(text_ori)
                            if len(cache_position) == 0:
                                cache_position.update({count: (0, length - 1)})
                            else:
                                end_idx = list(cache_position.values())[-1][1] + 1
                                cache_position.update({count: (end_idx, end_idx + length - 1)})

                            if not is_chinese(text_ori):
                                text = normalize_numbers(text_ori)
                                wds_lg = wds_lg_en
                            else:
                                text = text_ori
                                wds_lg = wds_lg_zh

                            text = self.normalizer.normalize(text)
                            if text and text[0] == "，":
                                text = text[1:]

                            if count == 0:  # 首句流式
                                first_chunk_start_time = time.perf_counter()
                                for idx, this_tts_speech_dict in enumerate(
                                    self.tts_job(
                                        prompt=prompt,
                                        text=text,
                                        spk_emb=spk_emb,
                                        instruction=instruction,
                                        audio_detokenizer=audio_detokenizer,
                                        prompt_text=prompt_text,
                                        prompt_wav_lat=prompt_wav_lat,
                                        prompt_wav_emb=prompt_wav_emb,
                                        stream=stream & True,
                                    )
                                ):
                                    if idx == 0:
                                        logging.info(f"first_chunk time cost: {time.perf_counter() - first_chunk_start_time:.3f} seconds")
                                    if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                        break
                                    else:
                                        this_dura = float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)
                                        if idx == 0:
                                            this_start_idx = 0
                                            this_end_idx = min(math.ceil(this_dura * wds_lg), length) - 1
                                        else:
                                            this_start_idx = min(list(cache_position.values())[-1][1] + 1, length - 1)
                                            this_end_idx = min((math.ceil(this_dura * wds_lg) + this_start_idx), length) - 1
                                        cache_position.update({f'{count}_{idx}': (this_start_idx, this_end_idx)})
                                        if this_start_idx == this_end_idx:
                                            this_text_ori = ''
                                        else:
                                            this_text_ori = text_ori[this_start_idx: this_end_idx + 1]
                                        all_wavs.append(this_tts_speech_dict["tts_speech"])
                                        yield this_tts_speech_dict["tts_speech"], this_text_ori, cache_position[f'{count}_{idx}'], this_dura*1000

                            else:  # 非流式
                                for idx, this_tts_speech_dict in enumerate(
                                    self.tts_job(
                                        prompt=prompt,
                                        text=text,
                                        spk_emb=spk_emb,
                                        instruction=instruction,
                                        audio_detokenizer=audio_detokenizer,
                                        prompt_text=prompt_text,
                                        prompt_wav_lat=prompt_wav_lat,
                                        prompt_wav_emb=prompt_wav_emb,
                                        stream=False,
                                    )
                                ):
                                    if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                        break
                                    else:
                                        all_wavs.append(this_tts_speech_dict["tts_speech"])
                                        yield this_tts_speech_dict["tts_speech"], text_ori, cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)*1000
                            count += 1
                        streaming_text = []
                        # count += 1

                # 处理最后一句
                if len(streaming_text) > 0 and re.search(
                    r"[a-zA-Z\u4e00-\u9fff1-9]", "".join(streaming_text)
                ):
                    streaming_text = "".join(streaming_text)
                    text_list = cut_text_by_semantic_length(streaming_text, max_length)
                    text_list = text_list["fragments"]

                    if text_list:
                        # print(text_list, "for last sentence")
                        logging.info("for last sentence")
                        for text_ori in text_list:
                            all_wavs = []

                            length = len(text_ori)
                            if len(cache_position) == 0:
                                cache_position.update({count: (0, length - 1)})
                            else:
                                end_idx = list(cache_position.values())[-1][1] + 1
                                cache_position.update({count: (end_idx, end_idx + length - 1)})

                            if not is_chinese(text_ori):
                                text = normalize_numbers(text_ori)
                                wds_lg = wds_lg_en
                            else:
                                text = text_ori
                                wds_lg = wds_lg_zh

                            text = self.normalizer.normalize(text)
                            if text and text[0] == "，":
                                text = text[1:]

                            if count == 0:  # 首句流式
                                first_chunk_start_time = time.perf_counter()
                                for idx, this_tts_speech_dict in enumerate(
                                    self.tts_job(
                                        prompt=prompt,
                                        text=text,
                                        spk_emb=spk_emb,
                                        instruction=instruction,
                                        audio_detokenizer=audio_detokenizer,
                                        prompt_text=prompt_text,
                                        prompt_wav_lat=prompt_wav_lat,
                                        prompt_wav_emb=prompt_wav_emb,
                                        stream=stream & True,
                                    )
                                ):
                                    if idx == 0:
                                        logging.info(f"first_chunk time cost: {time.perf_counter() - first_chunk_start_time:.3f} seconds")
                                    if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                        break
                                    else:
                                        this_dura = float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)
                                        if idx == 0:
                                            this_start_idx = 0
                                            this_end_idx = min(math.ceil(this_dura * wds_lg), length) - 1
                                        else:
                                            this_start_idx = min(list(cache_position.values())[-1][1] + 1, length - 1)
                                            this_end_idx = min((math.ceil(this_dura * wds_lg) + this_start_idx), length) - 1
                                        cache_position.update({f'{count}_{idx}': (this_start_idx, this_end_idx)})
                                        if this_start_idx == this_end_idx:
                                            this_text_ori = ''
                                        else:
                                            this_text_ori = text_ori[this_start_idx: this_end_idx + 1]
                                        all_wavs.append(this_tts_speech_dict["tts_speech"])
                                        yield this_tts_speech_dict["tts_speech"], this_text_ori, cache_position[f'{count}_{idx}'], this_dura*1000

                            else:  # 非流式
                                for idx, this_tts_speech_dict in enumerate(
                                    self.tts_job(
                                        prompt=prompt,
                                        text=text,
                                        spk_emb=spk_emb,
                                        instruction=instruction,
                                        audio_detokenizer=audio_detokenizer,
                                        prompt_text=prompt_text,
                                        prompt_wav_lat=prompt_wav_lat,
                                        prompt_wav_emb=prompt_wav_emb,
                                        stream=False,
                                    )
                                ):
                                    if len(all_wavs) != 0 and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate * (16000/5818) >= len(text) and torch.cat(all_wavs, dim=-1).shape[1] / audio_detokenizer.config.sample_rate > 2:
                                        break
                                    else:
                                        all_wavs.append(this_tts_speech_dict["tts_speech"])
                                        yield this_tts_speech_dict["tts_speech"], text_ori, cache_position[count], float(this_tts_speech_dict["tts_speech"].shape[-1]/audio_detokenizer.config.sample_rate)*1000
