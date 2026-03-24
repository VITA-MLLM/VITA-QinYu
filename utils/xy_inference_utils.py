import time
import tqdm
import torch
from typing import Optional, List, Union
from vita_qinyu.constants import (
    AUD_END_TOKEN,
    AUD_START_TOKEN,
    AUD_TAG_TOKEN,
    CONV_END_TOKEN
)
from concurrent.futures import ProcessPoolExecutor, as_completed

# https://github.com/gpt-omni/mini-omni/blob/main/litgpt/generate/base.py
def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)

# https://github.com/gpt-omni/mini-omni/blob/main/litgpt/generate/base.py
def sample_top_p(logits_A: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits

# https://github.com/gpt-omni/mini-omni/blob/main/litgpt/generate/base.py
def do_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    logits = logits[0, -1]
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)

def apply_repetition_penalty(logits: torch.Tensor,
                            generated_tokens: Union[list, torch.tensor],
                            penalty: float = 1.0,
                            num_tokens_to_keep: int = 5
                            ) -> torch.Tensor:
    """
    应用重复惩罚到logits
    :param logits: [vocab_size] 当前步的logits
    :param generated_tokens: 已生成的所有token IDs列表
    :param penalty: 惩罚系数（>1抑制重复，<1鼓励重复）
    """
    if penalty == 1.0 or len(generated_tokens) == 0:
        return logits
    logits = logits[0, -1]
    # import pdb; pdb.set_trace()
    # 只处理已出现的唯一token
    # for token in generated_tokens:
    for token in set(generated_tokens[-num_tokens_to_keep:]):
        if logits[token] < 0:
            logits[token] *= penalty  # 降低负logits（高概率token）
        else:
            logits[token] /= penalty  # 降低正logits（低概率token）
    return logits[None, None]


@torch.no_grad()
def next_token(
    model,
    inputs_embeds=None,
    attention_mask=None,
    past_key_values=None,
    text_vocab_size=151690,
    codebook_size=1024,
    num_codebook=8,
    audio_start_id=None,
    i=0,
    **kwargs,
) -> torch.Tensor:
    vocab_size = text_vocab_size + codebook_size
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        first_audio_token_id=vocab_size,
        use_cache=True,
        audios=kwargs.get("audios", None),
        audio_indices=kwargs.get("audio_indices", None),
        audio_tokens=kwargs.get("history_audio_tokens", None),
        audio_token_starts=kwargs.get("audio_token_starts", None),
        audio_token_ends=kwargs.get("audio_token_ends", None),
        speaker_embeddings=kwargs.get("speaker_embeddings", None),
        speaker_mask=kwargs.get("speaker_mask", None),
    )
    # logits_st = outputs.logits[:,-1:]
    # logits_t = outputs.logits[:,-1:,:vocab_size] # last item in batch
    logits_t = outputs.logits[:, -1:].clone() # last item in batch
    logits_t[..., vocab_size:vocab_size+codebook_size*(num_codebook-1)] = -1e10 # mask out audio token ids

    next_t = do_sample(logits_t, top_k=1) # B x 1
    # if i == 5:
    #     next_t[:] = audio_start_id
    if next_t >= text_vocab_size and \
        next_t < text_vocab_size+codebook_size*num_codebook and \
        len(kwargs["audio_tokens"]) > 0 and len(kwargs["audio_tokens"]) > 0:
        audio_tokens = torch.stack(kwargs["audio_tokens"])
        i = 0
        start = text_vocab_size + i * codebook_size
        end = text_vocab_size + (i+1) * codebook_size
        logits_t = apply_repetition_penalty(
            outputs.logits[...,:end],
            text_vocab_size+audio_tokens[:,0],
            penalty=kwargs["penalty"]
        )
        next_t = do_sample(logits_t, top_k=1)

    next_a, next_ua = [], [] # layer shifted/unshifted audio tokens

    for i in range(num_codebook):
        start = text_vocab_size + i * codebook_size
        end = text_vocab_size + (i+1) * codebook_size
        logits_a_i = outputs.logits[..., start:end]
        ua_i = do_sample(logits_a_i, top_k=1) # B x 1
        next_ua.append(ua_i)
        next_a.append(start+ua_i)

    next_ua = torch.cat(next_ua, dim=-1) # B x 8
    next_a = torch.cat(next_a, dim=-1)

    next_ua = torch.cat([next_t-text_vocab_size, next_ua[1:]]) # 0 ~ 1024
    next_a = torch.cat([next_t, next_a[1:]])
    past_key_values = outputs.past_key_values
    return next_t, next_a, next_ua, past_key_values

def generate(
    tokenizer, model, input_ids, audios, audio_indices,
    history_audio_tokens=None, audio_token_starts=None, audio_token_ends=None,
    speaker_embedding=None, speaker_embeddings=None, speaker_mask=None,
    prepare_model_for_generation=False, mtp_inference_mode=[1, 0],
    streamer=None, stop_event=None,

):

    max_new_tokens = 4096
    max_ratio = 10
    max_new_tokens = 1024
    # if prepare_model_for_generation:
    if hasattr(model, "_prepare_mtp_for_generation"):
        model._prepare_mtp_for_generation(mtp_inference_mode, max_new_tokens)
    input_ids = input_ids.cuda()
    curr_modality = 'text'
    AUD_START_ID = tokenizer.convert_tokens_to_ids(AUD_START_TOKEN)
    AUD_END_ID = tokenizer.convert_tokens_to_ids(AUD_END_TOKEN)
    AUD_TAG_ID = tokenizer.convert_tokens_to_ids(AUD_TAG_TOKEN)
    IM_END_ID = tokenizer.convert_tokens_to_ids('<|im_end|>')
    IM_END_IDS = tokenizer("<|im_end|>", add_special_tokens=False).input_ids
    CONV_END_ID = tokenizer.convert_tokens_to_ids(CONV_END_TOKEN)
    first_audio_token_id = tokenizer.convert_tokens_to_ids('<|audio_0_0|>')
    last_audio_token_id = tokenizer.convert_tokens_to_ids('<|audio_7_7_pad|>')
    query_start_id = tokenizer.convert_tokens_to_ids('<function=')
    query_end_id = tokenizer.convert_tokens_to_ids('</function>')
    SPEAKER_TAG_ID = tokenizer.convert_tokens_to_ids('<|speaker|>')

    inputs_embeds = model.model.embed_tokens(input_ids) # 1 x Ti x H
    all_tokens, text_tokens, pure_text_tokens, audio_tokens = [], [], [], []
    past_key_values = None
    api_results = None

    AUDIO_TOKEN_NUM = 8
    audio_steps_to_decode = 0

    if speaker_embedding is not None:
        input_ids = torch.cat([input_ids, torch.tensor(SPEAKER_TAG_ID).to(input_ids).view(1,1)], dim=1)
        inputs_embeds = model.model.embed_tokens(input_ids) # 1 x Ti x H

        speaker_embedding = speaker_embedding.cuda()[None,:]
        speaker_embeddings = torch.cat([speaker_embeddings, speaker_embedding], dim=0).unsqueeze(0).to(inputs_embeds)

    if speaker_embeddings is not None:
        speaker_embeddings = speaker_embeddings.cuda()
        speaker_mask = speaker_mask.cuda()
        if speaker_embeddings.dim() == 2:
            speaker_embeddings = speaker_embeddings.unsqueeze(0)
        speaker_embeddings = speaker_embeddings.to(inputs_embeds)

    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        query_futures = []
        for i in tqdm.tqdm(range(max_new_tokens), total=max_new_tokens):
            # import pdb; pdb.set_trace()
            kwargs = {"i": i, "audio_start_id": AUD_START_ID}
            if i == 0:
                kwargs["audios"] = audios
                kwargs["audio_indices"] = audio_indices
                kwargs["history_audio_tokens"] = history_audio_tokens
                kwargs["audio_token_starts"] = audio_token_starts
                kwargs["audio_token_ends"] = audio_token_ends
                kwargs["speaker_mask"] = speaker_mask
                kwargs["speaker_embeddings"] = speaker_embeddings

            next_t, next_a, next_ua, past_key_values = next_token(
                model,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                text_vocab_size=tokenizer.convert_tokens_to_ids('<|audio_0_0|>'),
                codebook_size=model.config.codebook_size,
                audio_tokens=audio_tokens,
                penalty=1.1,
                **kwargs
            )
            if streamer is not None:
                streamer.put(next_t, next_ua, None)

            next_t_id = next_t.item()
            if next_t == IM_END_ID:
                api_results = []
                if len(query_futures) > 0:
                    for future in as_completed(query_futures):
                        try:
                            result = future.result()
                            api_results.append(result)
                        except Exception as e:
                            api_results.append({"error": str(e)})
                if len(api_results) > 0:
                    try:
                        print(f'{api_results=}')
                        api_results = [
                            "\n".join([
                                api_result['results']['organic'][_]['snippet']
                                for _ in range(len(api_result['results']['organic']))
                                if 'snippet' in api_result['results']['organic'][_]
                            ]) for api_result in api_results] # TODO
                        api_results = "\n".join(api_results)
                        streamer.put(None, None, api_results)
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

                if streamer is not None:
                    streamer.end()
                break
            elif tokenizer.decode(
                (text_tokens + [next_t_id])[-len(IM_END_IDS):]
            ) == '<|im_end|>':
                api_results = []
                if len(query_futures) > 0:
                    for future in as_completed(query_futures):
                        try:
                            result = future.result()
                            api_results.append(result)
                        except Exception as e:
                            api_results.append({"error": str(e)})

                if len(api_results) > 0:
                    try:
                        print(f'{api_results=}')
                        api_results = [
                            "\n".join([
                                api_result['results']['organic'][_]['snippet']
                                for _ in range(len(api_result['results']['organic']))
                                if 'snippet' in api_result['results']['organic'][_]
                            ]) for api_result in api_results] # TODO
                        api_results = "\n".join(api_results)
                        streamer.put(None, None, api_results)
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

                text_tokens.append(next_t_id)
                if streamer is not None:
                    streamer.end()
                break

            if next_t >= first_audio_token_id and next_t <= last_audio_token_id:
                inputs_embeds = model.model.embed_tokens(next_a).mean(0, keepdim=True).unsqueeze(0) # 8 => 8 x H => 1 x 1 x H
                audio_tokens.append(next_ua)

            else:
                inputs_embeds = model.model.embed_tokens(next_t)[None,:]
                text_tokens.append(next_t_id)
                if next_t_id != AUD_START_ID and next_t_id != AUD_END_ID and next_t_id != CONV_END_ID:
                    pure_text_tokens.append(next_t_id)
                if next_t_id == query_end_id and query_start_id in pure_text_tokens:
                    query_content = pure_text_tokens[pure_text_tokens.index(query_start_id):]
                    query_content = tokenizer.decode(query_content).replace("<function=google_search>{\"query\": ", "").replace("}</function>", "")
                    query_content = query_content.replace("\"", "")
                    future = executor.submit(api_search, query_content)
                    query_futures.append(future)

            if stop_event is not None and stop_event.is_set():
                print("generate stop")
                break

            all_tokens.append(next_t[0])
            # print('curr text |', tokenizer.decode(text_tokens))
            end = time.time()
            # print('cost', i, '', end - start)
            start = end

    all_tokens = torch.stack(all_tokens)
    # text_tokens = torch.stack(text_tokens)
    if len(audio_tokens) > 0:
        audio_tokens = torch.stack(audio_tokens)
    return all_tokens, text_tokens, audio_tokens, api_results


def api_search(query: str):
    import requests
    import json
    import os
    print(query)
    # API_KEY = os.environ["SERPER_API_KEY"]
    API_KEY = "6f1fd13380065f6d4cd060ff2c92cfe626c320b6"
    API_ENDPOINT = "https://google.serper.dev/search"

    params = {
        "q": query,
        "num": 5
    }

    headers = {
        'X-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(API_ENDPOINT, data=json.dumps(params), headers=headers)

    results = response.json()

    return {
        "query": query,
        "results": results
    }
