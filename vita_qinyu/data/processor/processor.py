
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import re


class AutoConfig:
    def __init__(
        self,
    ):
        pass

class Processor:
    def __init__(
        self,
        tokenizer,
    ):

        from ...constants import (
            IMG_START_TOKEN,
            IMG_END_TOKEN,
            IMG_TAG_TOKEN,
            IMG_CONTEXT_TOKEN,
            VID_START_TOKEN,
            VID_END_TOKEN,
            VID_TAG_TOKEN,
            VID_CONTEXT_TOKEN,
            PATCH_START_TOKEN,
            PATCH_END_TOKEN,
            PATCH_CONTEXT_TOKEN,
            AUD_START_TOKEN,
            AUD_END_TOKEN,
            AUD_TAG_TOKEN,
            AUD_CONTEXT_TOKEN,
        )

        IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
        IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
        IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

        AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids
        AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
        AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids

        VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
        VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
        VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

        PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
        PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
        PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

        IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
        AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids
        VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids

        assert len(IMG_CONTEXT_ID) == 1
        assert len(IMG_START_ID) == 1
        assert len(IMG_END_ID) == 1

        assert len(AUD_CONTEXT_ID) == 1
        assert len(AUD_START_ID) == 1
        assert len(AUD_END_ID) == 1

        assert len(VID_CONTEXT_ID) == 1
        assert len(VID_START_ID) == 1
        assert len(VID_END_ID) == 1

        assert len(PATCH_CONTEXT_ID) == 1
        assert len(PATCH_START_ID) == 1
        assert len(PATCH_END_ID) == 1

        IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
        IMG_START_ID = IMG_START_ID[0]
        IMG_END_ID = IMG_END_ID[0]

        AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
        AUD_START_ID = AUD_START_ID[0]
        AUD_END_ID = AUD_END_ID[0]

        VID_CONTEXT_ID = VID_CONTEXT_ID[0]
        VID_START_ID = VID_START_ID[0]
        VID_END_ID = VID_END_ID[0]

        PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
        PATCH_START_ID = PATCH_START_ID[0]
        PATCH_END_ID = PATCH_END_ID[0]

        IMG_TAG_ID = IMG_TAG_ID[0]
        AUD_TAG_ID = AUD_TAG_ID[0]
        VID_TAG_ID = VID_TAG_ID[0]

        nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

        self.config = AutoConfig()

        self.spatial_merge_size = 2
        self.config.image_token_id = IMG_CONTEXT_ID
        self.config.patch_token_id = PATCH_CONTEXT_ID
        self.config.video_token_id = VID_CONTEXT_ID
        self.config.audio_token_id = AUD_CONTEXT_ID
        self.config.image_start_token_id = IMG_START_ID
        self.config.patch_start_token_id = PATCH_START_ID
        self.config.video_start_token_id = VID_START_ID
        self.config.audio_start_token_id = AUD_START_ID
        self.config.position_id_per_seconds = 2
        self.config.seconds_per_chunk = 2


    def get_llm_pos_ids_for_vision(
        self,
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: List[int],
        grid_hs: List[int],
        grid_ws: List[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(len(t_index), -1, llm_grid_w).flatten()
        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(len(t_index), llm_grid_h, -1).flatten()
        t_index = torch.Tensor(t_index).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten().long()
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    def get_chunked_index(
        self, token_indices: torch.Tensor, tokens_per_chunk: int, remove_index: int
    ) -> list[tuple[int, int]]:
        """
        Splits token index list into chunks based on token value ranges.

        Given a list of token indices, returns a list of (start, end) index tuples representing
        slices of the list where the token values fall within successive ranges of `t_ntoken_per_chunk`.

        For example, if `t_ntoken_per_chunk` is 1000, the function will create chunks such that:
        - the first chunk contains token values < 1000,
        - the second chunk contains values >= 1000 and < 2000, and so on.

        Parameters:
            token_indices (`torch.Tensor` of shape `(seq_len, )`): A monotonically increasing list of
                                token index values.
            t_ntoken_per_chunk (`int`): Number of tokens per chunk (used as the chunk size threshold).
            remove_index (`int`) An index id to subtract from `token_indices` before chunking

        Returns:
            `List[Tuple[int, int]]`: A list of tuples, each representing the start (inclusive)
                                and end (exclusive) indices of a chunk in `token_indices`.
        """

        def _iter():
            i, start_idx = 0, 0  # skip bos token
            current_chunk = 1
            while i < len(token_indices):  # skip eos token
                if token_indices[i] - remove_index >= current_chunk * tokens_per_chunk:
                    yield (start_idx, i)
                    start_idx = i
                    current_chunk += 1
                i += 1
            yield (start_idx, len(token_indices))

        return list(_iter())

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.config.image_token_id
        patch_token_id = self.config.patch_token_id
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        image_start_token_id = self.config.image_start_token_id
        patch_start_token_id = self.config.patch_start_token_id
        video_start_token_id = self.config.video_start_token_id
        audio_start_token_id = self.config.audio_start_token_id
        position_id_per_seconds = self.config.position_id_per_seconds
        seconds_per_chunk = self.config.seconds_per_chunk

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None or audio_seqlens is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums, audio_nums, patch_nums = 0, 0, 0, 0
                audio_nums = torch.sum(input_ids == audio_start_token_id)
                image_nums = torch.sum(input_ids == image_start_token_id)
                patch_nums = torch.sum(input_ids == patch_start_token_id)
                video_nums = torch.sum(input_ids == video_start_token_id)
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_audios, remain_patchs = image_nums, video_nums, audio_nums, patch_nums
                multimodal_nums = (
                    image_nums + audio_nums + patch_nums if use_audio_in_video else image_nums + video_nums + audio_nums + patch_nums
                )
                for _ in range(multimodal_nums):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if patch_token_id in input_tokens and remain_patchs > 0:
                        ed_patch = input_tokens.index(patch_token_id, st)
                    else:
                        ed_patch = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if audio_token_id in input_tokens and remain_audios > 0:
                        ed_audio = input_tokens.index(audio_token_id, st)
                    else:
                        ed_audio = len(input_tokens) + 1
                    min_ed = min(ed_image, ed_video, ed_audio, ed_patch)
                    if min_ed == ed_audio:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = audio_seqlens[audio_idx]
                        llm_pos_ids = []
                        while len(llm_pos_ids) < audio_len:
                            pos_id = max(llm_pos_ids) + 1 if len(llm_pos_ids) > 0 else 0
                            llm_pos_ids += [pos_id] * 16 + [pos_id + 1] * 17 + [pos_id + 2] * 17
                        llm_pos_ids = llm_pos_ids[:audio_len]
                        llm_pos_ids = torch.tensor(llm_pos_ids).view(1, -1).expand(3, -1) + st_idx
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + audio_len + eos_len
                        audio_idx += 1
                        remain_audios -= 1

                    elif min_ed == ed_image:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_images -= 1

                    elif min_ed == ed_patch:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + image_len + eos_len
                        image_idx += 1
                        remain_patchs -= 1

                    elif min_ed == ed_video and not use_audio_in_video:
                        text_len = min_ed - st - 1
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                        llm_pos_ids_list.append(llm_pos_ids)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len + video_len + eos_len
                        video_idx += 1
                        remain_videos -= 1

                    elif min_ed == ed_video and use_audio_in_video:
                        text_len = min_ed - st - 2
                        if text_len != 0:
                            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        bos_len = 1
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        audio_len = ((audio_seqlens[audio_idx] - 1) // 2 + 1 - 2) // 2 + 1
                        audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                        ).long()
                        video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                            st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                        )

                        t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                        video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                        sub_len = 0
                        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                            video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                            audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                            if video_chunk_index is not None:
                                sub_len += video_chunk_index[1] - video_chunk_index[0]

                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                                )
                            if audio_chunk_index is not None:
                                sub_len += audio_chunk_index[1] - audio_chunk_index[0]

                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                                )
                        video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        eos_len = 1
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                        llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                        st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

            return position_ids, mrope_position_deltas


def classify_language(text):
    """Classifies a string as predominantly Chinese or English."""
    chinese_count = 0
    english_count = 0

    for char in text:
        # if re.match(r"\p{Han}", char, re.UNICODE):
        if re.match(r"[\u4E00-\u9FFF]", char, re.UNICODE):
            chinese_count += 1
        elif re.match(r"[a-zA-Z]", char):
            english_count += 1

    if chinese_count > english_count:
        return "zh"
        return "Predominantly Chinese"
    elif english_count > chinese_count:
        return "en"
        return "Predominantly English"
    else:
        return "zh"
        return "Mixed or Undetermined"


def render_sample_to_images(sample, tokenizer, visualizer=False):
    messages = sample["messages"]

    images = sample["images"]
    image_idx = 0

    new_images = []

    for j, message in enumerate(messages):
        content = message["content"]
        role = message["role"]
        if role == "user":
            pass
        else:
            continue

        # language = classify_language(content)
        language = "zh"

        image_nums = content.count("<|image|>")
        audio_nums = content.count("<|audio|>")
        video_nums = content.count("<|video|>")

        multimodal_nums = image_nums + audio_nums + video_nums

        new_content = ""
        text = ""
        st = 0

        for _ in range(multimodal_nums):
        
            if "<|image|>" in content:
                ed_image = content.index("<|image|>", st)
            else:
                ed_image = len(content) + 1

            if "<|audio|>" in content:
                ed_audio = content.index("<|audio|>", st)
            else:
                ed_audio = len(content) + 1

            if "<|video|>" in content:
                ed_video = content.index("<|video|>", st)
            else:
                ed_video = len(content) + 1

            min_ed = min(ed_image, ed_audio, ed_video)

            if min_ed == ed_image:
                text_len = min_ed - st

                image_len = len("<|image|>")

                if text_len > 0:
                    _text = content[st:min_ed]
                    _images = render_text_to_images(_text, tokenizer, language=language)
                    new_images.extend(_images)
                    new_content += "<|image|>" * len(_images)

                    if visualizer:
                        for i, img in enumerate(_images):
                            output_filename = f"/data/output/render_sample_to_images/{_text[:32].strip()}_{i+1}.png"
                            img.save(output_filename)
                            print(f"Image saved to {output_filename}")

                image = images[image_idx]
                new_images.append(image)
                new_content += "<|image|>"
                image_idx += 1

                st += text_len + image_len

            elif min_ed == ed_audio:
                text_len = min_ed - st

                audio_len = len("<|audio|>")

                if text_len > 0:
                    _text = content[st:min_ed]
                    _images = render_text_to_images(_text, tokenizer, language=language)
                    new_images.extend(_images)
                    new_content += "<|image|>" * len(_images)

                    if visualizer:
                        for i, img in enumerate(_images):
                            output_filename = f"/data/output/render_sample_to_images/{_text[:32].strip()}_{i+1}.png"
                            img.save(output_filename)
                            print(f"Image saved to {output_filename}")

                new_content += "<|audio|>"

                st += text_len + audio_len

            elif min_ed == ed_video:
                text_len = min_ed - st

                video_len = len("<|video|>")

                if text_len > 0:
                    _text = content[st:min_ed]
                    _images = render_text_to_images(_text, tokenizer, language=language)
                    new_images.extend(_images)
                    new_content += "<|image|>" * len(_images)

                    if visualizer:
                        for i, img in enumerate(_images):
                            output_filename = f"/data/output/render_sample_to_images/{_text[:32].strip()}_{i+1}.png"
                            img.save(output_filename)
                            print(f"Image saved to {output_filename}")

                new_content += "<|video|>"

                st += text_len + video_len

        if st < len(content):
            _text = content[st:]
            _images = render_text_to_images(_text, tokenizer, language=language)
            new_images.extend(_images)
            new_content += "<|image|>" * len(_images)

            if visualizer:
                for i, img in enumerate(_images):
                    output_filename = f"/data/output/render_sample_to_images/{_text[:32].strip()}_{i+1}.png"
                    img.save(output_filename)
                    print(f"Image saved to {output_filename}")

        messages[j]["content"] = new_content

    sample["messages"] = messages
    sample["images"] = new_images

    return sample


from PIL import Image, ImageDraw, ImageFont

def render_text_to_images(text, tokenizer, font_path=None, font_size=None, image_width=None, image_height=None, language="en"):
    """
    Renders text to one or more images, wrapping it to fit within the specified dimensions.

    Args:
        text (str): The text to be rendered.
        font_path (str): The path to the font file (e.g., '.ttf').
        font_size (int): The desired font size.
        image_width (int): The width of each output image.
        image_height (int): The height of each output image.

    Returns:
        list: A list of PIL Image objects, each containing a portion of the text.
    """
    if font_path == None:
        # font_path = "times.ttf"
        font_path = "simsun.ttc"

    if font_size == None:
        # font_size = 20
        font_size = 30

    if image_width == None:
        image_width = 448

    if image_height == None:
        image_height = 448

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Could not load font from {font_path}. Please check the path.")
        return []

    temp_image = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_image)

    lines = []

    input_ids = tokenizer(text)["input_ids"]

    current_line = []
    for input_id in input_ids:
        raw_word = tokenizer.decode(input_id)
        if "\n" in raw_word:

            split_word = raw_word.split("\n")
            for word_idx, word in enumerate(split_word):
                _input_ids = tokenizer(word).input_ids
                current_line = current_line + _input_ids

                if word_idx == len(split_word) - 1:
                    break

                lines.append(current_line)
                current_line = []

            continue

        test_line = current_line + [input_id]

        bbox = temp_draw.textbbox((0, 0), tokenizer.decode(test_line), font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= image_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = [input_id]

    if current_line:
        lines.append(current_line)

    images = []
    current_image = None
    draw = None
    y_offset = 0
    line_height = temp_draw.textbbox((0,0), "Ay", font=font)[3] # A bit of a hack to get line height

    for line in lines:
        if current_image is None or y_offset + line_height > image_height:
            if current_image:
                images.append(current_image)

            current_image = Image.new('RGB', (image_width, image_height), 'white')
            draw = ImageDraw.Draw(current_image)
            y_offset = 0

        draw.text((0, y_offset), tokenizer.decode(line), font=font, fill='black')
        y_offset += line_height

    if current_image:
        images.append(current_image)

    return images

# Example usage:
if __name__ == '__main__':
    # You'll need to provide a path to a TrueType font file (.ttf)
    # For example, you can download one from Google Fonts or use a system font.
    # On Windows, fonts are often in C:\Windows\Fonts\
    # On macOS, they are in /Library/Fonts/
    # On Linux, they are often in /usr/share/fonts/

    # You can get a simple font like 'arial.ttf' if it's available on your system.
    font_path = "simsun.ttc"  # Replace with the actual path to your font file

    sample_text = (
        "This is a long sample text that will be wrapped and rendered "
        "across multiple lines and potentially multiple images. The goal "
        "is to fit the text as tightly as possible within the given image dimensions. "
        "This is a simple demonstration of how to handle text wrapping and rendering "
        "using the Pillow library in Python. The quick brown fox jumps over the lazy dog."
    ) * 10
    language = "en"
    font_path = "times.ttf"

    # sample_text = "这是一个较长的示例文本，将被换行并渲染到多行，甚至可能渲染到多张图片中。目标是使文本尽可能紧密地适应给定的图片尺寸。这是一个简单的演示，演示如何使用 Python 中的 Pillow 库处理文本换行和渲染。敏捷的棕色狐狸跳过了懒狗" * 20
    # language = "zh"
    # font_path = "simsun.ttc"

    images = render_text_to_images(
        text=sample_text,
        font_path=font_path,
        font_size=20,
        image_width=448,
        image_height=448,
        language=language,
    )

    if images:
        for i, img in enumerate(images):
            output_filename = f"output_image_{i+1}.png"
            img.save(output_filename)
            print(f"Image saved to {output_filename}")
    else:
        print("No images were generated.")
