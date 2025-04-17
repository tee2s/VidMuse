import typing as tp
import warnings

import omegaconf
import torch

from .encodec import CompressionModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model, get_wrapped_compression_model
from .loaders import load_compression_model, load_lm_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition
from ..utils.autocast import TorchAutocast

# Define type aliases for melody-related inputs.
MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

# Backward compatible names mapping for model checkpoints.
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}

class VidMuse:
    """
    VidMuse models audio generation using a compression model and a language model.
    
    This class encapsulates the logic to generate music/audio from various conditioning methods 
    including text, melody, or even prompt audio for continuation.
    """

    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        """
        Initialize the VidMuse instance.
        
        Args:
            name (str): Identifier for the model.
            compression_model (CompressionModel): The model used for encoding/decoding audio tokens.
            lm (LMModel): The language model responsible for generating tokens.
            max_duration (Optional[float]): The maximum duration for generated audio (in seconds).
            
        Raises:
            ValueError: If max_duration is not provided and configuration is missing.
        """
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        self.cfg: tp.Optional[omegaconf.DictConfig] = None
        # Set both models to evaluation mode to disable training-specific behaviors.
        self.compression_model.eval()
        self.lm.eval()

        # If the LM model has a configuration, store it.
        if hasattr(lm, 'cfg'):
            cfg = lm.cfg
            assert isinstance(cfg, omegaconf.DictConfig)
            self.cfg = cfg

        # Wrap the compression model if configuration is available.
        if self.cfg is not None:
            self.compression_model = get_wrapped_compression_model(self.compression_model, self.cfg)

        # Determine the maximum duration for generation.
        if max_duration is None:
            if self.cfg is not None:
                max_duration = lm.cfg.dataset.segment_duration  # type: ignore
            else:
                raise ValueError("You must provide max_duration when building directly MusicGen")
        assert max_duration is not None
        self.max_duration: float = max_duration
        
        # Get the device from the LM model parameters (CPU or CUDA).
        self.device = next(iter(lm.parameters())).device

        # Set default generation parameters for a 15-second output.
        self.generation_params: dict = {}
        self.set_generation_params(duration=15)
        
        # Optional progress callback to report generation progress.
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        
        # Set up automatic casting for operations if using GPU.
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(enabled=True, device_type=self.device.type, dtype=torch.float16)

    @property
    def frame_rate(self) -> float:
        """
        Get the approximate number of autoregressive steps per second.
        
        Returns:
            float: The frame rate as defined by the compression model.
        """
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """
        Get the sample rate for the generated audio.
        
        Returns:
            int: Sample rate provided by the compression model.
        """
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """
        Get the number of audio channels for the generated audio.
        
        Returns:
            int: The number of channels (e.g., 1 for mono, 2 for stereo).
        """
        return self.compression_model.channels

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody', device=None):
        """
        Load and return a pretrained VidMuse model based on a model identifier.
        
        Available pretrained models include:
            - facebook/musicgen-small: Text-to-music, 300M parameters.
            - facebook/musicgen-medium: Text-to-music, 1.5B parameters.
            - facebook/musicgen-melody: Text and melody-to-music, 1.5B parameters.
            - facebook/musicgen-large: Text-to-music, 3.3B parameters.
        
        Args:
            name (str): The pretrained model identifier.
            device (Optional): The device to load the model on; if None, auto-select CUDA if available.
        
        Returns:
            VidMuse: An instance of VidMuse configured with the loaded pretrained models.
        """
        # Determine device if not provided.
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        # For debugging/unit tests.
        if name == 'debug':
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return VidMuse(name, compression_model, lm, max_duration=30)

        # Map shorthand names to full model identifiers.
        if name in _HF_MODEL_CHECKPOINTS_MAP:
            # Optionally, a warning could be issued here.
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        # Load language and compression models.
        lm = load_lm_model(name, device=device)
        compression_model = load_compression_model(name, device=device)
        
        # Adjust conditions for self-generated wave if available.
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
            lm.condition_provider.conditioners['self_wav']._use_masking = False
        
        return VidMuse(name, compression_model, lm, max_duration=30)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, extend_stride: float = 29.5):
        """
        Set parameters that control the audio token generation process.
        
        Args:
            use_sampling (bool): Whether to use sampling (True) or argmax (False) for decoding.
            top_k (int): The top_k parameter for sampling.
            top_p (float): The top_p parameter for sampling (if 0, top_k is used).
            temperature (float): Temperature parameter for softmax.
            duration (float): Total duration (in seconds) of generated audio.
            cfg_coef (float): Classifier free guidance coefficient.
            two_step_cfg (bool): Whether to use two-step forward for guidance.
            extend_stride (float): Stride to extend the audio for generation beyond max_duration.
        
        Raises:
            AssertionError: If extend_stride is not less than max_duration.
        """
        # Ensure stride does not exceed maximum generation duration.
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        
        # Set generation parameters for the language model.
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }

    def set_custom_progress_callback(self, progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None):
        """
        Register a custom callback function to monitor generation progress.
        
        Args:
            progress_callback (Optional[Callable[[int, int], None]]): A function that accepts the number
                of generated tokens and total tokens as arguments.
        """
        self._progress_callback = progress_callback

    def generate_unconditional(self, num_samples: int, progress: bool = False,
                               return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                        tp.Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate audio samples without any text conditioning.
        
        This method produces audio tokens unconditionally.
        
        Args:
            num_samples (int): Number of audio samples to generate.
            progress (bool): Whether to display generation progress.
            return_tokens (bool): If True, also return the generated tokens along with audio.
        
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Generated audio waveform and,
                optionally, the corresponding tokens.
        """
        # Generate a list of "None" to represent no conditioning descriptions.
        descriptions: tp.List[tp.Optional[torch.Tensor]] = [None] * num_samples
        
        # Prepare conditioning attributes from descriptions.
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        
        # Generate tokens given the (None) prompt and attributes.
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        
        # Convert tokens to waveform.
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate(self, descriptions_list: tp.List, progress: bool = False, return_tokens: bool = False) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate audio samples conditioned on text.
        
        Note: This method expects a list of two elements:
            - local descriptions (e.g., for video or localized conditions)
            - global descriptions (e.g., overall text condition)
        
        Args:
            descriptions_list (list): A list containing exactly two elements which are text conditions.
            progress (bool): Whether to display generation progress.
            return_tokens (bool): If True, also return the generated tokens.
        
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Generated audio waveform and, optionally, tokens.
        """
        # Enforce that descriptions_list is a list of exactly two elements.
        assert isinstance(descriptions_list, list)
        assert len(descriptions_list) <= 2
        assert len(descriptions_list) == 2

        # Split descriptions into local and global.
        local_descriptions = [descriptions_list[0]]
        global_descriptions = [descriptions_list[1]]

        # Stack descriptions into tensors (assumes descriptions are already in tensor form).
        local_attributes = torch.stack(local_descriptions)
        global_attributes = torch.stack(global_descriptions)

        prompt_tokens = None
        # Make sure there is no prompt tokens available.
        assert prompt_tokens is None

        # Generate tokens using the local and global conditioning attributes.
        tokens = self._generate_tokens([local_attributes, global_attributes], prompt_tokens, progress)
        
        # Decode tokens into audio waveform.
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_with_chroma(self, descriptions: tp.List[torch.Tensor], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate audio samples conditioned on both text and melody.
        
        Args:
            descriptions (List[torch.Tensor]): A list of text condition tensors.
            melody_wavs (torch.Tensor or List[torch.Tensor]): Waveform tensor(s) used for melody conditioning.
                Must have shape [B, C, T] or [C, T] (for single sample) or a list of tensors.
            melody_sample_rate (int): The sample rate of the input melody waveforms.
            progress (bool): Whether to display progress during generation.
            return_tokens (bool): If True, also return the generated tokens.
        
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Generated audio waveform and, optionally, tokens.
        """
        # Ensure melody inputs are in list form and in the correct shape.
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        # Convert melody audio to the model's sample rate.
        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]

        # Prepare conditioning attributes including melody.
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        
        # Generate tokens conditioned on both text and melody.
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[torch.Tensor]]] = None,
                              progress: bool = False, return_tokens: bool = False) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate audio that continues from a given audio prompt.
        
        Args:
            prompt (torch.Tensor): Waveform tensor used as the starting prompt (shape: [B, C, T] or [C, T]).
            prompt_sample_rate (int): Sample rate of the prompt waveform.
            descriptions (Optional[List[Optional[torch.Tensor]]]): Text conditions for each prompt. Defaults to None.
            progress (bool): Whether to display generation progress.
            return_tokens (bool): If True, also return the generated tokens.
        
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Generated audio continuation and,
                optionally, the tokens.
        
        Raises:
            ValueError: If prompt does not have the required dimensions.
        """
        # Ensure prompt tensor has three dimensions [B, C, T].
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        
        # Convert the prompt audio to match the model's sample rate.
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        
        # If no descriptions are provided, default to None for each prompt sample.
        if descriptions is None:
            descriptions = [None] * len(prompt)
        
        # Prepare tokens and conditioning attributes.
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        
        # Generate tokens using the prompt.
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """
        Prepare conditioning attributes and encode the audio prompt (if provided).
        
        Args:
            descriptions (Sequence[Optional[str]]): List of text descriptions used for conditioning.
            prompt (Optional[torch.Tensor]): Audio prompt for continuation (if any).
            melody_wavs (Optional[MelodyList]): List of melody waveforms for melody conditioning.
        
        Returns:
            Tuple:
                - List of ConditioningAttributes containing text (and optionally melody) information.
                - Encoded prompt tokens (if prompt is given), otherwise None.
        """
        # Create conditioning attributes from text descriptions.
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            # If no melody is provided, use a default silent wave.
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            # Check that the model supports melody conditioning.
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), (
                f"number of melody wavs must match number of descriptions! "
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}")
            # Attach melody conditions to each attribute.
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        # If a prompt is provided, encode it using the compression model.
        if prompt is not None:
            if descriptions is not None:
                # Ensure number of provided prompts matches number of descriptions.
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List,
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """
        Generate discrete audio tokens based on conditions and optional audio prompt.
        
        This method either generates tokens in one step (if the requested duration is within max_duration)
        or in iterative strided steps for longer generations.
        
        Args:
            attributes (List): List of ConditioningAttributes for the generation (text and/or melody).
            prompt_tokens (Optional[torch.Tensor]): Pre-encoded audio prompt tokens (if available).
            progress (bool): Whether to show progress updates.
        
        Returns:
            torch.Tensor: Generated tokens representing discrete audio codes.
        """
        # For iterative generation, reset max_duration to 30 seconds.
        self.max_duration = 30
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        # Define a progress callback to update progress.
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            # Adjust generated tokens count.
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        # If prompt exists, check that it is not longer than allowed.
        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], "Prompt is longer than audio to generate"

        callback = _progress_callback if progress else None

        # Simple generation when duration is within max allowed duration.
        if self.duration <= self.max_duration:
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)
        else:
            # Extended generation: generate in strides for durations longer than max_duration.
            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            all_tokens = []
            # If no prompt tokens, initial prompt length is zero.
            prompt_length = 0 if prompt_tokens is None else prompt_tokens.shape[-1]
            
            # Determine stride length in token space.
            stride_tokens = int(self.frame_rate * self.extend_stride)
            self.fps = 2  # Fixed frame rate for video frames.
            stride_video_frames = int(self.fps * self.extend_stride)

            # Iteratively generate tokens until reaching total desired length.
            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                # Determine the length of the current chunk.
                chunk_duration = min(self.duration - time_offset, self.max_duration) 
                max_gen_len = int(chunk_duration * self.frame_rate)

                with self.autocast:
                    # For extended generation, attributes are expected to be a list with two elements.
                    assert len(attributes) == 2
                    gen_tokens = self.lm.generate(
                        prompt_tokens, [attributes[0][:, :, :int(chunk_duration * self.fps), :, :], attributes[1]],
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)

                # Append new tokens (skip previously generated prompt portion if any).
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                
                # Update prompt_tokens with the new segment tokens for continuity.
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                
                # Update attributes to retain the most recent context for continued generation.
                if attributes[0].shape[2] - stride_video_frames < self.max_duration * self.fps:
                    attributes[0] = attributes[0][:, :, -self.max_duration * self.fps:, :, :]
                else:
                    attributes[0] = attributes[0][:, :, stride_video_frames:, :, :]
                current_gen_offset += stride_tokens

            # Concatenate all generated tokens.
            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

    def generate_audio(self, gen_tokens: torch.Tensor):
        """
        Decode discrete tokens into a continuous audio waveform.
        
        Args:
            gen_tokens (torch.Tensor): Generated tokens with shape [B, C, T].
        
        Returns:
            torch.Tensor: The generated audio waveform.
        
        Raises:
            AssertionError: If the generated tokens do not have the expected dimensions.
        """
        # Ensure tokens tensor has three dimensions.
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            # Decode tokens back to audio waveform using the compression model.
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio