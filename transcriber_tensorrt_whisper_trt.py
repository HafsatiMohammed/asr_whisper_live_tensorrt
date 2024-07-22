import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union

import torch
import numpy as np
import torch.nn.functional as F
from whisper.tokenizer import get_tokenizer
from whisper_live.tensorrt_utils import (mel_filters, load_audio_wav_format, pad_or_trim, load_audio)
import librosa

'''
import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
'''


from whisper_trt.model import *  

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


class WhisperEncoding:

    def __init__(self, engine_dir):
        self.session = self.get_session(engine_dir)

    def get_session(self, engine_dir):
        config_path = engine_dir / 'encoder_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        use_gpt_attention_plugin = config['plugin_config'][
            'gpt_attention_plugin']
        dtype = config['builder_config']['precision']
        n_mels = config['builder_config']['n_mels']
        num_languages = config['builder_config']['num_languages']

        self.dtype = dtype
        self.n_mels = n_mels
        self.num_languages = num_languages

        serialize_path = engine_dir / f'whisper_encoder_{self.dtype}_tp1_rank0.engine'

        with open(serialize_path, 'rb') as f:
            session = Session.from_serialized_engine(f.read())

        return session

    def get_audio_features(self, mel):

        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device)

        inputs = OrderedDict()
        inputs['x'] = mel
        inputs['input_lengths'] = input_lengths

        output_list = [
            TensorInfo('x', str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo('input_lengths', str_dtype_to_trt('int32'),
                       input_lengths.shape)
        ]

        output_info = (self.session).infer_shapes(output_list)

        logger.debug(f'output info {output_info}')
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream()
        ok = self.session.run(inputs=inputs,
                              outputs=outputs,
                              stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'
        stream.synchronize()
        audio_features = outputs['output']
        return audio_features





class WhisperTRT(object):

    def __init__(self, model_name, assets_dir='~/.cache/whisper/', device=None, is_multilingual=False,
                 language="en", task="transcribe"):

        self.device = device
        self.model = load_trt_model(name= model_name, language=language, task=task)
        print(self.model)
        self.language = language
        self.task = task
        
        file_path = os.path.join(assets_dir,"mel_filters.npz")
        if not os.path.exists(file_path):
            os.makedirs(assets_dir, exist_ok=True)    
            
            
            np.savez_compressed(
                file_path,
                mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
            )   
            np.savez_compressed(
                file_path,
                mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            )            

        if 'large' in model_name:
            file_path = os.path.join(assets_dir,"mel_filters.npz")
            n_mels = 128 
            if not os.path.exists(file_path):
                np.savez_compressed(
                    file_path,
                    mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels),
                )            
        else:
            file_path = os.path.join(assets_dir,"mel_filters.npz") 
            n_mels = 80
            if not os.path.exists(file_path):
                np.savez_compressed(
                    file_path,
                    mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels),
                )



        self.filters = mel_filters(self.device, n_mels, assets_dir)


    def log_mel_spectrogram(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        padding: int = 0,
        return_duration=True
    ):
            """
            Compute the log-Mel spectrogram of

            Parameters
            ----------
            audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
                The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

            n_mels: int
                The number of Mel-frequency filters, only 80 and 128 are supported

            padding: int
                Number of zero samples to pad to the right

            device: Optional[Union[str, torch.device]]
                If given, the audio tensor is moved to this device before STFT

            Returns
            -------
            torch.Tensor, shape = (80 or 128, n_frames)
                A Tensor that contains the Mel spectrogram
            """
            if not torch.is_tensor(audio):
                if isinstance(audio, str):
                    if audio.endswith('.wav'):
                        audio, _ = load_audio_wav_format(audio)
                    else:
                        audio = load_audio(audio)
                assert isinstance(audio, np.ndarray), f"Unsupported audio type: {type(audio)}"
                duration = audio.shape[-1] / SAMPLE_RATE
                audio = pad_or_trim(audio, N_SAMPLES)
                audio = audio.astype(np.float32)
                audio = torch.from_numpy(audio) 
            if self.device is not None:
                audio = audio.to(self.device)
            if padding > 0:
                audio = F.pad(audio, (0, padding))
            window = torch.hann_window(N_FFT).to(audio.device)
            stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
            magnitudes = stft[..., :-1].abs()**2
            mel_spec = self.filters @ magnitudes

            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            if return_duration:
                return log_spec, duration
            else:
                return log_spec

    def transcribe(
            self,
            audio, 
            batch_size=1):
        mel = whisper.audio.log_mel_spectrogram(audio)[None, ...].to(self.device)

        mel , duration = self.log_mel_spectrogram(audio)
        mel = mel.unsqueeze(0)
        prediction = self.model.process_batch(mel, language=self.language, task=self.task)
        return prediction.strip() , duration



