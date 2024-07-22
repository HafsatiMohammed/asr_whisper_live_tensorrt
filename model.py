import os
import torch
import whisper
from whisper import load_model
from whisper.model import ModelDimensions, LayerNorm, Tensor
from whisper.tokenizer import Tokenizer
import torch2trt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import asdict
import tensorrt
import re
import subprocess
import numpy as np
import onnxruntime


__version__ = "1.0.0"

def get_cache_dir():
    return os.path.expanduser("~/.cache/whisper/")

def make_cache_dir():
    os.makedirs(get_cache_dir(), exist_ok=True)

class _AudioEncoderEngine(nn.Module):
    def __init__(self, conv1, conv2, blocks, ln_post):
        super().__init__()
        self.blocks = blocks
        self.conv1 = conv1
        self.conv2 = conv2
        self.ln_post = ln_post

    @torch.no_grad()
    def forward(self, x, positional_embedding):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = (x + positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x

class AudioEncoderTRT(nn.Module):
    def __init__(self, engine: torch2trt.TRTModule, positional_embedding: torch.Tensor):
        super().__init__()
        self.engine = engine
        self.register_buffer("positional_embedding", positional_embedding)

    @torch.no_grad()
    def forward(self, x: Tensor):
        n_audio_ctx = int(x.shape[2] // 2)
        pos_embed = self.positional_embedding[-n_audio_ctx:, :]
        x = self.engine(x, pos_embed)
        return x

class _TextDecoderEngine(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    @torch.no_grad()
    def forward(self, x, xa, mask):
        for block in self.blocks:
            x = block(x, xa, mask)
        return x


class TextDecoderTRT(nn.Module):
    def __init__(self, engine: torch2trt.TRTModule, token_embedding: nn.Embedding, positional_embedding: nn.Parameter, ln: nn.LayerNorm, mask: torch.Tensor):
        super().__init__()
        self.engine = engine
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.ln = ln
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, xa: Tensor):
        offset = 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        x = x.to(xa.dtype)
        x = self.engine(x, xa, self.mask)
        x = self.ln(x)
        logits = x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        return logits.float()

    @torch.no_grad()
    def generate(self, decoder_input_ids: Tensor, encoder_outputs: Tensor, eot_id: int, max_new_tokens: int = 40, num_beams: int = 1) :
        batch_size = decoder_input_ids.size(0)
        generated_ids = decoder_input_ids

        for _ in range(max_new_tokens):
            logits = self.forward(generated_ids, encoder_outputs)
            next_token_logits = logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            generated_ids = torch.cat((generated_ids, next_tokens), dim=-1)
            if next_tokens.item() == eot_id:
                break

        return generated_ids.tolist()

class WhisperTRT(nn.Module):
    def __init__(self, dims: ModelDimensions, encoder: AudioEncoderTRT, decoder: TextDecoderTRT, tokenizer: Tokenizer | None = None):
        super().__init__()
        self.dims = dims
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.audio_features = None

    def embed_audio(self, mel: Tensor):
        result = self.encoder(mel) 

        return result    
    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)
    
    def forward(self, mel: Tensor, tokens: Tensor):
        return self.decoder(tokens, self.encoder(mel))

    @torch.no_grad()
    def process_batch(self, mel, language, task,text_prefix = '' ,num_beams=1):
        # Define the start token and other task-specific tokens
        SOT_TOKEN = "<|startoftranscript|>"
        LANGUAGE_TOKEN = f"<|{language}|>"  # Assuming English, change this as per requirement
        TASK_TOKEN = f"<|{task}|>"
        TIMESTAMPS = "<|notimestamps|>"

        # Create the full prefix including start, language, and task tokens
        full_prefix = f"{SOT_TOKEN}{LANGUAGE_TOKEN}{TASK_TOKEN}{TIMESTAMPS}"

        # Encode the full prefix to get initial token IDs
        prompt_id = self.tokenizer.encode(
            full_prefix, allowed_special=set(self.tokenizer.special_tokens.keys()))

        # Convert to tensor and move to the appropriate device
        prompt_id = torch.tensor(prompt_id).cuda()
        # Repeat the prompt ID for each example in the batch
        batch_size = mel.shape[0]
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        # Get the audio features from the encoder
        encoder_output = self.embed_audio(mel)

        # Generate output IDs using the decoder
        output_ids = self.decoder.generate(decoder_input_ids,
                                           encoder_output,
                                           self.tokenizer.eot,
                                           max_new_tokens=96,
                                           num_beams=num_beams)
       
        # Decode the output IDs to text
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i]).strip()
            #print(text)
            texts.append(text)
        
        texts = re.sub(r'<\|.*?\|>', '', texts[0]) 
        #print(texts)   
        return texts



class WhisperTRTBuilder:
    model: str
    fp16_mode: bool = True
    max_workspace_size: int = 4 << 30
    verbose: bool = False

    @classmethod
    @torch.no_grad()
    def build_text_decoder_engine(cls) -> torch2trt.TRTModule:
        model = load_model(cls.model).cuda().eval()
        dims = model.dims
        decoder_blocks_module = _TextDecoderEngine(model.decoder.blocks)
        x = torch.randn(1, 1, dims.n_text_state).cuda()
        xa = torch.randn(1, dims.n_audio_ctx, dims.n_audio_state).cuda()
        mask = torch.randn(dims.n_text_ctx, dims.n_text_ctx).cuda()
        engine = torch2trt.torch2trt(
            decoder_blocks_module, 
            [x, xa, mask], 
            use_onnx=True, 
            min_shapes=[(1, 1, dims.n_text_state), (1, 1, dims.n_audio_state), (dims.n_text_ctx, dims.n_text_ctx)],
            opt_shapes=[(1, 1, dims.n_text_state), (1, dims.n_audio_ctx, dims.n_audio_state), (dims.n_text_ctx, dims.n_text_ctx)],
            max_shapes=[(1, dims.n_text_ctx, dims.n_text_state), (1, dims.n_audio_ctx, dims.n_audio_state), (dims.n_text_ctx, dims.n_text_ctx)],
            input_names=["x", "xa", "mask"],
            output_names=["output"],
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode,
            log_level=tensorrt.Logger.VERBOSE if cls.verbose else tensorrt.Logger.ERROR,
        )
        return engine

    @classmethod
    @torch.no_grad()
    def build_audio_encoder_engine(cls) -> torch2trt.TRTModule:
        model = load_model(cls.model).cuda().eval()
        dims = model.dims
        encoder_module = _AudioEncoderEngine(model.encoder.conv1, model.encoder.conv2, model.encoder.blocks, model.encoder.ln_post)
        n_frames = dims.n_audio_ctx * 2
        x = torch.randn(1, dims.n_mels, n_frames).cuda()
        positional_embedding = model.encoder.positional_embedding.cuda().detach()
        engine = torch2trt.torch2trt(
            encoder_module, 
            [x, positional_embedding], 
            use_onnx=True, 
            min_shapes=[(1, dims.n_mels, 1), (1, dims.n_audio_state)],
            opt_shapes=[(1, dims.n_mels, n_frames), (dims.n_audio_ctx, dims.n_audio_state)],
            max_shapes=[(1, dims.n_mels, n_frames), (dims.n_audio_ctx, dims.n_audio_state)],
            input_names=["x", "positional_embedding"],
            output_names=["output"],
            max_workspace_size=cls.max_workspace_size,
            fp16_mode=cls.fp16_mode,
            log_level=tensorrt.Logger.VERBOSE if cls.verbose else tensorrt.Logger.ERROR,
        )
        return engine

    @classmethod
    @torch.no_grad()
    def get_text_decoder_extra_state(cls):
        model = load_model(cls.model).cuda().eval()
        extra_state = {
            "token_embedding": model.decoder.token_embedding.state_dict(),
            "positional_embedding": model.decoder.positional_embedding,
            "ln": model.decoder.ln.state_dict(),
            "mask": model.decoder.mask
        }
        return extra_state
    
    @classmethod
    @torch.no_grad()
    def get_audio_encoder_extra_state(cls):
        model = load_model(cls.model).cuda().eval()
        extra_state = {
            "positional_embedding": model.encoder.positional_embedding
        }
        return extra_state

    @classmethod
    @torch.no_grad()
    def build(cls, output_path: str, verbose: bool = False):
        cls.verbose = verbose
        checkpoint = {
            "whisper_trt_version": __version__,
            "dims": asdict(load_model(cls.model).dims),
            "text_decoder_engine": cls.build_text_decoder_engine().state_dict(),
            "text_decoder_extra_state": cls.get_text_decoder_extra_state(),
            "audio_encoder_engine": cls.build_audio_encoder_engine().state_dict(),
            "audio_encoder_extra_state": cls.get_audio_encoder_extra_state()
        }
        torch.save(checkpoint, output_path)

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        model = load_model(cls.model)
        tokenizer = whisper.tokenizer.get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=task,
        )
        print(f"Tokenizer initialized for task: {task} and language: {language}")
        return tokenizer
    
    @classmethod
    @torch.no_grad()
    def load(cls, trt_model_path: str, language="fr", task="transcribe"):
        checkpoint = torch.load(trt_model_path)
        dims = ModelDimensions(**checkpoint['dims'])
        audio_encoder_engine = torch2trt.TRTModule()
        audio_encoder_engine.load_state_dict(checkpoint['audio_encoder_engine'])
        aes = checkpoint['audio_encoder_extra_state']
        audio_positional_embedding = aes['positional_embedding']
        encoder = AudioEncoderTRT(audio_encoder_engine, audio_positional_embedding)
        text_decoder_engine = torch2trt.TRTModule()
        text_decoder_engine.load_state_dict(checkpoint['text_decoder_engine'])
        tes = checkpoint['text_decoder_extra_state']
        text_token_embedding = nn.Embedding(dims.n_vocab, dims.n_text_state)
        text_token_embedding.load_state_dict(tes['token_embedding'])
        text_positional_embedding = nn.Parameter(tes['positional_embedding'])
        text_ln = LayerNorm(dims.n_text_state)
        text_ln.load_state_dict(tes['ln'])
        text_mask = tes['mask']
        decoder = TextDecoderTRT(text_decoder_engine, text_token_embedding, text_positional_embedding, text_ln, text_mask)
        whisper_trt = WhisperTRT(dims, encoder, decoder, cls.get_tokenizer(language, task))
        whisper_trt = whisper_trt.cuda().eval()
        return whisper_trt

class MultilingualBuilder(WhisperTRTBuilder):
    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        model = load_model(cls.model)
        tokenizer = whisper.tokenizer.get_tokenizer(
            True,
            num_languages=model.num_languages,
            language=language,
            task=task,
        )
        print(f"Tokenizer initialized for task: {task} and language: {language}")

        return tokenizer

class TinyMultilingualBuilder(MultilingualBuilder):
    model: str = "tiny"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)



class BaseMultilingualBuilder(MultilingualBuilder):
    model: str = "base"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)



class SmallMultilingualBuilder(MultilingualBuilder):
    model: str = "small"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)




class Large_v2_MultilingualBuilder(MultilingualBuilder):
    model: str = "large-v2"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)


class Large_v2_MultilingualBuilder(MultilingualBuilder):
    model: str = "large-v2"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)


class Large_v1_MultilingualBuilder(MultilingualBuilder):
    model: str = "large-v1"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)


class Large_v3_MultilingualBuilder(MultilingualBuilder):
    model: str = "large-v3"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)


class Medium_MultilingualBuilder(MultilingualBuilder):
    model: str = "medium"

    @classmethod
    def get_tokenizer(cls, language="fr", task="transcribe"):
        return super().get_tokenizer(language, task)



MODEL_FILENAMES = {

    "tiny": "tiny_multilingual_trt.pth",
    "base": "base_multilingual_trt.pth",
    "small": "small_multilingual_trt.pth",
    "large_v2" : "large_v2_multilingual_trt.pth",
    "large_v1" : "large_v1_multilingual_trt.pth",
    "large_v3" : "large_v3_multilingual_trt.pth",
    "medium" : "madium_multilingual_trt.pth"

}

MODEL_BUILDERS = {
    
    "tiny": TinyMultilingualBuilder,
   
    "base": BaseMultilingualBuilder,
   
    "small": SmallMultilingualBuilder,
   
    "large_v2": Large_v2_MultilingualBuilder,
    
    "large_v1": Large_v1_MultilingualBuilder,
    
    "large_v3": Large_v3_MultilingualBuilder,

    "medium": Medium_MultilingualBuilder,


}





def load_trt_model(name: str, path: str | None = None, build: bool = True, verbose: bool = False, language="en", task="transcribe"):
    print("Handled Models: Tiny, Base, Small, Medium, Large_v1, Large_v2, Large_v3")
    if name not in MODEL_BUILDERS:
        raise RuntimeError(f"Model '{name}' is not supported by WhisperTRT.")
    
    if path is None:
        path = os.path.join(get_cache_dir(), MODEL_FILENAMES[name])
        make_cache_dir()

    builder = MODEL_BUILDERS[name]

    if not os.path.exists(path):
        if not build:
            raise RuntimeError(f"No model found at {path}. Please call load_trt_model with build=True.")
        else:
            builder.build(path, verbose=verbose)

    # remove the download pth file 
    
    
    # file_name = name+'.pt'  
    # file_path = os.path.expanduser(f'~/.cache/whisper-live/{file_name}')

    # # Check if the file exists before attempting to remove it
    # if os.path.exists(file_path):
    #     os.remove(file_path)
    #     print(f'Removed {file_path}')
    # else:
    #     print(f'File {file_path} does not exist')
    return builder.load(path, language, task)







if __name__ == "__main__":
    # Usage example
    import librosa
    import shutil

    model_url = "https://raw.githubusercontent.com/snakers4/silero-vad/1baf307b35ab3bbb070ab374b43a0a3c3604fa2a/files/silero_vad.onnx"
    target_dir = os.path.expanduser("~/.cache/whisper/")
    # Ensure the target directory exists
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
    except: 
        pass
    # Define the target file path
    model_filename = os.path.join(target_dir, "silero_vad.onnx")
    # Check if the model file already exists
    if not os.path.exists(model_filename):
        # If it doesn't exist, download the model using wget
        try:
            subprocess.run(["wget", "-O", model_filename, model_url], check=True)
        except subprocess.CalledProcessError:
            print("Failed to download the model using wget.")

    # if os.path.exists('~/.cache/whisper'):
    # # If it exists, remove it
    #     shutil.rmtree('~/.cache/whisper')
   
    load_trt_model('tiny', language='fr', task='transcribe')
    
    # if os.path.exists('~/.cache/whisper'):
    # # If it exists, remove it
    #     shutil.rmtree('~/.cache/whisper')
   
    load_trt_model('base',  language='fr', task='transcribe')
    # if os.path.exists('~/.cache/whisper'):
    # # If it exists, remove it
    #     shutil.rmtree('~/.cache/whisper')
   
    load_trt_model('medium', language='fr', task='transcribe')
    # if os.path.exists('~/.cache/whisper'):
    # # If it exists, remove it
    #     shutil.rmtree('~/.cache/whisper')
   
    load_trt_model('large_v1', language='fr', task='transcribe')

    # if os.path.exists('~/.cache/whisper'):
    # # If it exists, remove it
    #     shutil.rmtree('~/.cache/whisper')
   
    load_trt_model('large_v2', language='fr', task='transcribe')
    
    # if os.path.exists('~/.cache/whisper'):
    # # If it exists, remove it
    #     shutil.rmtree('~/.cache/whisper')
   
    load_trt_model('large_v3', language='fr', task='transcribe')

    # if os.path.exists('~/.cache/whisper'):
    # # If it exists, remove it
    #     shutil.rmtree('~/.cache/whisper')

    # audio, sr = librosa.load('/usr/local/lib/python3.10/dist-packages/whisper_live/assets/jfk.flac', sr=16000)
    # mel = whisper.audio.log_mel_spectrogram(audio)[None, ...].cuda()
    # texts = model.process_batch(mel, language='fr', task='translate')
    # print(texts)

