import torch
from TTS.api import TTS
import json
import io
import locale
from functools import partial

# 设置默认的locale为C.UTF-8
locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# 确保所有的文件操作都使用UTF-8编码
io.open = partial(io.open, encoding='utf8')

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

if 'tts' not in globals():
    # 模型未加载，现在加载模型
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print("Model Successfully loaded.")
else:
    # 模型已加载，无需重复操作
    print("Model already loaded.")

def voice_conversion(text, speaker_wav, language, output_path):
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_path)
    print("Voice conversion completed.")

# 配置文件路径
config_path = '/content/TTS/voice_conversion_config.json'  # 替换为您的配置文件路径

# 读取配置文件
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

# 调用函数
voice_conversion(config['text'], config['speaker_wav'], config['language'], config['output_path'])


