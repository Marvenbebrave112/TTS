import torch
from TTS.api import TTS
import json
import argparse

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
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, output_path=output_path)

# 设置命令行参数解析
parser = argparse.ArgumentParser(description="Run voice conversion with config file.")
parser.add_argument('config_path', type=str, help='Path to the config file')
args = parser.parse_args()

# 读取配置文件
with open(args.config_path, 'r') as config_file:
    config = json.load(config_file)

# 调用函数
voice_conversion(config['text'], config['speaker_wav'], config['language'], config['output_path'])
