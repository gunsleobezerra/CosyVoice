import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['Pre-trained Voice', '3s Fast Cloning', 'Cross-lingual Cloning', 'Natural Language Control']
instruct_dict = {'Pre-trained Voice': '1. Select pre-trained voice\n2. Click the generate audio button',
                 '3s Fast Cloning': '1. Select prompt audio file or record prompt audio, note that it should not exceed 30s. If both are provided, the prompt audio file will be prioritized\n2. Enter prompt text\n3. Click the generate audio button',
                 'Cross-lingual Cloning': '1. Select prompt audio file or record prompt audio, note that it should not exceed 30s. If both are provided, the prompt audio file will be prioritized\n2. Click the generate audio button',
                 'Natural Language Control': '1. Select pre-trained voice\n2. Enter instruct text\n3. Click the generate audio button'}
stream_mode_list = [('No', False), ('Yes', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['Natural Language Control']:
        if cosyvoice.instruct is False:
            gr.Warning('You are using Natural Language Control mode, {} model does not support this mode, please use iic/CosyVoice-300M-Instruct model'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text == '':
            gr.Warning('You are using Natural Language Control mode, please enter instruct text')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Natural Language Control mode, prompt audio/prompt text will be ignored')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['Cross-lingual Cloning']:
        if cosyvoice.instruct is True:
            gr.Warning('You are using Cross-lingual Cloning mode, {} model does not support this mode, please use iic/CosyVoice-300M model'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using Cross-lingual Cloning mode, instruct text will be ignored')
        if prompt_wav is None:
            gr.Warning('You are using Cross-lingual Cloning mode, please provide prompt audio')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('You are using Cross-lingual Cloning mode, please ensure the synthesis text and prompt text are in different languages')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s Fast Cloning', 'Cross-lingual Cloning']:
        if prompt_wav is None:
            gr.Warning('Prompt audio is empty, did you forget to input prompt audio?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('Prompt audio sample rate {} is lower than {}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['Pre-trained Voice']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('You are using Pre-trained Voice mode, prompt text/prompt audio/instruct text will be ignored!')
        if sft_dropdown == '':
            gr.Warning('No available pre-trained voice!')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s Fast Cloning']:
        if prompt_text == '':
            gr.Warning('Prompt text is empty, did you forget to input prompt text?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('You are using 3s Fast Cloning mode, pre-trained voice/instruct text will be ignored!')

    if mode_checkbox_group == 'Pre-trained Voice':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3s Fast Cloning':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == 'Cross-lingual Cloning':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Codebase [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    Pre-trained models [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### Please enter the text to be synthesized, select the inference mode, and follow the instructions")

        tts_text = gr.Textbox(label="Enter synthesis text", lines=1, value="I am a new generative speech model launched by the Tongyi Lab speech team, providing comfortable and natural speech synthesis capabilities.")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='Select inference mode', value=inference_mode_list[0])
            instruction_text = gr.Text(label="Instructions", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='Select pre-trained voice', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='Stream inference', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="Speed adjustment (only supports non-streaming inference)", minimum=0.5, maximum=2.0, step=0.1)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2")
                seed = gr.Number(value=0, label="Random inference seed")

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='Select prompt audio file, note that the sample rate should not be lower than 16kHz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='Record prompt audio file')
        prompt_text = gr.Textbox(label="Enter prompt text", lines=1, placeholder="Please enter prompt text, it should match the content of the prompt audio, automatic recognition is not supported yet...", value='')
        instruct_text = gr.Textbox(label="Enter instruct text", lines=1, placeholder="Please enter instruct text.", value='')

        generate_button = gr.Button("Generate audio")

        audio_output = gr.Audio(label="Synthesized audio", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
