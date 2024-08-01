import os
from IPython.display import Audio
import os
import re
import glob
import json
import tempfile
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import commons
import utils
import argparse
import subprocess
from data_utils import (
    TextAudioLoader,
    TextAudioCollate,
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)

from models import SynthesizerTrn
from scipy.io.wavfile import write
import time


LANG = "ind"
ckpt_dir = "./ind"
wav_path = "./audio/test.wav"


def preprocess_char(text, lang=None):
    """
    Special treatement of characters in certain languages
    """
    print(lang)
    if lang == "ron":
        text = text.replace("ț", "ţ")
    return text


class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [
            x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()
        ]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        """
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt


def preprocess_text(txt, text_mapper, hps, uroman_dir=None, lang=None):
    txt = preprocess_char(txt, lang=lang)
    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"
    if is_uroman:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                print(cmd)
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print(f"uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl)
            print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt


def genaration():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Run inference with {device}")
    print(f"Run model with {ckpt_dir}")

    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"

    print(os.getcwd())

    print("config file is" + str(os.path.isfile(config_file)))
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g.to(device)
    _ = net_g.eval()

    g_pth = f"{ckpt_dir}/G_100000.pth"
    print(f"load {g_pth}")

    _ = utils.load_checkpoint(g_pth, net_g, None)

    # txt = "Halo, selamat datang di Layanan Universal BPR. Saya adalah asisten virtual anda saat ini. apakah penelponan ini bertujuan untuk keperluan pribadi atau kerperluan bisnis ?"
    txt = "Memperluas cakupan bahasa dari teknologi wicara berpotensi meningkatkan akses informasi bagi lebih banyak orang"


    print(f"text: {txt}")
    txt = preprocess_text(txt, text_mapper, hps, lang=LANG)
    stn_tst = text_mapper.get_text(txt, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        hyp = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
            )[0][0, 0]
            .cpu()
            .float()
            .numpy()
        )

    print(f"Generated audio")
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    print(f"wav: {wav_path}")
    write(wav_path, hps.data.sampling_rate, hyp)


if __name__ == "__main__":
    start_time = time.time()
    genaration()
    end_time = time.time()
    device = torch.cuda.is_available()
    print(" 生成时间: ", end_time - start_time, f",是否使用 GPU {device}")
