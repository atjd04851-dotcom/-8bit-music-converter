import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import yt_dlp
from scipy.io.wavfile import write as write_wav

st.set_page_config(page_title="8-Bit Music Converter", page_icon="🎮")
st.title("🎮 8-Bit Music Converter")
st.write("YouTube URLから音声を取得できる環境では曲を解析し、ファミコン風の電子音に再構成します。")
st.caption("自分が権利を持つ曲、または変換の許可を得た音源だけを使用してください。YouTubeの利用規約や地域の法令にも従ってください。")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
uploaded = st.file_uploader("または音声ファイルをアップロード（WAV / MP3 / M4Aなど）", type=["wav", "mp3", "m4a", "ogg", "flac"])
duration_limit = st.slider("解析する長さ（分）", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
tempo = st.slider("出力テンポ", min_value=70, max_value=180, value=120, step=1)
pulse_mix = st.slider("矩形波の明るさ", min_value=0.15, max_value=0.95, value=0.50, step=0.05)
add_bass = st.checkbox("簡易ベースを追加", value=True)

st.warning("この試作版は曲全体を完全コピーするものではありません。音声から主旋律らしい音程を推定するため、ボーカルや複雑な編曲では誤検出があります。")

def synth_pulse(freq, length, sr, duty=0.5, volume=0.22):
    n = max(1, int(length * sr))
    t = np.arange(n) / sr
    phase = (t * freq) % 1.0
    wave = np.where(phase < duty, 1.0, -1.0)
    # small fade prevents clicks at note boundaries
    fade = min(int(sr * 0.008), n // 2)
    env = np.ones(n)
    if fade > 0:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
    return (wave * env * volume).astype(np.float32)

def synth_triangle(freq, length, sr, volume=0.12):
    n = max(1, int(length * sr))
    t = np.arange(n) / sr
    phase = (t * freq) % 1.0
    wave = 1.0 - 4.0 * np.abs(phase - 0.5)
    fade = min(int(sr * 0.008), n // 2)
    env = np.ones(n)
    if fade > 0:
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
    return (wave * env * volume).astype(np.float32)

def make_chip_tune(audio_path, max_seconds, out_tempo, duty, bass_enabled):
    sr = 22050
    y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_seconds)
    if len(y) < sr:
        raise ValueError("音声が短すぎます。")
    y = librosa.util.normalize(y)

    # HPSS helps reduce some percussive content before pitch tracking.
    harmonic, _ = librosa.effects.hpss(y)
    hop = 256
    pitches, magnitudes = librosa.piptrack(
        y=harmonic, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"), threshold=0.12
    )
    idx = magnitudes.argmax(axis=0)
    freq = pitches[idx, np.arange(pitches.shape[1])]
    strength = magnitudes[idx, np.arange(magnitudes.shape[1])]
    midi = np.zeros_like(freq)
    valid = (freq > 0) & (strength > np.percentile(strength, 35))
    midi[valid] = np.rint(librosa.hz_to_midi(freq[valid]))
    # Smooth tiny jumps and silence gaps; quantize to a simple grid.
    for i in range(1, len(midi) - 1):
        if midi[i] and midi[i - 1] == midi[i + 1]:
            midi[i] = midi[i - 1]

    seconds_per_beat = 60.0 / out_tempo
    grid = seconds_per_beat / 2.0
    total_seconds = len(y) / sr
    out = np.zeros(int(total_seconds * sr), dtype=np.float32)

    # Summarize the most common pitch in each half-beat cell.
    starts = np.arange(0, total_seconds, grid)
    for cell_i, start in enumerate(starts):
        end = min(start + grid, total_seconds)
        a = max(0, int(start * sr / hop))
        b = min(len(midi), max(a + 1, int(end * sr / hop)))
        candidates = midi[a:b]
        candidates = candidates[candidates > 0]
        if len(candidates) == 0:
            continue
        # Use median pitch to avoid octave flicker.
        note = int(np.median(candidates))
        note = int(np.clip(note, 45, 96))
        f = float(librosa.midi_to_hz(note))
        chunk = synth_pulse(f, end - start, sr, duty=duty, volume=0.20)
        pos = int(start * sr)
        n = min(len(chunk), len(out) - pos)
        if n > 0:
            out[pos:pos+n] += chunk[:n]

        if bass_enabled and cell_i % 2 == 0:
            # A deliberately simple octave-lower supporting tone.
            bass_note = max(33, note - 12)
            bass = synth_triangle(float(librosa.midi_to_hz(bass_note)), end-start, sr, volume=0.08)
            if n > 0:
                out[pos:pos+n] += bass[:n]

    # Normalize and apply a soft limiter.
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0:
        out = out / max(1.0, peak / 0.92)
    return sr, (np.clip(out, -1, 1) * 32767).astype(np.int16)

def download_youtube_audio(video_url, target_dir):
    output_template = str(Path(target_dir) / "source.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        file_path = ydl.prepare_filename(info)
    if not os.path.exists(file_path):
        candidates = list(Path(target_dir).glob("source.*"))
        if not candidates:
            raise FileNotFoundError("取得した音声ファイルが見つかりませんでした。")
        file_path = str(candidates[0])
    return file_path, info.get("title", "converted_track")

if st.button("8ビットに変換", type="primary"):
    if not uploaded and not url.strip():
        st.error("YouTube URLを入力するか、音声ファイルをアップロードしてください。")
    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                if uploaded:
                    source_path = str(Path(temp_dir) / uploaded.name)
                    with open(source_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    track_title = Path(uploaded.name).stem
                else:
                    with st.spinner("音声を取得しています…"):
                        source_path, track_title = download_youtube_audio(url.strip(), temp_dir)

                with st.spinner("音程を解析してチップチューンに再構成しています…"):
                    sample_rate, pcm = make_chip_tune(
                        source_path, duration_limit * 60, tempo, pulse_mix, add_bass
                    )
                    output_path = str(Path(temp_dir) / "8bit_converted.wav")
                    write_wav(output_path, sample_rate, pcm)
                    audio_bytes = Path(output_path).read_bytes()

            st.success(f"変換完了：{track_title}")
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                "変換したWAVをダウンロード",
                data=audio_bytes,
                file_name="8bit_converted.wav",
                mime="audio/wav",
            )
        except Exception as exc:
            st.error("変換に失敗しました。URLが利用可能か、または音声ファイル形式を確認してください。")
            st.code(str(exc))
            st.info("YouTubeからの取得ができない場合は、利用許可のある音声ファイルをアップロードして試してください。")
