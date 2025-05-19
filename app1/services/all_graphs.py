import librosa
import numpy as np
from sklearn.cluster import KMeans
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import pyloudnorm as pyln
from app1.utils.logging_config import logging
from app1.services.transcription import transcribe_audio_file
from app1.services.diarization import diarize_with_speechbrain
from app1.services.speaker_stats import count_words, extract_pitch
from app1.utils.audio_utils import get_speaker_embedding


def plot_audio_with_speakers_akshat(file_path, sr=10, threshold=90):
    try:
        speaker_dict=diarize_with_speechbrain(file_path)
        
        # Loading audio file
        y, sr = librosa.load(file_path, sr=sr)
        y_db = abs(librosa.amplitude_to_db(y))
        x = np.arange(len(y_db)) / sr

        # Plot setup
        plt.figure(figsize=(15, 5))

        # Plot amplitude with threshold coloring
        for i in range(1, len(y_db)):
            x_segment = [x[i-1], x[i]]
            y_segment = [y_db[i-1], y_db[i]]

            if y_segment[0] <= threshold and y_segment[1] <= threshold:
                plt.plot(x_segment, y_segment, color='green')
            elif y_segment[0] >= threshold and y_segment[1] >= threshold:
                plt.plot(x_segment, y_segment, color='red')
            else:
                slope = (y_segment[1] - y_segment[0]) / (x_segment[1] - x_segment[0])
                x_cross = x_segment[0] + ((threshold - y_segment[0]) / slope)
                if y_segment[0] <= threshold:
                    plt.plot([x_segment[0], x_cross], [y_segment[0], threshold], color='green')
                    plt.plot([x_cross, x_segment[-1]], [threshold, y_segment[-1]], color='red')
                else:
                    plt.plot([x_segment[0], x_cross], [y_segment[0], threshold], color='red')
                    plt.plot([x_cross, x_segment[-1]], [threshold, y_segment[-1]], color='green')

        # Speaker bars
        for ele in speaker_dict:
            start, end = ele['start'], ele['end']
            speaker_color = 'blue' if ele['speaker'] == 'Speaker_0' else 'orange'
            plt.plot(np.linspace(start, end, 100), [y_db.min()-3]*100, color=speaker_color, linewidth=10)

        # WPM bars
        for ele in speaker_dict:
            start, end = ele['start'], ele['end']
            wpm = (len(ele['text'].split()) / (end - start)) * 60
            color = 'darkgreen' if wpm > 160 else 'red'
            plt.plot(np.linspace(start, end, 100), [y_db.max()+5]*100, color=color, linewidth=10)

        #  Legend
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Below Threshold'),
            Line2D([0], [0], color='red', lw=2, label='Above Threshold'),
            Line2D([0], [0], color='blue', lw=2, label='Speaker 1'),
            Line2D([0], [0], color='orange', lw=2, label='Speaker 2')
        ]
        plt.legend(handles=legend_elements)
        plt.ylim(30, y_db.max() + 10)
        plt.grid()
        plt.show()
    except Exception as e:
        logging.error(f"Error in Akshat's plot: {e}")
        raise


async def plot_audio_with_speakers_bharath(audio_path, lufs_threshold_value=18):

    try:
        logging.info(f"Starting Bharath's plot for audio file: {audio_path}")

        # Resample transform for audio to 16kHz mono
        resampler = T.Resample(orig_freq=44100, new_freq=16000)

        def compute_lufs(audio_segment, sample_rate):
            audio_np = audio_segment.numpy()
            if audio_segment.shape[0] > 1:
                audio_np = audio_np[0]
            meter = pyln.Meter(sample_rate)
            try:
                lufs = meter.integrated_loudness(audio_np)
                return lufs if np.isfinite(lufs) else -np.inf
            except:
                return -np.inf

        def detect_overlaps(segments):
            overlaps = []
            for i, seg1 in enumerate(segments):
                for j, seg2 in enumerate(segments[i+1:], start=i+1):
                    start1, end1 = seg1["start"], seg1["end"]
                    start2, end2 = seg2["start"], seg2["end"]
                    if start1 <= end2 and start2 <= end1:
                        overlap_start = max(start1, start2)
                        overlap_end = min(end1, end2)
                        overlaps.append({"start": overlap_start, "end": overlap_end})
            return overlaps

        # Step 1: Transcribe with Whisper
        logging.info("Transcribing audio file...")
        transcription_result = await transcribe_audio_file(audio_path)
        segments = transcription_result["segments"]
        logging.info(f"Transcription completed. Number of segments: {len(segments)}")

        # Step 2: Perform diarization and dynamically determine the number of speakers
        logging.info("Performing diarization...")
        diarized_segments = diarize_with_speechbrain(audio_path, dynamic_speakers=True)
        unique_speakers = sorted(set(seg["speaker"] for seg in diarized_segments))
        num_speakers = len(unique_speakers)
        logging.info(f"Diarization completed. Number of speakers: {num_speakers}")
        print("number of speakers:",num_speakers)

        # Step 3: Loading the full audio
        logging.info("Loading audio file...")
        waveform, sample_rate = torchaudio.load(audio_path)
        logging.info(f"Audio file loaded. Sample rate: {sample_rate}, Waveform shape: {waveform.shape}")

        # Step 4: Extract embeddings
        logging.info("Extracting embeddings...")
        embeddings = []
        valid_segments = []
        for seg in diarized_segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]
            emb = get_speaker_embedding(segment_audio, sample_rate)
            if emb is not None:
                embeddings.append(emb)
                valid_segments.append(seg)

        if len(embeddings) == 0:
            raise ValueError("No valid embeddings found for clustering.")

        embeddings = np.vstack(embeddings)
        diarized_segments = valid_segments  
        logging.info(f"Embeddings extracted. Shape: {embeddings.shape}")
        logging.info(f"Valid diarized segments: {len(diarized_segments)}")

        # Step 5: Cluster embeddings into the determined number of speakers
        logging.info("Clustering embeddings...")
        labels = KMeans(n_clusters=num_speakers, random_state=0).fit_predict(embeddings)
        logging.info(f"Clustering completed. Labels: {labels}")

        # Step 6: Stats 
        logging.info("Calculating speaker stats...")
        for seg, label in zip(diarized_segments, labels):
            speaker = f"Speaker_{label}"
            seg["speaker"] = speaker
            text = seg.get("text", "")
            duration = seg["end"] - seg["start"]
            word_count = count_words(text)
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]
            avg_f0 = extract_pitch(segment_audio, sample_rate)
            lufs = compute_lufs(segment_audio, sample_rate)
            seg.update({
                "lufs": lufs,
                "speaking_rate": word_count / duration if duration > 0 else 0.0
            })
        logging.info("Speaker stats calculated.")

        # Step 7: Determine LUFS threshold
        if lufs_threshold_value is not None:
            lufs_threshold = lufs_threshold_value
        else:
            valid_lufs = [seg["lufs"] for seg in diarized_segments if seg["lufs"] != -np.inf]
            lufs_threshold = np.mean(valid_lufs) + (1.5 * np.std(valid_lufs)) if valid_lufs else -18.0
            print("LUFS threshold:",lufs_threshold)
        logging.info(f"LUFS threshold determined: {lufs_threshold:.2f} LUFS")
    
        # Prepare for plotting
        logging.info("Preparing for plotting...")
        if waveform.shape[0] > 1:
            waveform = waveform[0]
        waveform_np = waveform.numpy()
        times = np.arange(len(waveform_np)) / sample_rate

        # Dynamically assigning colors for speakers
        logging.info(f"Assigning colors for {num_speakers} speakers...")
        base_colors = ["blue", "green", "black", "brown", "pink", "yellow", 'orange']  
        if num_speakers > len(base_colors):
            logging.warning(f"Number of speakers ({num_speakers}) exceeds predefined colors. Colors may repeat.")
        speaker_colors = {speaker: base_colors[i % len(base_colors)] for i, speaker in enumerate(unique_speakers)}
        logging.info(f"Speaker colors assigned: {speaker_colors}")

        loud_color = "red"
        fast_speech_color = "red"
        overlap_color = "purple"

        speaking_rate_threshold = 3.5
        scaling_factor = 0.1
        min_height = 0.1

        # Plotting logic
        logging.info("Starting plotting...")
        fig, (ax_wave, ax_speaker_speed) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

        ax_wave.plot(times, waveform_np, color='gray', alpha=0.3, label='Background Waveform')

        for seg in diarized_segments:
            start_time = seg["start"]
            end_time = seg["end"]
            speaker = seg["speaker"]
            lufs = seg["lufs"]
            base_color = speaker_colors[speaker]
            color = loud_color if lufs != -np.inf and lufs > lufs_threshold else base_color
            linewidth = 2 if lufs != -np.inf and lufs > lufs_threshold else 1
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            seg_times = times[start_idx:end_idx]
            seg_waveform = waveform_np[start_idx:end_idx]
            ax_wave.plot(seg_times, seg_waveform, color=color, linewidth=linewidth)
            logging.debug(f"Plotted segment for {speaker}: {start_time}-{end_time}")

        ax_wave.set_title("Continuous Waveform with Speaker Diarization and Loudness (LUFS)")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.grid(True)

        overlaps = detect_overlaps(diarized_segments)

        ax_speaker_speed.set_yticks(np.arange(len(unique_speakers)))
        ax_speaker_speed.set_yticklabels(unique_speakers)
        ax_speaker_speed.set_ylim(-0.5, len(unique_speakers) - 0.5)
        ax_speaker_speed.set_xlabel("Time (s)")
        ax_speaker_speed.set_ylabel("Speaker")
        ax_speaker_speed.set_title("Speaker Activity and Speaking Rate")
        ax_speaker_speed.grid(True, axis='x')

        for seg in diarized_segments:
            start = seg["start"]
            end = seg["end"]
            speaker = seg["speaker"]
            speaking_rate = seg["speaking_rate"]
            speaker_index = unique_speakers.index(speaker)
            speed_color = fast_speech_color if speaking_rate > speaking_rate_threshold else speaker_colors[speaker]
            height = max(speaking_rate * scaling_factor, min_height)
            y_center = speaker_index
            rect = patches.Rectangle((start, y_center - height / 2), end - start, height,
                                     edgecolor='none', facecolor=speed_color, alpha=0.7)
            ax_speaker_speed.add_patch(rect)
            logging.debug(f"Plotted speaking rate for {speaker}: {start}-{end}")

        for overlap in overlaps:
            ax_speaker_speed.axvspan(overlap["start"], overlap["end"], color=overlap_color, alpha=0.3,
                                     label='Overlap' if overlap == overlaps[0] else "")
            logging.debug(f"Plotted overlap: {overlap['start']}-{overlap['end']}")

        legend_elements_wave = [
            Line2D([0], [0], color='gray', alpha=0.3, label='Background Waveform'),
            *[Line2D([0], [0], color=speaker_colors[sp], lw=1, label=f'{sp}') for sp in unique_speakers],
            Line2D([0], [0], color=loud_color, lw=2, label='Loud Segments')
        ]
        ax_wave.legend(handles=legend_elements_wave, loc='upper right')

        legend_elements_speed = [
            *[patches.Patch(facecolor=speaker_colors[sp], edgecolor='none', alpha=0.7, label=f'{sp}') for sp in unique_speakers],
            patches.Patch(facecolor=fast_speech_color, edgecolor='none', alpha=0.7,
                          label=f'Fast Speech (>{speaking_rate_threshold:.1f} wps)'),
            patches.Patch(facecolor=overlap_color, edgecolor='none', alpha=0.3, label='Speech Overlap')
        ]
        ax_speaker_speed.legend(handles=legend_elements_speed, loc='upper right')

        plt.tight_layout()
        plt.show()
        logging.info("Plotting completed successfully.")
    except Exception as e:
        logging.error(f"Error in Bharath's plot: {e}", exc_info=True)
        raise


