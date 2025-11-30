import webrtcvad
import wave
import numpy as np
from pydub import AudioSegment
import librosa
import speech_recognition as sr
from typing import Dict, List, Tuple
import json

class ComprehensiveSpeechAnalyzer:
    """
    Complete speech analysis using WebRTC VAD with detailed metrics and transcription.
    Provides both percentage scores and exact values for all measurements.
    """
    
    def __init__(self, audio_file: str, aggressiveness: int = 2, frame_duration_ms: int = 30):
        """
        Initialize analyzer.
        
        Args:
            audio_file: Path to audio file
            aggressiveness: VAD sensitivity (0-3, where 3 is most aggressive)
            frame_duration_ms: Frame duration for VAD (10, 20, or 30 ms)
        """
        self.audio_file = audio_file
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = 16000
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Load and convert audio
        print("🎵 Loading and preprocessing audio...")
        self.audio = AudioSegment.from_file(audio_file)
        self.audio = self.audio.set_channels(1).set_frame_rate(self.sample_rate)

        # adding validation to audio | if showrt audio an exception is raised
        if len(self.audio) < 1000:
            raise ValueError("Audio too short for analysis (<1 second).")

        
        # Save as temporary WAV for processing
        self.temp_wav = "temp_analysis.wav"
        self.audio.export(self.temp_wav, format="wav")
        
        # Cache for results
        self._segments = None
        self._librosa_data = None
        
    def _get_librosa_data(self):
        """Load audio with librosa (cached)."""
        if self._librosa_data is None:
            y, sr = librosa.load(self.temp_wav, sr=self.sample_rate)
            self._librosa_data = (y, sr)
        return self._librosa_data
    
    def _read_frames(self):
        """Generator to read audio frames for VAD processing."""
        with wave.open(self.temp_wav, 'rb') as wf:
            sample_rate = wf.getframerate()
            bytes_per_sample = wf.getsampwidth()
            channels = wf.getnchannels()
            # frame_size = int(sample_rate * (self.frame_duration_ms / 1000.0) * 
            #                bytes_per_sample * channels)

            # suggested during logic verificatioin
            frame_bytes = int(self.sample_rate * (self.frame_duration_ms / 1000) * 2)

            
            while True:
                # frame = wf.readframes(frame_size // bytes_per_sample)

                # suggested during logic verification
                frame = wf.readframes(int(self.sample_rate * (self.frame_duration_ms / 1000)))

                if len(frame) < frame_bytes:
                    break
                yield frame
    
    def _compute_frame_energy(self, frame: bytes) -> float:
        """Compute RMS energy of frame."""
        audio_data = np.frombuffer(frame, dtype=np.int16)
        rms_energy = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        return rms_energy
    
    # def _compute_pitch_confidence(self, frame: bytes) -> Dict:
    #     """Compute autocorrelation-based pitch confidence."""
    #     audio_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        
    #     if np.max(np.abs(audio_data)) > 0:
    #         audio_data = audio_data / np.max(np.abs(audio_data))
    #     else:
    #         return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
    #     autocorr = np.correlate(audio_data, audio_data, mode='full')
    #     autocorr = autocorr[len(autocorr)//2:]
        
    #     if autocorr[0] > 0:
    #         autocorr = autocorr / autocorr[0]
    #     else:
    #         return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
    #     min_period = int(self.sample_rate / 500)
    #     max_period = int(self.sample_rate / 50)
        
    #     if max_period >= len(autocorr):
    #         max_period = len(autocorr) - 1
    #     if min_period >= max_period:
    #         return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
    #     search_range = autocorr[min_period:max_period]
    #     if len(search_range) == 0:
    #         return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
    #     peak_index = np.argmax(search_range) + min_period
    #     pitch_confidence = autocorr[peak_index]
    #     estimated_pitch = self.sample_rate / peak_index if peak_index > 0 else 0.0
    #     is_voiced = pitch_confidence > 0.35
        
    #     return {
    #         'pitch_confidence': float(pitch_confidence),
    #         'estimated_pitch_hz': float(estimated_pitch),
    #         'is_voiced': bool(is_voiced)
    #     }
    
    def _compute_pitch_confidence(self, frame: bytes) -> Dict:
        """Compute autocorrelation-based pitch confidence with pre-emphasis filtering."""
        audio_data = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        else:
            return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
        # 🔧 Pre-emphasis filter (boost high frequencies)
        # Formula: y[t] = x[t] - pre_emphasis * x[t-1]
        pre_emphasis = 0.97
        audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Compute autocorrelation
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize autocorrelation
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
        # Define search range for pitch frequency (50–500 Hz typical human speech)
        min_period = int(self.sample_rate / 500)
        max_period = int(self.sample_rate / 50)
        
        if max_period >= len(autocorr):
            max_period = len(autocorr) - 1
        if min_period >= max_period:
            return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
        # Find the most prominent peak in autocorrelation (ignoring lag=0)
        search_range = autocorr[min_period:max_period]
        if len(search_range) == 0:
            return {'pitch_confidence': 0.0, 'estimated_pitch_hz': 0.0, 'is_voiced': False}
        
        peak_index = np.argmax(search_range) + min_period
        pitch_confidence = autocorr[peak_index]
        estimated_pitch = self.sample_rate / peak_index if peak_index > 0 else 0.0
        is_voiced = pitch_confidence > 0.35  # heuristic threshold
        
        return {
            'pitch_confidence': float(pitch_confidence),
            'estimated_pitch_hz': float(estimated_pitch),
            'is_voiced': bool(is_voiced)
        }

    
    # def _calibrate_energy_thresholds(self, sample_size: int = 100) -> Tuple[float, float]:
    #     """Auto-calibrate energy thresholds based on audio characteristics."""
    #     energies = []
    #     frame_count = 0
        
    #     for frame in self._read_frames():
    #         energy = self._compute_frame_energy(frame)
    #         energies.append(energy)
    #         frame_count += 1
    #         if frame_count >= sample_size:
    #             break
        
    #     energies = np.array(energies)
    #     mean_energy = np.mean(energies)
    #     std_energy = np.std(energies)
        
    #     silence_threshold = mean_energy - 0.5 * std_energy
    #     low_volume_threshold = mean_energy + 0.3 * std_energy
        
    #     return silence_threshold, low_volume_threshold

    def _calibrate_energy_thresholds(self, sample_size: int = 100) -> Tuple[float, float]:
        """Auto-calibrate energy thresholds based on audio characteristics."""
        
        # --- Step 1: Read all frames once into a list ---
        all_frames = list(self._read_frames())
        total_frames = len(all_frames)
        
        # --- Step 2: Choose evenly spaced frame indices ---
        sample_indices = np.linspace(0, total_frames - 1, num=sample_size, dtype=int)
        
        # --- Step 3: Compute energy only for sampled frames ---
        energies = [self._compute_frame_energy(all_frames[i]) for i in sample_indices]
        energies = np.array(energies)
        
        # --- Step 4: Compute thresholds ---
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        silence_threshold = mean_energy - 0.5 * std_energy
        low_volume_threshold = mean_energy + 0.3 * std_energy
        
        return silence_threshold, low_volume_threshold

    """
    Added a new method _merge_short_silences that works with your segment dicts.

    detect_speech_segments now calls _merge_short_silences(min_silence_ms=300) before caching/returning segments.

    When merging, energies/pitch lists and durations are preserved (we concatenate arrays when extending a speech segment to include the short silence).
    """

    def _merge_short_silences(self, segments: List[Dict], min_silence_ms: int = 300) -> List[Dict]:
        """
        Merge short silence segments that occur between speech segments.
        Keeps energies/pitch lists and updates start/end/duration/avg values for merged segments.

        Args:
            segments: list of segment dicts (with keys 'start', 'end', 'type', 'energies', 'pitch_confidences', 'pitch_frequencies', ...)
            min_silence_ms: silence shorter than this (milliseconds) will be merged into surrounding speech
        Returns:
            new list of segments with short silences merged into adjacent speech segments.
        """
        if not segments:
            return segments

        min_silence_sec = min_silence_ms / 1000.0
        merged = [segments[0].copy()]

        for seg in segments[1:]:
            last = merged[-1]

            # Determine whether segments are speech or silence
            last_is_speech = last["type"] in ("NORMAL_SPEECH", "LOW_VOLUME_SPEECH")
            cur_is_speech = seg["type"] in ("NORMAL_SPEECH", "LOW_VOLUME_SPEECH")
            cur_duration = seg.get("duration", seg["end"] - seg["start"])

            # If current is a short silence and previous is speech, merge current into previous
            if (not cur_is_speech) and last_is_speech and (cur_duration < min_silence_sec):
                # Extend last's end and duration
                last["end"] = seg["end"]
                last["duration"] = last.get("duration", last["end"] - last["start"]) + cur_duration

                # Merge energies and pitch lists (if present)
                if "energies" in seg:
                    last["energies"].extend(seg.get("energies", []))
                if "pitch_confidences" in seg:
                    last["pitch_confidences"].extend(seg.get("pitch_confidences", []))
                if "pitch_frequencies" in seg:
                    last["pitch_frequencies"].extend(seg.get("pitch_frequencies", []))

                # Update averages (lazy — we'll recompute later when finalizing segments)
                merged[-1] = last
            else:
                # Otherwise keep as a separate segment (important to copy to avoid aliasing)
                merged.append(seg.copy())

        # Recompute aggregated stats for merged segments (avg_energy, avg_pitch_confidence, avg_pitch_hz)
        for s in merged:
            if "energies" in s and len(s["energies"]) > 0:
                s["avg_energy"] = float(np.mean(s["energies"]))
            else:
                s["avg_energy"] = 0.0

            if "pitch_confidences" in s and len(s["pitch_confidences"]) > 0:
                s["avg_pitch_confidence"] = float(np.mean(s["pitch_confidences"]))
            else:
                s["avg_pitch_confidence"] = 0.0

            valid_pitches = [p for p in s.get("pitch_frequencies", []) if p > 0]
            s["avg_pitch_hz"] = float(np.mean(valid_pitches)) if valid_pitches else 0.0

            # Ensure duration/end/start consistent
            s["duration"] = s.get("duration", s["end"] - s["start"])

        return merged



    def detect_speech_segments(self, verbose: bool = False) -> List[Dict]:
        """
        Detect speech segments using WebRTC VAD with energy and pitch analysis.
        
        Returns:
            List of segments with type, timing, energy, and pitch information
        """
        if self._segments is not None:
            return self._segments
        
        print("\n🎤 Detecting speech segments with WebRTC VAD...")
        
        silence_threshold, low_volume_threshold = self._calibrate_energy_thresholds()
        
        frame_time = self.frame_duration_ms / 1000.0
        segments = []
        current_segment = None
        frame_index = 0
        
        for frame in self._read_frames():
            timestamp = frame_index * frame_time
            
            # VAD detection
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            # Energy analysis
            energy = self._compute_frame_energy(frame)
            
            # Pitch analysis
            pitch_info = self._compute_pitch_confidence(frame)
            
            # Three-way classification
            if not is_speech and energy < silence_threshold and not pitch_info['is_voiced']:
                classification = "SILENCE"
            elif is_speech and pitch_info['is_voiced'] and energy < low_volume_threshold:
                classification = "LOW_VOLUME_SPEECH"
            elif (is_speech and pitch_info['is_voiced']) or \
                 (pitch_info['is_voiced'] and energy > silence_threshold):
                classification = "NORMAL_SPEECH"
            else:
                classification = "SILENCE"
            
            # Segment tracking
            if current_segment is None or current_segment["type"] != classification:
                if current_segment is not None:
                    current_segment["end"] = timestamp
                    current_segment["duration"] = current_segment["end"] - current_segment["start"]
                    current_segment["avg_energy"] = np.mean(current_segment["energies"])
                    if current_segment["pitch_confidences"]:
                        current_segment["avg_pitch_confidence"] = np.mean(current_segment["pitch_confidences"])
                        valid_pitches = [p for p in current_segment["pitch_frequencies"] if p > 0]
                        current_segment["avg_pitch_hz"] = np.mean(valid_pitches) if valid_pitches else 0.0
                    segments.append(current_segment)
                
                current_segment = {
                    "start": timestamp,
                    "type": classification,
                    "energies": [energy],
                    "pitch_confidences": [pitch_info['pitch_confidence']],
                    "pitch_frequencies": [pitch_info['estimated_pitch_hz']]
                }
            else:
                current_segment["energies"].append(energy)
                current_segment["pitch_confidences"].append(pitch_info['pitch_confidence'])
                current_segment["pitch_frequencies"].append(pitch_info['estimated_pitch_hz'])
            
            if verbose and frame_index % 50 == 0:
                print(f"[{timestamp:6.2f}s] {classification:20s} | Energy: {energy:7.2f} | "
                      f"Pitch: {pitch_info['estimated_pitch_hz']:6.1f}Hz")
            
            frame_index += 1
        

        # Close last segment
        if current_segment is not None:
            current_segment["end"] = frame_index * frame_time
            current_segment["duration"] = current_segment["end"] - current_segment["start"]
            current_segment["avg_energy"] = np.mean(current_segment["energies"])
            if current_segment["pitch_confidences"]:
                current_segment["avg_pitch_confidence"] = np.mean(current_segment["pitch_confidences"])
                valid_pitches = [p for p in current_segment["pitch_frequencies"] if p > 0]
                current_segment["avg_pitch_hz"] = np.mean(valid_pitches) if valid_pitches else 0.0
            segments.append(current_segment)

    #suggested        

        # ---- NEW: merge short silences to avoid choppy detection ----
        segments = self._merge_short_silences(segments, min_silence_ms=300)
        # ------------------------------------------------------------
        
        self._segments = segments
        return segments
    
    def calculate_silence_metrics(self) -> Dict:
        """Calculate detailed silence metrics."""
        segments = self.detect_speech_segments()
        
        normal_speech_time = sum(s["duration"] for s in segments if s["type"] == "NORMAL_SPEECH")
        low_volume_time = sum(s["duration"] for s in segments if s["type"] == "LOW_VOLUME_SPEECH")
        silence_time = sum(s["duration"] for s in segments if s["type"] == "SILENCE")
        
        total_speech_time = normal_speech_time + low_volume_time
        total_time = total_speech_time + silence_time
        
        normal_speech_count = sum(1 for s in segments if s["type"] == "NORMAL_SPEECH")
        low_volume_count = sum(1 for s in segments if s["type"] == "LOW_VOLUME_SPEECH")
        silence_count = sum(1 for s in segments if s["type"] == "SILENCE")
        
        return {
            'silence_ratio_percent': (silence_time / total_time * 100) if total_time > 0 else 0,
            'speech_ratio_percent': (total_speech_time / total_time * 100) if total_time > 0 else 0,
            'silence_time_seconds': silence_time,
            'normal_speech_time_seconds': normal_speech_time,
            'low_volume_speech_time_seconds': low_volume_time,
            'total_speech_time_seconds': total_speech_time,
            'total_duration_seconds': total_time,
            'silence_segment_count': silence_count,
            'normal_speech_segment_count': normal_speech_count,
            'low_volume_segment_count': low_volume_count
        }
    
    def calculate_speech_rate(self, transcript: str = None) -> Dict:
        """Calculate speech rate with or without transcript."""
        silence_metrics = self.calculate_silence_metrics()
        total_speech_time = silence_metrics['total_speech_time_seconds']
        
        y, sr = self._get_librosa_data()
        
        # Syllable-based estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # syllables = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))
        # estimated_words = syllables / 1.5
        
        #for above two lines - improvement suggestioin
        syllables = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True))
        estimated_words = syllables / 1.7  # slightly lower ratio for realism


        if total_speech_time > 0:
            wpm_estimated = (estimated_words / total_speech_time) * 60
            wps_estimated = estimated_words / total_speech_time
        else:
            wpm_estimated = 0
            wps_estimated = 0
        
        result = {
            'estimated_syllables': int(syllables),
            'estimated_words': estimated_words,
            'words_per_minute_estimated': wpm_estimated,
            'words_per_second_estimated': wps_estimated,
            'speaking_time_seconds': total_speech_time
        }
        
        # If transcript provided, calculate actual rate
        if transcript:
            actual_words = len(transcript.strip().split())
            if total_speech_time > 0:
                result['actual_word_count'] = actual_words
                result['words_per_minute_actual'] = (actual_words / total_speech_time) * 60
                result['words_per_second_actual'] = actual_words / total_speech_time
        
        return result
    
    def calculate_pitch_variation(self) -> Dict:
        """Calculate detailed pitch variation metrics."""
        y, sr = self._get_librosa_data()
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return {
                'mean_pitch_hz': 0, 'median_pitch_hz': 0, 'pitch_std_hz': 0,
                'pitch_range_hz': 0, 'min_pitch_hz': 0, 'max_pitch_hz': 0,
                'pitch_variation_percent': 0, 'voiced_frame_count': 0
            }
        
        mean_pitch = np.mean(f0_clean)
        variation_percent = (np.std(f0_clean) / mean_pitch * 100) if mean_pitch > 0 else 0
        
        return {
            'mean_pitch_hz': float(mean_pitch),
            'median_pitch_hz': float(np.median(f0_clean)),
            'pitch_std_hz': float(np.std(f0_clean)),
            'pitch_range_hz': float(np.max(f0_clean) - np.min(f0_clean)),
            'min_pitch_hz': float(np.min(f0_clean)),
            'max_pitch_hz': float(np.max(f0_clean)),
            'pitch_variation_percent': float(variation_percent),
            'voiced_frame_count': int(len(f0_clean))
        }
    
    def calculate_clarity(self) -> Dict:
        """Calculate speech clarity metrics."""
        y, sr = self._get_librosa_data()
        
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        avg_flatness = np.mean(spectral_flatness)
        avg_zcr = np.mean(zcr)
        avg_centroid = np.mean(spectral_centroid)
        
        # clarity_score = (1 - avg_flatness) * 100
        # below is the suggested improvement for the above line
        clarity_score = max(0, min(100, (1 - avg_flatness) * 100))

        
        return {
            'spectral_flatness': float(avg_flatness),
            'zero_crossing_rate': float(avg_zcr),
            'spectral_centroid_hz': float(avg_centroid),
            'clarity_score_percent': float(clarity_score)
        }
    
    def calculate_pacing(self) -> Dict:
        """Calculate pacing and consistency metrics."""
        segments = self.detect_speech_segments()
        
        speech_segments = [s for s in segments if s["type"] in ["NORMAL_SPEECH", "LOW_VOLUME_SPEECH"]]
        
        if not speech_segments:
            return {
                'avg_speech_segment_duration_seconds': 0, 'speech_duration_std_seconds': 0,
                'pacing_consistency_percent': 0, 'total_speech_segments': 0,
                'min_segment_duration_seconds': 0, 'max_segment_duration_seconds': 0
            }
        
        durations = [s["duration"] for s in speech_segments]
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        consistency = 100 - min(100, (std_duration / avg_duration) * 100) if avg_duration > 0 else 0
        
        return {
            'avg_speech_segment_duration_seconds': float(avg_duration),
            'speech_duration_std_seconds': float(std_duration),
            'pacing_consistency_percent': float(consistency),
            'total_speech_segments': len(speech_segments),
            'min_segment_duration_seconds': float(min(durations)),
            'max_segment_duration_seconds': float(max(durations))
        }
    
    def calculate_expressiveness(self) -> Dict:
        """Calculate expressiveness metrics."""
        y, sr = self._get_librosa_data()
        
        rms = librosa.feature.rms(y=y)
        energy_std = np.std(rms)
        energy_mean = np.mean(rms)
        
        rms_db = librosa.amplitude_to_db(rms)
        dynamic_range = np.max(rms_db) - np.min(rms_db)
        
        pitch_metrics = self.calculate_pitch_variation()
        
        energy_variation = (energy_std / energy_mean * 100) if energy_mean > 0 else 0
        energy_variation_capped = min(100, energy_variation)
        
        expressiveness_score = (energy_variation_capped * 0.5 + 
                               pitch_metrics['pitch_variation_percent'] * 0.5)
        
        return {
            'energy_variation_rms': float(energy_std),
            'mean_energy_rms': float(energy_mean),
            'dynamic_range_db': float(dynamic_range),
            'energy_variation_percent': float(energy_variation),
            'expressiveness_score_percent': float(expressiveness_score)
        }
    
    # def calculate_confidence(self) -> Dict:
    #     """Calculate confidence score from multiple factors."""
    #     silence_metrics = self.calculate_silence_metrics()
    #     pitch_metrics = self.calculate_pitch_variation()
    #     pacing_metrics = self.calculate_pacing()
        
    #     y, sr = self._get_librosa_data()
    #     rms = librosa.feature.rms(y=y)
    #     avg_energy = np.mean(rms)
        
    #     silence_score = silence_metrics['speech_ratio_percent']
    #     pitch_stability = 100 - min(100, pitch_metrics['pitch_variation_percent'])
    #     # energy_score = min(100, avg_energy * 1000)
    #     #suggestion for the above line
    #     energy_score = min(100, max(0, avg_energy * 1000))


    #     pacing_score = pacing_metrics['pacing_consistency_percent']
        
    #     confidence_score = (
    #         silence_score * 0.3 +
    #         pitch_stability * 0.2 +
    #         energy_score * 0.3 +
    #         pacing_score * 0.2
    #     )
        
    #     return {
    #         'confidence_score_percent': float(confidence_score),
    #         'silence_contribution': float(silence_score),
    #         'pitch_stability_contribution': float(pitch_stability),
    #         'energy_contribution': float(energy_score),
    #         'pacing_contribution': float(pacing_score)
    #     }
    
    def calculate_confidence(self) -> Dict:
        """Calculate confidence score from multiple weighted factors."""
        silence_metrics = self.calculate_silence_metrics()
        pitch_metrics = self.calculate_pitch_variation()
        pacing_metrics = self.calculate_pacing()

        y, sr = self._get_librosa_data()
        rms = librosa.feature.rms(y=y)
        avg_energy = np.mean(rms)

        # Normalize and scale energy more effectively (perceptual correction)
        # Convert to dB scale and normalize between 0–100
        energy_db = librosa.amplitude_to_db([avg_energy], ref=1.0)[0]
        energy_score = np.clip((energy_db + 60) / 0.6, 0, 100)  # maps -60–0 dB to 0–100

        # Silence: prefer more speech (less silence)
        silence_score = silence_metrics['speech_ratio_percent']

        # Pitch stability: less variation = more confidence
        pitch_stability = max(0, 100 - min(100, pitch_metrics['pitch_variation_percent']))

        # Pacing consistency (avoid 0)
        pacing_score = max(20, pacing_metrics['pacing_consistency_percent'])

        # Weighted confidence blend
        confidence_score = (
            silence_score * 0.25 +
            pitch_stability * 0.2 +
            energy_score * 0.3 +
            pacing_score * 0.25
        )

        return {
            'confidence_score_percent': float(confidence_score),
            'silence_contribution': float(silence_score),
            'pitch_stability_contribution': float(pitch_stability),
            'energy_contribution': float(energy_score),
            'pacing_contribution': float(pacing_score)
        }


    def transcribe_audio(self) -> Dict:
        """Extract text from audio using speech recognition."""
        print("\n🎙️ Transcribing audio...")
        
        recognizer = sr.Recognizer()
        
        try:
            with sr.AudioFile(self.temp_wav) as source:
                audio_data = recognizer.record(source)
                
                # Try Google Speech Recognition
                try:
                    transcript = recognizer.recognize_google(audio_data)
                    confidence = "high"
                except sr.UnknownValueError:
                    transcript = "[Could not understand audio]"
                    confidence = "none"
                except sr.RequestError as e:
                    transcript = f"[API error: {e}]"
                    confidence = "error"
                
                word_count = len(transcript.split()) if transcript and "[" not in transcript else 0
                
                return {
                    'transcript': transcript,
                    'word_count': word_count,
                    'confidence': confidence,
                    'method': 'Google Speech Recognition'
                }
        
        except Exception as e:
            return {
                'transcript': f"[Transcription failed: {e}]",
                'word_count': 0,
                'confidence': 'error',
                'method': 'Google Speech Recognition'
            }
    
    def analyze_all(self, include_transcription: bool = True) -> Dict:
        """Perform complete analysis."""
        print(f"\n{'='*80}")
        print(f"🎵 COMPREHENSIVE SPEECH ANALYSIS")
        print(f"{'='*80}")
        print(f"Audio File: {self.audio_file}")
        print(f"VAD Aggressiveness: {self.aggressiveness}")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Frame Duration: {self.frame_duration_ms} ms")
        
        results = {
            'silence_metrics': self.calculate_silence_metrics(),
            'speech_rate': self.calculate_speech_rate(),
            'pitch_variation': self.calculate_pitch_variation(),
            'clarity': self.calculate_clarity(),
            'pacing': self.calculate_pacing(),
            'expressiveness': self.calculate_expressiveness(),
            'confidence': self.calculate_confidence()
        }
         
        # Add transcription last
        if include_transcription:
            transcription = self.transcribe_audio()
            results['transcription'] = transcription
            
            # Recalculate speech rate with actual transcript
            if transcription['word_count'] > 0:
                results['speech_rate'] = self.calculate_speech_rate(transcription['transcript'])
        
        return results
    
    def print_report(self, include_transcription: bool = True):
        """Print formatted analysis report."""
        results = self.analyze_all(include_transcription)
        
        print(f"\n{'='*80}")
        print("📊 DETAILED ANALYSIS REPORT")
        print(f"{'='*80}")
        
        # Silence Metrics
        sm = results['silence_metrics']
        print(f"\n🔇 SILENCE ANALYSIS:")
        print(f"   Silence Ratio: {sm['silence_ratio_percent']:.1f}% ({sm['silence_time_seconds']:.2f}s)")
        print(f"   Speech Ratio: {sm['speech_ratio_percent']:.1f}% ({sm['total_speech_time_seconds']:.2f}s)")
        print(f"   Normal Speech: {sm['normal_speech_time_seconds']:.2f}s ({sm['normal_speech_segment_count']} segments)")
        print(f"   Low Volume Speech: {sm['low_volume_speech_time_seconds']:.2f}s ({sm['low_volume_segment_count']} segments)")
        print(f"   Total Duration: {sm['total_duration_seconds']:.2f}s")
        
        # Speech Rate
        sr_data = results['speech_rate']
        print(f"\n🗣️  SPEECH RATE:")
        if 'actual_word_count' in sr_data:
            print(f"   Words: {sr_data['actual_word_count']}")
            print(f"   Rate: {sr_data['words_per_minute_actual']:.1f} WPM ({sr_data['words_per_second_actual']:.2f} WPS)")
        else:
            print(f"   Estimated Words: {sr_data['estimated_words']:.0f} (from {sr_data['estimated_syllables']} syllables)")
            print(f"   Estimated Rate: {sr_data['words_per_minute_estimated']:.1f} WPM")
        print(f"   Speaking Time: {sr_data['speaking_time_seconds']:.2f}s")
        
        # Pitch Variation
        pv = results['pitch_variation']
        print(f"\n🎵 PITCH VARIATION:")
        print(f"   Mean: {pv['mean_pitch_hz']:.1f} Hz | Median: {pv['median_pitch_hz']:.1f} Hz")
        print(f"   Range: {pv['min_pitch_hz']:.1f} - {pv['max_pitch_hz']:.1f} Hz ({pv['pitch_range_hz']:.1f} Hz)")
        print(f"   Std Dev: {pv['pitch_std_hz']:.1f} Hz")
        print(f"   Variation: {pv['pitch_variation_percent']:.1f}%")
        
        # Clarity
        cl = results['clarity']
        print(f"\n🔊 CLARITY:")
        print(f"   Clarity Score: {cl['clarity_score_percent']:.1f}%")
        print(f"   Spectral Flatness: {cl['spectral_flatness']:.4f}")
        print(f"   Zero Crossing Rate: {cl['zero_crossing_rate']:.4f}")
        print(f"   Spectral Centroid: {cl['spectral_centroid_hz']:.1f} Hz")
        
        # Pacing
        pc = results['pacing']
        print(f"\n⏱️  PACING:")
        print(f"   Consistency: {pc['pacing_consistency_percent']:.1f}%")
        print(f"   Avg Segment Duration: {pc['avg_speech_segment_duration_seconds']:.2f}s")
        print(f"   Duration Std Dev: {pc['speech_duration_std_seconds']:.2f}s")
        print(f"   Range: {pc['min_segment_duration_seconds']:.2f}s - {pc['max_segment_duration_seconds']:.2f}s")
        print(f"   Total Segments: {pc['total_speech_segments']}")
        
        # Expressiveness
        ex = results['expressiveness']
        print(f"\n✨ EXPRESSIVENESS:")
        print(f"   Expressiveness Score: {ex['expressiveness_score_percent']:.1f}%")
        print(f"   Dynamic Range: {ex['dynamic_range_db']:.1f} dB")
        print(f"   Energy Variation: {ex['energy_variation_percent']:.1f}%")
        print(f"   Mean Energy: {ex['mean_energy_rms']:.4f}")
        
        # Confidence
        cf = results['confidence']
        print(f"\n💪 CONFIDENCE:")
        print(f"   Overall Score: {cf['confidence_score_percent']:.1f}%")
        print(f"   Contributing Factors:")
        print(f"      • Silence: {cf['silence_contribution']:.1f}%")
        print(f"      • Pitch Stability: {cf['pitch_stability_contribution']:.1f}%")
        print(f"      • Energy: {cf['energy_contribution']:.1f}%")
        print(f"      • Pacing: {cf['pacing_contribution']:.1f}%")
        
        # Transcription
        if include_transcription and 'transcription' in results:
            tr = results['transcription']
            print(f"\n📝 TRANSCRIPTION:")
            print(f"   Method: {tr['method']}")
            print(f"   Confidence: {tr['confidence']}")
            print(f"   Word Count: {tr['word_count']}")
            print(f"   Text: \"{tr['transcript']}\"")
        
        print(f"\n{'='*80}\n")
        
        return results


    def export_analysis_json(self, include_transcription: bool = True, save_path: str = None) -> str:
        """
        Perform full speech analysis and return results as a JSON string.
        Optionally saves to a JSON file if save_path is provided.
        """
        # Perform full analysis
        results = self.analyze_all(include_transcription)

        # Add metadata for better traceability
        export_data = {
            "metadata": {
                "audio_file": self.audio_file,
                "sample_rate": self.sample_rate,
                "frame_duration_ms": self.frame_duration_ms,
                "vad_aggressiveness": self.aggressiveness
            },
            "results": results
        }

        # Convert to JSON string (pretty formatted)
        json_output = json.dumps(export_data, indent=4, ensure_ascii=False)

        # Optionally save to file
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"✅ Analysis JSON saved to: {save_path}")

        return json_output



# ============================================================
# 🧪 EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    # Replace with your audio file
    AUDIO_FILE = "sp-anls.wav"
    
    # Create analyzer
    analyzer = ComprehensiveSpeechAnalyzer(
        audio_file=AUDIO_FILE,
        aggressiveness=3,  # 0-3, higher = more aggressive silence detection
        frame_duration_ms=20  # 10, 20, or 30 ms
    )
    
    # Run complete analysis with transcription
    # results = analyzer.print_report(include_transcription=True)
    json_data = analyzer.export_analysis_json(include_transcription=True, save_path="analysis_report.json")

    print(json_data)
    # Access specific metrics
    # print("Quick Access Examples:")
    # print(f"Silence Ratio: {results['silence_metrics']['silence_ratio_percent']:.1f}%")
    # print(f"Speech Rate: {results['speech_rate'].get('words_per_minute_actual', 
    #       results['speech_rate']['words_per_minute_estimated']):.1f} WPM")
    # print(f"Confidence: {results['confidence']['confidence_score_percent']:.1f}%")
    
    # if 'transcription' in results:
    #     print(f"Transcript: {results['transcription']['transcript']}")