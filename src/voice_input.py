"""Optional speech-to-text. Uses Streamlit's st.audio_input (browser microphone) plus the optional
SpeechRecognition package, which sends the audio to Google's free web speech service (needs internet).
If anything is missing the caller falls back to typed input."""
import io


def recognizer_available():
    try:
        import speech_recognition  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe_wav(wav_bytes: bytes, language="en-IN"):
    """Return (text, error). Exactly one of them is None."""
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "Voice recognition package not installed. Run: pip install SpeechRecognition. You can type instead."
    try:
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language=language), None
    except sr.UnknownValueError:
        return None, "I could not understand the audio. Please try again or type your question."
    except sr.RequestError:
        return None, "The speech service could not be reached (internet needed). Please type your question."
    except Exception:
        return None, "Could not process the recording. Please type your question."
