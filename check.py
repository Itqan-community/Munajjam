import traceback
import py_compile

try:
    py_compile.compile("munajjam/munajjam/transcription/whisperx.py", doraise=True)
except Exception:
    traceback.print_exc()
