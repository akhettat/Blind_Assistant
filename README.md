# BLINDY
This is an offline voice assistant project that aims to support people with visual disability to achieve several tasks.It will be also enhanced to be compatible with several platforms (Raspberry Pi, Android, OSX, ...etc)

## Key features:
1. Initial speech rate of 250
2. Multi-language support with Ctrl+L to cycle through voices
3. Dynamic speech rate control
4. Voice switching during speech
5. Display of available voices at startup

## To use the assistant:
1. Install dependencies:
```bash
pip install transformers pyttsx3 keyboard
```

2. Run the script

## Keyboard shortcuts:
- Escape: Stop current speech
- Ctrl+S: Speed up speech
- Ctrl+D: Slow down speech
- Ctrl+L: Change voice/language
- Ctrl+R: Repeat last response
- Ctrl+Q: Quit

**Note:** The available languages will depends on your system and the installed languages. you can install additional voices/languages then it will works with the assistant
