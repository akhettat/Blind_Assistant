import pyttsx3

# Initialisation de la synthèse vocale
engine = pyttsx3.init()

# Tester la synthèse vocale simple
engine.say("Test de la synthèse vocale. Si vous entendez cette voix, pyttsx3 fonctionne.")
engine.runAndWait()
