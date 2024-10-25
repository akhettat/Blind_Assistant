import keyboard
import time

def test_keyboard():
    print("Le raccourci clavier a été activé!")

# Associer le raccourci clavier à la fonction de test
keyboard.add_hotkey('ctrl+shift+a', test_keyboard)

print("Appuyez sur Ctrl+Shift+A pour tester le raccourci clavier.")

try:
    while True:
        time.sleep(0.1)  # Petite pause pour éviter une utilisation excessive du CPU
except KeyboardInterrupt:
    print("\nProgramme interrompu.")
