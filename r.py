from Reduino.Displays import LCD
from Reduino.Utils import sleep
from Reduino import target

# Change COM3 to your serial port (for Linux/macOS use something like /dev/ttyUSB0)
target("COM6")

lcd = LCD(rs=12, en=11, d4=5, d5=4, d6=3, d7=2, cols=16, rows=2, backlight_pin=9)

lcd.message("REDUINO CORE", "boot sequence", top_align="center", bottom_align="center")
sleep(1200)

# Fake boot progress (written explicitly for maximum transpiler compatibility)
lcd.progress(1, 5, max_value=100, width=10, label="BOOT")
sleep(220)
lcd.progress(1, 15, max_value=100, width=10, label="BOOT")
sleep(220)
lcd.progress(1, 30, max_value=100, width=10, label="BOOT")
sleep(220)
lcd.progress(1, 45, max_value=100, width=10, label="BOOT")
sleep(220)
lcd.progress(1, 60, max_value=100, width=10, label="BOOT")
sleep(220)
lcd.progress(1, 75, max_value=100, width=10, label="BOOT")
sleep(220)
lcd.progress(1, 90, max_value=100, width=10, label="BOOT")
sleep(220)
lcd.progress(1, 100, max_value=100, width=10, label="BOOT")
sleep(220)

lcd.line(0, "SYSTEM ONLINE", align="center")
lcd.brightness(255)
sleep(800)
lcd.brightness(140)
sleep(300)
lcd.brightness(255)

# Non-blocking marquee vibe
lcd.animate("scroll", 1, ":: Welcome to Reduino LCD FX ::", speed_ms=120, loop=True)

while True:
    # Keep loop alive so animation keeps ticking
    sleep(80)