import RPi.GPIO as GPIO
import time

# Motor driver pins
IN1 = 18#17
IN2 = 23#27
ENA = 24#22

# Encoder pins
encoderA = 13#5
encoderB = 19#6

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(encoderA, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(encoderB, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwm = GPIO.PWM(ENA, 1000)  # PWM on ENA at 1kHz
pwm.start(0)

# Variables for encoder
encoder_position = 0
pulses_per_revolution = 1000  # Replace with your encoder's value
target_pulses = pulses_per_revolution * 5  # 5 rotations

def encoder_callback(channel):
    global encoder_position
    if GPIO.input(encoderA) == GPIO.input(encoderB):
        encoder_position += 1
    else:
        encoder_position -= 1

# Setup encoder interrupt
GPIO.add_event_detect(encoderA, GPIO.BOTH, callback=encoder_callback)

def move_motor(direction, rotations):
    global encoder_position
    encoder_position = 0
    target_pulses = pulses_per_revolution * rotations
    
    # Set direction
    if direction == "forward":
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    
    # Start motor
    pwm.ChangeDutyCycle(2)  # Adjust speed as needed
    
    # Run motor until target pulses are reached
    while abs(encoder_position) < target_pulses:
        time.sleep(0.01)  # Adjust sleep time as necessary
    
    # Stop motor
    pwm.ChangeDutyCycle(0)

try:
    # Move 5 rotations forward
    move_motor("forward", 5)
    
    time.sleep(2)  # Wait 2 seconds before changing direction
    
    # Move 5 rotations backward
    move_motor("backward", 5)

except KeyboardInterrupt:
    pass

pwm.stop()
GPIO.cleanup()
