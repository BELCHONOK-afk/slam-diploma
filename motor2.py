import sys
import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)


class Motor:
	def __init__(self, pin1: int, pin2: int):
		self.ForwarPin = pin1
		self.BackwardPin = pin2
		GPIO.setup(self.ForwarPin, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.BackwardPin, GPIO.OUT, initial=GPIO.LOW)
	def forward(self):
		GPIO.output(self.ForwarPin, GPIO.HIGH)
		GPIO.output(self.BackwardPin, GPIO.LOW)
	def backward(self):
		GPIO.output(self.ForwarPin, GPIO.LOW)
		GPIO.output(self.BackwardPin, GPIO.HIGH)
	def stop(self):
		GPIO.output(self.ForwarPin, GPIO.LOW)
		GPIO.output(self.BackwardPin, GPIO.LOW)

class Robot:
	def __init__(self, motorLeft: Motor, motorRight: Motor) -> None:
		self.motorLeft = motorLeft
		self.motorRight = motorRight
	def forward(self):
		self.motorLeft.forward()
		self.motorRight.forward()
	def backward(self):
		self.motorLeft.backward()
		self.motorRight.backward()
	def right(self):
		self.motorLeft.forward()
		self.motorRight.backward()
	def left(self):
		self.motorLeft.backward()
		self.motorRight.forward()
	def stop(self):
		self.motorLeft.stop()
		self.motorRight.stop()

def main():
	R = Robot(Motor(5, 22), Motor(17, 27))
	command = ""
	while command != "q":
		command = input()
		if command == "f":
			print("forward")
			R.forward()

		if command == "b":
			R.backward()
			print("backward")

		if command == "r":
			R.right()
			print("right")

		if command == "l":
			R.left()
			print("left")

		if command == "s":
			R.stop()
			print("stop")
	
	R.stop()

if __name__=="__main__":
	try:
		main()
	except KeyboardInterrupt:
		pass
	finally:
		GPIO.cleanup()