# General


#  Setup
## Checking the connection
The pico board is already flashed. Make sure that you have inserted the antennas in the GP6 and GP27 ports. The received packets are directly printed to the serial output.
You can read them using picocom or a similar program. The baudrate used is 115200. The port needs to be replaced.

example: picocom -b 115200 /dev/tty.usbmodem2101

If the board is working you should see something outputs similar to the one below:

00:00:05.775 | 0f 00 00 00 13 5d 1f 43 1b 45 25 6f 21 fe 0f 7e | -69 CRC error

If you see this it means that the the tag, the carrier and the receiver are all working as they should be.


## Statistics Notebook
Create a python virtual environment and use the requirements.txt to install the dependencies. Tested using python3.13. The main libraries used are matplotlib, pandas, numpy and pyserial.

Create the virtual environment:

python3 -m venv .venv

Activate it. Check that the activation matches your shell. 
Example for fish shell:

source .venv/bin/activate.fish

pip3 install -r requirements.txt

You should now be read to use the statistics.ipynb


# The Pico Binary
Your board should be flashed. However, you can flash it using picotool if you have it installed.
picotool reboot -uf 
picotool load carrier_receiver_baseband.elf
picotool reboot





