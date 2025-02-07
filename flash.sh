picotool reboot -uf;
sleep 5;
picotool load carrier_receiver_baseband.elf;
picotool reboot;
sleep 5;
picocom -b 115200 /dev/tty.usbmodem2101
