#!/Users/wenya589/.pyenv/shims/python
#
# Copyright 2023, 2023 Wenqing Yan <yanwenqingindependent@gmail.com>
#
# This file is part of the pico backscatter project
# Analyze the communication systme performance with the metrics (time, reliability and distance).

from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from functools import cache
import pandas as pd
# from numpy import NaN
from pylab import rcParams
rcParams["figure.figsize"] = 16, 4
import math
import datetime
import serial
import time


# read the log file
def readfile(filename):
    types = {
        "time_rx": str,
        "frame": str,
        "rssi": str,
    }
    df = pd.read_csv(
        StringIO(" ".join(l for l in open(filename))),
        skiprows=0,
        header=None,
        dtype=types,
        delimiter="|",
        on_bad_lines='warn',
        names = ["time_rx", "frame", "rssi"]
    )
    df.dropna(inplace=True)
    # covert to time data type
    df.time_rx = df.time_rx.str.rstrip().str.lstrip()
    df.time_rx = pd.to_datetime(df.time_rx, format='%H:%M:%S.%f')
    for i in range(len(df)):
        df.iloc[i,0] = df.iloc[i,0].strftime("%H:%M:%S.%f")
    # parse the payload to seq and payload
    df.frame = df.frame.str.rstrip().str.lstrip()
    df = df[df.frame.str.contains("packet overflow") == False]
    df['seq'] = df.frame.apply(lambda x: int(x[3:5], base=16))
    df['payload'] = df.frame.apply(lambda x: x[6:])
    # parse the rssi data
    df.rssi = df.rssi.str.lstrip().str.split(" ", expand=True).iloc[:,0]
    df.rssi = df.rssi.astype('int')
    df = df.drop(columns=['frame'])
    df.reset_index(inplace=True)
    return df

# parse the hex payload, return a list with int numbers for each byte
def parse_payload(payload_string):
    tmp = map(lambda x: int(x, base=16), payload_string.split())
    return list(tmp)


def popcount(n):
    return bin(n).count("1")

# compare the received frame and transmitted frame and compute the number of bit errors
def compute_bit_errors(payload, sequence, PACKET_LEN=32):
    return sum(
        map(
            popcount,
            (
                np.array(payload[:PACKET_LEN])
                ^ np.array(sequence[: len(payload[:PACKET_LEN])])
            ),
        )
    )

# a 8-bit random number generator with uniform distribution
def rnd(seed):
    A1 = 1664525
    C1 = 1013904223
    RAND_MAX1 = 0xFFFFFFFF
    seed = ((seed * A1 + C1) & RAND_MAX1)
    return seed

# a 16-bit generator returns compressible 16-bit data sample
def data(seed):
    two_pi = np.float64(2.0 * np.float64(math.pi))
    u1 = 0
    u2 = 0
    while(u1 == 0 or u2 == 0):
        seed = rnd(seed)
        u1 = np.float64(seed/0xFFFFFFFF)
        seed = rnd(seed)
        u2 = np.float64(seed/0xFFFFFFFF)
    tmp = 0x7FF * np.float64(math.sqrt(np.float64(-2.0 * np.float64(math.log(u1)))))
    return np.trunc(max([0,min([0x3FFFFF,np.float64(np.float64(tmp * np.float64(math.cos(np.float64(two_pi * u2)))) + 0x1FFF)])])), seed

# generate the transmitted file for comparison
TOTAL_NUM_16RND = 512*40 # generate a 40MB file, in case transmit too many data (larger than required 2MB)
def generate_data(NUM_16RND, TOTAL_NUM_16RND):
    LOW_BYTE = (1 << 8) - 1
    length = int(np.ceil(TOTAL_NUM_16RND/NUM_16RND))
    index = [NUM_16RND*i*2 for i in range(length)]
    df = pd.DataFrame(index=index, columns=['data'])
    initial_seed = 0xabcd # initial seed
    pseudo_seq = 0 # (16-bit)
    seed  = initial_seed
    for i in index:
        payload_data = []
        for j in range(NUM_16RND):
            if pseudo_seq > 0xffff:
                pseudo_seq = 0
                seed = initial_seed
            pseudo_seq = pseudo_seq + 2
            number, seed = data(seed)
            payload_data.append((int(number) >> 8) - 0)
            payload_data.append(int(number) & LOW_BYTE)
        df.loc[i, "data"] = payload_data
    return df

file_content = None
def payload_for_peudo_seq(pseudo_seq,PACKET_LEN):
    global file_content
    if type(file_content) == type(None): # generate data
        file_content = generate_data(int(PACKET_LEN/2), TOTAL_NUM_16RND)
    if pseudo_seq in file_content.index:
        return file_content.loc[pseudo_seq, 'data']
    else:
        return file_content.loc[0, 'data'] # TODO: pseudo sequence not within the first expected range

def compute_ber_packet(df_row, PACKET_LEN=32):
    payload = parse_payload(df_row.payload)
    pseudoseq = int(((payload[0]<<8) - 0) + payload[1])
    expected_data = payload_for_peudo_seq(pseudoseq,PACKET_LEN)
    # compute the bit errors
    return (compute_bit_errors(payload[2:], expected_data, PACKET_LEN=PACKET_LEN), 8*(2+len(payload[2:]))) # 2+ for pseudo sequence

# main function to compute the BER for each frame, return both the error statistics dataframe and in total BER for the received data
def compute_ber(df, PACKET_LEN=32):
    # seq number initialization
    if len(df) > 0:
        errors,total = zip(*[compute_ber_packet(row,PACKET_LEN) for (_,row) in df.iterrows()])
        return sum(errors)/sum(total)
    else:
        print("Warning, the log-file seems empty.")
        return 0.5

# plot radar chart
def radar_plot(metrics):
    categories = ['Time', 'Reliability', 'Distance']
    system_ref = [62.321888, 0.201875*100, 39.956474923886844]
    system = [metrics[0], metrics[1], metrics[2]]

    label_loc = np.linspace(start=0.5 * np.pi, stop=11/6 * np.pi, num=len(categories))
    plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)

    # please keep the reference for your plot, we will update the reference after each SR session
    plt.plot(np.append(label_loc, (0.5 * np.pi)), system_ref+[system_ref[0]], label='Reference', color='grey')
    plt.fill(label_loc, system_ref, color='grey', alpha=0.25)

    plt.plot(np.append(label_loc, (0.5 * np.pi)), system+[system[0]], label='Our system', color='#77A136')
    plt.fill(label_loc, system, color='#77A136', alpha=0.25)
    plt.title('System Performance', size=20,pad=25)

    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories, fontsize=18)
    plt.legend(fontsize=18, loc='upper right',pad=20)
    plt.show()


def performance_time(df):
    # compute the file delay
    file_delay = df.time_rx[len(df) - 1] - df.time_rx[0]
    return file_delay
    

def performance_ber(df, packet_len):
    # compute the BER for all received packets
    # return the in total ber for received file, error statistics and correct file content supposed to be transmitted
    ber = compute_ber(df, PACKET_LEN=packet_len*2)
    return ber


def analysis_ber_packet(df, packet_len):
    # BER for each packet
    # payload only / without seq-number and pseudo-seq-number
    print("Note: if individual packets have a high bit-error rate, it could be that the pseudo-sequence number was corrupted and the script could not identify the expected payload correctly.")
    plt.scatter(range(len(df)), [compute_ber_packet(row,PACKET_LEN=packet_len*2)[0] for (_,row) in df.iterrows()], marker='o', s=6, color='black')
    plt.grid()
    plt.ylabel('Bit Error Rate [%]', fontsize=16)
    plt.xlabel('Seq. Number', fontsize=16)
    return [compute_ber_packet(row,PACKET_LEN=packet_len*2)[0] for (_,row) in df.iterrows()]


def performance_distance(distance_carrier_tag, distance_tag_rx):
    # record the distance [m]
    dis_metric = distance_carrier_tag**2*distance_tag_rx**2
    return dis_metric


def configure_board(port, div0, div1, baud, deviation, center, bandwidth):
    with serial.Serial(port, 115200) as ser:
        # configure tag (PIO)
        command = f"b {div0} {div1} {baud}\r"
        print(command)
        ser.write(command.encode('utf-8'))
        time.sleep(1)
        # configure receiver
        command = f"c {center} {deviation} {baud} {bandwidth}\r"
        print(command)
        ser.write(command.encode('utf-8'))
        time.sleep(1)
        # start receiver
        command = f"s\r"
        ser.write(command.encode('utf-8'))
        time.sleep(1)


def record_data(port, filename, time):
    with serial.Serial(port, 115200) as ser:
        with open(filename, "w") as f:
            t_end = datetime.datetime.now() + datetime.timedelta(seconds=time)
            while datetime.datetime.now() < t_end:
                rec = ser.readline().decode("utf-8")
                print(rec)
                f.write(rec)
            f.close()

