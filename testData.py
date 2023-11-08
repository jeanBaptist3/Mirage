import secrets as sc
import csv
def binary_to_hex(binary_str):
    # Check if the length of the binary string is a multiple of 16
    if len(binary_str) % 8 != 0:
        raise ValueError("Binary string length is not a multiple of 16")

    hex_str = ""
    # Process the binary string in 16-bit chunks and convert to hexadecimal
    for i in range(0, len(binary_str), 8):
        # Extract a 16-bit chunk from the binary string
        chunk = binary_str[i:i+8]
        # Convert the chunk to an integer

        chunk_int = int(chunk, 2)
        # Convert the integer to a hexadecimal string and remove the '0x' prefix
        hex_chunk = hex(chunk_int)[2:]
        # Make sure the hexadecimal chunk is 4 characters long by adding leading zeros if needed
        hex_chunk = hex_chunk.zfill(2)
        hex_str += hex_chunk

    return hex_str

def xoRand(full_size,path,start_token,end_token):
    data = []
    for i in range(full_size):
        binary_source_one = sc.randbits(8)
        binary_source_two = sc.randbits(8)
        source_one = binary_to_hex(format(binary_source_one,'08b'))
        source_two = binary_to_hex(format(binary_source_two, '08b'))
        print("source one =" + str(format(binary_source_one,'08b')))
        print(source_one)
        print("source two = " + str(format(binary_source_two, '08b')))
        print(source_two)
        print(format(binary_source_one^binary_source_two,'08b'))
        """if format(binary_source_two, '08b')[-1] == '1' :
            data.append((start_token + "," + source_one + "," + source_two, "ff", "0"))
        if format(binary_source_two, '08b')[-1] == '0' :
            data.append((start_token + "," + source_one + "," + source_two, "00", "0"))"""
        data.append((start_token + "," + source_one + "," + source_two, binary_to_hex(format(binary_source_one^binary_source_two,'08b')), "0"))


    header = ['source_sequence', 'target_sequence', 'decoder_sequence']

    with open(f'{path}', 'w', encoding='UTF8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for text in data:
            writer.writerow(text)
