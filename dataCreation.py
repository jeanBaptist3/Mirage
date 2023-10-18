import csv
from Crypto.Cipher import ChaCha20 as Cha
from Crypto.Random import get_random_bytes
from base64 import b64encode

#this is the plaintext, 256*blocks bit only 0, so the encrytion will yield only the stream from ChaCha

def binary_to_hex(binary_str):
    # Check if the length of the binary string is a multiple of 16
    if len(binary_str) % 16 != 0:
        raise ValueError("Binary string length is not a multiple of 16")

    hex_str = ""
    # Process the binary string in 16-bit chunks and convert to hexadecimal
    for i in range(0, len(binary_str), 16):
        # Extract a 16-bit chunk from the binary string
        chunk = binary_str[i:i+16]
        # Convert the chunk to an integer
        chunk_int = int(chunk, 2)
        # Convert the integer to a hexadecimal string and remove the '0x' prefix
        hex_chunk = hex(chunk_int)[2:]
        # Make sure the hexadecimal chunk is 4 characters long by adding leading zeros if needed
        hex_chunk = hex_chunk.zfill(4)
        hex_str += hex_chunk

    return hex_str

def create_data(generated_blocks, prediction_blocks, bytes_per_token, train_size,val_size,test_size) :
    prefix = "predict the next 512 bit "
    plaintext = b''
    full_size = val_size+ test_size+ train_size


    gen_blocks = generated_blocks
    pred_blocks = prediction_blocks
    blocks = gen_blocks + pred_blocks
    for i in range (0,blocks*64) :
        plaintext = plaintext + b'\x00'


    """
    These are the arrays consisting of the keys to train and their ciphertexts and nonces
    """
    keys = []
    ciphers = []
    ciphetexts = []
    nonces = []
    data = []
    plaintexts = []
    number_of_keys = 1
    counter = 0
    cipher = Cha.new(key = get_random_bytes(32))
    for i in range(0, full_size) :
        if (i % (full_size / number_of_keys) == 0):
            keys.append(get_random_bytes(32))
            counter = counter + 1
        ciphers.append(Cha.new(key=keys[counter - 1]))
        ciphers[i].seek(0)
        ciphetext = ciphers[i].encrypt(plaintext)
        ciphetexts.append(ciphetext)
        nonce = ciphers[i].nonce
        ciphetext_bits = ''.join(format(byte, '08b') for byte in ciphetext)
        nonce_bits = ''.join(format(byte, '08b') for byte in nonce)
        nonces.append(nonce)
        pretokenized_input = binary_to_hex(ciphetext_bits[:512*gen_blocks])
        pretokenized_target = binary_to_hex(ciphetext_bits[512*gen_blocks:])
        nonce_bits = binary_to_hex(nonce_bits)


        for j in range(1, gen_blocks * 64 //bytes_per_token):
            pretokenized_input = pretokenized_input[:gen_blocks*128 - j * 2*bytes_per_token]+ "," + pretokenized_input[gen_blocks*128 - j *2*bytes_per_token:]

        for j in range(1, prediction_blocks*64 //bytes_per_token):
            pretokenized_target = pretokenized_target[:prediction_blocks*128-j * 2 * bytes_per_token]+ "," + pretokenized_target[prediction_blocks*128-j * 2 * bytes_per_token:]

        for y in range(0,8//bytes_per_token) :
            nonce_bits = nonce_bits[:16-(y+1)*2*bytes_per_token] + ',' + nonce_bits[16-(y+1)*2*bytes_per_token:]

        data.append((pretokenized_input + nonce_bits,  pretokenized_target))

    """
    This is for testing the correct encoding and decoding 
    
    ciphersnew = []
    plaintextTests = []
    for i in range(0, testsize) :
        nonce = b64decode(nonces[i])
        ciphertext = b64decode(ciphetexts[i])
        ciphersnew.append(Cha.new(key=keys[i], nonce=nonce))
        plaintextTest = ciphersnew[i].decrypt(ciphertext)
        plaintextTests.append(plaintextTest)
        print(plaintextTest
    
    cipherTwo =b64encode(ciphers[0].encrypt(plaintextTwo)).decode('utf-8')
    print(cipherTwo)
    
    Neural Net consisting of 570 Input Nodes and 2 hidden layer with 1024 Nodes and 512 output layer => 2 Million Parameters
    """
    header = ['block_and_nonce','next_block']

    print(b64encode(plaintext).decode('utf-8') + "test")
    with open(r'data/data_dump.csv', 'w', encoding='UTF8',newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for text in data :
            writer.writerow(text)


