import os

f = open('down_files.txt', 'w')
for i in range(0, 127):
    f.write(
        f'https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-{i:05d}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet\n'
    )