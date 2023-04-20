# Image-Compression

This project implements two different algorithms for image compression: JPEG and LZW. JPEG is a lossy compression in which some of the data from the original image is lost, while LZW is lossless, meaning that the compressed image can be reconstructed exactly to the original without any loss of information.

## JPEG Compression

JPEG (Joint Photographic Experts Group) is a popular image compression standard that achieves high compression ratios by applying a series of mathematical transformations on the image data. The JPEG encoder in this project takes a color/grayscale image as input and applies the following steps:

1. Divide the image into blocks.

2. Apply a discrete cosine transform (DCT) on each block to convert it into a frequency domain representation.

3. Quantize the DCT coefficients by dividing by the quantization matrix.

4. Perform zigzag scanning for each block.

5. Truncate each encode block to the specified number of coefficient parameters and return the encoded image

The JPEG decoder reverses the above steps to reconstruct the original image from the compressed data.

![Brown Pastel Flowchart Diagram Graph Template](https://user-images.githubusercontent.com/102947018/233417587-5d362e93-45d5-438d-add2-e1ed3a407128.png)

## LZW Compression

LZW is a compression algorithm that exploit the spatial and statistical redundancies in the data by replacing frequently occurring patterns with shorter codes. The LZW encoder in this project takes a grayscale image, block size and the maximum allowed dictionary size as input and applies the following steps:

1. Divide the image into blocks of the specified size (or treat the whole image as a single block if the block size is set to -1).

2. Flatten each block into a 1D array of pixel values.

3. Apply the LZW compression algorithm on each block to obtain the corresponding encoded output.

4. Store the encoded outputs

The LZW decoder in this project reverses the above steps to reconstruct the original image from the encoded data. It applies the LZW decoding algorithm to obtain the decoded output for each block and concatenates them to obtain the reconstructed image. The RMSE between the original and the reconstructed image is zero for lossless compression.

![Brown Pastel Flowchart Diagram Graph Template (1)](https://user-images.githubusercontent.com/102947018/233417223-a08ad368-f0a6-4a90-8426-1b1cd28adf23.png)

## Usage

- Requires Python 3.10+

- Install the required modules

    ```
    pip3 install -r requirements.txt
    ```

- Run the programs
    
    JPEG Compression

    ```
    python3 jpeg_compression.py
    ```

    LZW Compression

    ```
    python3 lzw_compression.py
    ```

## Results

The algorithms were tested for a data set of color and grayscale images. 

Sample output:

**JPEG Compression**

![Figure_1](https://user-images.githubusercontent.com/102947018/233422987-fea9d017-6ce3-433e-840b-70ae185848fc.png)

**LZW Compression**

![a](https://user-images.githubusercontent.com/102947018/233423176-8f5f8a78-e579-4f56-89e6-ffba0cb985fb.png)