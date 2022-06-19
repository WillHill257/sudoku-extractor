# Sudoku Extractor

#### William Hill (2115261) and Ireton Liu (2089889)

## Structure

- `sudoku-extractor.py` is the script
- `v2_test/` contains sample images
- `MnistSimpeCNN-master` contains the model used to do digit recognition on the extracted images (from <a href="https://github.com/ansh941/MnistSimpleCNN">ansh941</a> on GitHub)

## How to run

Perform the following steps:

- navigate to the directory with the `sudoku-extractor.py` script
- execute the script: `python3 sudoku-extractor.py`
- you will be prompted for the filename and path of the sudoku image to process

## Output

### Errors

- If an error occurs opening the file, you will be informed
- If the image is too noisy, contains colour shading, or is rotated, the extraction will fail (refer to the report for a discussion)

### Sudoku Grid

- A text-based grid will be printed. Note that some of the digits may be incorrect due to an approximate model being used for prediction
- The directory containing the extracted box images will be printed. These images will be saved as separate files and will have a name corresponding to the box's linearised index (in row-major order)
