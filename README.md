<h1>CUDA: Image-Processing</h1>
This repository contains my solutions to the problem sets of <code>Udacity's cs344 Intro to Parallel Programming</code> course.<hr/>
During the course, I mastered the fundamentals of massively parallel computing by using CUDA C to program modern GPUs. I learned the GPU programming model and architecture, key algorithms and parallel programming patterns, and optimization techniques. The course assignments (this repository contains my solutions) illustrate these concepts through image processing applications.
<h2>Problem Set 1: Converting Images to Greyscale</h2>
<h3>Intro</h3>
Colored digital images are represented by the <code>RGBA</code> format, where each channel measures between 0-255 such that 0 means that the color is fully absent, while 255 means that the color is fully saturated. The A (Alpha) represents that transparency of the image, which we do not need to change in order to convert an image to greyscale. To calculate the appropriate greyscale intensity of a colored pixel, we must use the following formula: 

<code>I = .299f * R + .587 * G + .114 * B</code>
<h3>Task</h3>
<ol>
<li><i>Configuring Kernel Launch</i>

First of all, each thread should process a single pixel. Furthermore, each block should process a row of pixels. Therefore:
<ul>
<li>Each block contains a number of threads equivalent to the width of the image (number of columns)</li>
<li>The grid contains a number of blocks equivalent to the heigtht of the image (number of rows)</li>
<li>Both the grid and the block are 1D</li>
</ul>
<br/>

```
void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const dim3 blockSize(numCols, 1, 1);
  const dim3 gridSize(numRows, 1, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
```
</li>
<li><i>GPU Function Converts RGBA to Greyscale</i>

The following function is run on each thread. It begins by determining the index that corresponds to the current thread/block location. From the <code>rgbaImage</code> array, it extracts the colored pixel at this index. Then, it converts the colored pixel to a greyscale intensity value. Finally, the intensity value is inserted into the <code>greyImage</code> array at the correct index.
<br/>

```
__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int current_column = threadIdx.x;
	int current_row = blockIdx.x;
	int index = current_row * blockDim.x + current_column;
	if(current_column < numCols && current_row < numRows){
		uchar4 colorPixel = rgbaImage[index];
		float intensity = .299f * colorPixel.x + .587f * colorPixel.y + .114f * colorPixel.z;
		greyImage[index] = intensity;
	}
}
```
</li>
</ol>

The full code for this problem set can be found in the folder labeled <code>Problem Set 1</code>. It is important to note that within this folder, I only wrote code for the <code>student_func.cu</code> file (I described the contents of this file above). The rest of the code was provided by the course and was not part of the assignment.
