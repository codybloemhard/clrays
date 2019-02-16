using System;
using System.Diagnostics;
using clrays.Helpers;
using Cloo;
using FruckEngine.Graphics;
using FruckEngine.Helpers;
using OpenTK.Graphics.OpenGL;
using Template;

namespace clrays {
    public class GameProcessorCL {
        private const string KERNEL_FILE = "Assets/Kernels/raytrace.cl";

        private OpenCLProgram program;

        private OpenCLKernel drawKernel;
        private long[] drawKernelWork;
        private int[] screenData;
        private OpenCLBuffer<int> render_buffer;

        public Texture renderTexture;

        private bool download;
        private OpenCLImage<int> climg;

        public GameProcessorCL(int width, int height) {
            program = new OpenCLProgram(KERNEL_FILE);
            download = !program.GLInteropAvailable;

            if (!download) {
                climg = new OpenCLImage<int>(program, width, height);
            }

            screenData = new int[width * height];
            render_buffer = new OpenCLBuffer<int>(program, screenData);
            drawKernel = new OpenCLKernel(program, "render");
            if (download) {
                drawKernel.SetArgument(0, render_buffer);
            } else {
                drawKernel.SetArgument(0, climg);
            }
            drawKernel.SetArgument(1, (uint)width);
            drawKernel.SetArgument(2, (uint)height);

            drawKernelWork = new long[] {width, height};

            renderTexture = new Texture();
            renderTexture.Construct();
            renderTexture.Width = width;
            renderTexture.Height = height;
            renderTexture.InternalFormat = PixelInternalFormat.Rgba;
            renderTexture.Format = PixelFormat.Bgra;
            renderTexture.Target = TextureTarget.Texture2D;
            renderTexture.PixelType = PixelType.UnsignedByte;
            renderTexture.SetFilters(TextureMinFilter.Linear, TextureMagFilter.Linear);
            renderTexture.SetWrapping(TextureWrapMode.ClampToEdge, TextureWrapMode.ClampToEdge);
        }

        public void Render() {
            if (download) {
                drawKernel.Execute(drawKernelWork);
                render_buffer.CopyFromDevice();
                renderTexture.Bind();
                TextureHelper.LoadDataIntoTexture(renderTexture, renderTexture.Width, renderTexture.Height, screenData);
                renderTexture.Activate(0);
            } else {
                drawKernel.LockOpenGLObject(climg.texBuffer);
                drawKernel.Execute(drawKernelWork);
                drawKernel.UnlockOpenGLObject(climg.texBuffer);
                GL.BindTexture(TextureTarget.Texture2D, climg.OpenGLTextureID);
            }
        }
    }
}