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
        private byte[] screenData;
        private OpenCLBuffer<byte> render_buffer;

        public Texture renderTexture;

        private bool download;
        private OpenCLImage<int> climg;

        public GameProcessorCL(int width, int height) {
            program = new OpenCLProgram(KERNEL_FILE);
            download = !program.GLInteropAvailable;

            if (!download) {
                climg = new OpenCLImage<int>(program, width, height);
            }

            screenData = new byte[width * height];
            render_buffer = new OpenCLBuffer<byte>(program, screenData);
            drawKernel = new OpenCLKernel(program, "render");
            if (download) {
                drawKernel.SetArgument(0, render_buffer);
            } else {
                drawKernel.SetArgument(0, climg);
            }

            drawKernelWork = new long[] {width, height};

            renderTexture = new Texture();
            renderTexture.Construct();
            renderTexture.Width = width;
            renderTexture.Height = height;
            renderTexture.InternalFormat = PixelInternalFormat.R8;
            renderTexture.Format = PixelFormat.Red;
            renderTexture.Target = TextureTarget.Texture2D;
            renderTexture.PixelType = PixelType.UnsignedByte;
            renderTexture.SetFilters(TextureMinFilter.Nearest, TextureMagFilter.Nearest);
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