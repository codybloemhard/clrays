using System;
using System.Diagnostics;
using clrays.Helpers;
using Cloo;
using FruckEngine.Graphics;
using FruckEngine.Helpers;
using OpenTK.Graphics.OpenGL;
using Template;

namespace clrays {
    public class TraceProcessorCL {
        private readonly OpenCLProgram program;

        private OpenCLKernel drawKernel;
        private readonly long[] drawKernelWork;
        private readonly float[] floatmap;
        private OpenCLBuffer<float> render_buffer;
        private readonly OpenCLBuffer<int> scene_params;
        private readonly OpenCLBuffer<float> scene_items;

        private OpenCLKernel clearKernel;
        private readonly long[] clearKernelWork;
        //private readonly int[] imagemap;
        //private OpenCLBuffer<int> image_buffer;
        public Texture renderTexture;

        public TraceProcessorCL(int width, int height, uint AA, Scene scene) {
            program = new OpenCLProgram("Assets/Kernels/raytrace.cl");
            floatmap = new float[width * height * 3];
            render_buffer = new OpenCLBuffer<float>(program, floatmap);
            //make kernel and set args
            drawKernel = new OpenCLKernel(program, "render");
            drawKernel.SetArgument(0, render_buffer);
            //make scene buffers
            var scene_raw = scene.GetBuffers();
            var scene_params_raw = scene.GetParamsBuffer();
            scene_params = new OpenCLBuffer<int>(program, scene_params_raw);
            scene_items = new OpenCLBuffer<float>(program, scene_raw);
            //set constants
            drawKernel.SetArgument(1, (uint)width);
            drawKernel.SetArgument(2, (uint)height);
            drawKernel.SetArgument(3, AA);
            //set arrays
            drawKernel.SetArgument(4, scene_params);
            drawKernel.SetArgument(5, scene_items);
            //work
            drawKernelWork = new long[] {width*AA, height*AA};
            // clear kernel
            clearKernel = new OpenCLKernel(program, "clear");
            clearKernel.SetArgument(0, render_buffer);
            clearKernel.SetArgument(1, (uint)width);
            clearKernel.SetArgument(2, (uint)height);
            clearKernelWork = new long[] { width, height };
            //texture
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
            var events = new ComputeEventList();
            clearKernel.Execute(clearKernelWork, events);
            drawKernel.Execute(drawKernelWork, events);
            render_buffer.CopyFromDevice();
            renderTexture.Bind();
            TextureHelper.LoadDataIntoTexture(renderTexture, renderTexture.Width, renderTexture.Height, floatmap);
            renderTexture.Activate(0);
        }
    }
}