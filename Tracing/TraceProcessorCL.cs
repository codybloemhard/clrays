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

        public Texture renderTexture;

        private readonly bool download;
        //private OpenCLImage<int> climg;

        private readonly OpenCLBuffer<int> scene_params;
        private readonly OpenCLBuffer<float> scene_items;

        public TraceProcessorCL(int width, int height, uint AA, Scene scene, string kernel) {
            program = new OpenCLProgram(kernel);
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
            //constants
            //(uint)(scene_spheres.Length / Scene.sphereSize
            drawKernel.SetArgument(4, scene_params);
            drawKernel.SetArgument(5, scene_items);
            //work
            drawKernelWork = new long[] {width*AA, height*AA};
            //upload buffers
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
            drawKernel.Execute(drawKernelWork, events);
            render_buffer.CopyFromDevice();
            renderTexture.Bind();
            TextureHelper.LoadDataIntoTexture(renderTexture, renderTexture.Width, renderTexture.Height, floatmap);
            renderTexture.Activate(0);
        }
    }
}