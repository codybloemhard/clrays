﻿using System;
using System.Diagnostics;
using clrays.Helpers;
using Cloo;
using FruckEngine.Graphics;
using FruckEngine.Helpers;
using OpenTK.Graphics.OpenGL;
using Template;

namespace clrays {
    public class TraceProcessorCL {
        private OpenCLProgram program;

        private OpenCLKernel drawKernel;
        private readonly long[] drawKernelWork;
        private readonly int[] screenData;
        private OpenCLBuffer<int> render_buffer;

        public Texture renderTexture;

        private readonly bool download;
        private OpenCLImage<int> climg;

        private readonly OpenCLBuffer<int> scene_params;
        private readonly OpenCLBuffer<float> scene_items;

        public TraceProcessorCL(int width, int height, Scene scene, string kernel) {
            program = new OpenCLProgram(kernel);
            download = !program.GLInteropAvailable;

            if (!download) {
                climg = new OpenCLImage<int>(program, width, height);
            }

            screenData = new int[width * height];
            render_buffer = new OpenCLBuffer<int>(program, screenData);
            //make kernel and set args
            drawKernel = new OpenCLKernel(program, "render");
            if (download)
                drawKernel.SetArgument(0, render_buffer);
            else
                drawKernel.SetArgument(0, climg);

            //make scene buffers
            var scene_raw = scene.GetBuffers();
            var scene_params_raw = scene.GetParamsBuffer();
            scene_params = new OpenCLBuffer<int>(program, scene_params_raw);
            scene_items = new OpenCLBuffer<float>(program, scene_raw);
            //set constants
            drawKernel.SetArgument(1, (uint)width);
            drawKernel.SetArgument(2, (uint)height);
            //constants
            //(uint)(scene_spheres.Length / Scene.sphereSize
            drawKernel.SetArgument(3, scene_params);
            drawKernel.SetArgument(4, scene_items);
            //work
            drawKernelWork = new long[] {width, height};
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
            if (download) {
                drawKernel.Execute(drawKernelWork, events);
                render_buffer.CopyFromDevice();
                renderTexture.Bind();
                TextureHelper.LoadDataIntoTexture(renderTexture, renderTexture.Width, renderTexture.Height, screenData);
                renderTexture.Activate(0);
            } else {
                drawKernel.LockOpenGLObject(climg.texBuffer);
                drawKernel.Execute(drawKernelWork, events);
                drawKernel.UnlockOpenGLObject(climg.texBuffer);
                GL.BindTexture(TextureTarget.Texture2D, climg.OpenGLTextureID);
            }
        }
    }
}